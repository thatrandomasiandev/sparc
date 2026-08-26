"""Shared preference-RL training loop for PEBBLE and PrefPPO."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

from sparc.baselines.config import BaselineConfig
from sparc.baselines.pebble_reward import PebbleRewardModel
from sparc.baselines.relabel import relabel_sac_replay_buffer
from sparc.env.factory import make_env
from sparc.env.specs import get_env_spec
from sparc.harness.env_defaults import apply_env_defaults
from sparc.harness.metrics import BaselineRunResult, EvalMetrics
from sparc.logging.jsonl import JsonlLogger


class LearnedRewardWrapper(gym.Wrapper):
    """Relabel step rewards from the learned preference reward model."""

    def __init__(self, env: gym.Env, reward_model: PebbleRewardModel) -> None:
        super().__init__(env)
        self.reward_model = reward_model
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.reward_model.start_trajectory()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_obs is None:
            raise RuntimeError("reset() must precede step()")
        obs, true_r, term, trunc, info = self.env.step(action)
        action_arr = _as_action(self.env, action)
        self.reward_model.add_transition(self._last_obs, action_arr, float(true_r))
        r_hat = self.reward_model.r_hat(self._last_obs, action_arr)
        self._last_obs = obs
        info["true_reward"] = true_r
        return obs, r_hat, term, trunc, info


def _action_dim(env: gym.Env) -> int:
    space = env.action_space
    if hasattr(space, "shape") and space.shape is not None and len(space.shape) > 0:
        return int(space.shape[0])
    return 1


def _as_action(env: gym.Env, action: np.ndarray | int) -> np.ndarray:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return np.array([float(int(action))], dtype=np.float32)
    return np.asarray(action, dtype=np.float32)


def _evaluate(
    env: gym.Env,
    model: SAC | PPO,
    reward_model: PebbleRewardModel,
    num_episodes: int,
) -> EvalMetrics:
    returns: list[float] = []
    true_returns: list[float] = []
    successes: list[float] = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        ep_true = 0.0
        success = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, true_r, term, trunc, info = env.step(action)
            learned = reward_model.r_hat(obs, _as_action(env, action))
            ep_r += learned
            ep_true += float(true_r)
            success = max(success, float(info.get("success", 0.0)))
            done = term or trunc
        returns.append(ep_r)
        true_returns.append(ep_true)
        successes.append(success)
    return EvalMetrics(
        env_steps=0,
        mean_return=float(np.mean(returns)),
        mean_true_return=float(np.mean(true_returns)),
        total_queries=reward_model.total_labels,
        success_rate=float(np.mean(successes)) if successes else None,
    )


def _make_reward_model(cfg: BaselineConfig, obs_dim: int, action_dim: int) -> PebbleRewardModel:
    spec = get_env_spec(cfg.env_id)
    seg_len = spec.segment_horizon if spec else cfg.segment_length
    if cfg.env_id == "CartPole-v1":
        seg_len = min(seg_len, 10)
    return PebbleRewardModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        segment_length=seg_len,
        ensemble_size=cfg.ensemble_size,
        lr=cfg.reward_lr,
        mb_size=cfg.reward_batch,
        large_batch=cfg.large_batch,
        activation=cfg.activation,
        reward_update=cfg.reward_update,
        teacher_gamma=cfg.teacher_gamma,
        teacher_eps_mistake=cfg.teacher_eps_mistake,
        teacher_eps_skip=cfg.teacher_eps_skip,
        teacher_eps_equal=cfg.teacher_eps_equal,
        seed=cfg.seed,
    )


def _maybe_query(
    effective_step: int,
    cfg: BaselineConfig,
    reward_model: PebbleRewardModel,
    unsup_end: int,
    steps_since_query: int,
    total_feedback: int,
    first_query_done: bool,
    logger: JsonlLogger,
    model: SAC | PPO | None = None,
    base_env: gym.Env | None = None,
    learn_steps: int = 0,
) -> tuple[int, int, float, bool]:
    """Issue preference queries. ``num_interact`` is in env steps (PEBBLE / B-Pref)."""
    final_acc = 0.0
    updated = False
    if not first_query_done and effective_step >= unsup_end:
        labeled = reward_model.sample_preferences(feed_type=0)
        total_feedback += labeled
        final_acc = reward_model.train()
        steps_since_query = 0
        first_query_done = True
        updated = True
        logger.log("reward_update", step=effective_step, acc=final_acc, first=True)
    elif first_query_done and total_feedback < cfg.max_feedback:
        steps_since_query += learn_steps
        if steps_since_query >= cfg.num_interact:
            labeled = reward_model.sample_preferences(feed_type=cfg.feed_type)
            total_feedback += labeled
            final_acc = reward_model.train()
            steps_since_query = 0
            updated = True
            logger.log(
                "reward_update",
                step=effective_step,
                acc=final_acc,
                total_feedback=total_feedback,
            )
    if updated and isinstance(model, SAC) and base_env is not None:
        relabeled = relabel_sac_replay_buffer(model, reward_model, base_env)
        logger.log("replay_relabel", step=effective_step, transitions=relabeled)
    return steps_since_query, total_feedback, final_acc, first_query_done


def _random_steps(env: gym.Env, reward_model: PebbleRewardModel, n: int) -> int:
    obs, _ = env.reset()
    for _ in range(n):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
    return n


def _run_learned_policy(
    cfg: BaselineConfig,
    method: str,
    logger: JsonlLogger,
    policy_cls: type[SAC] | type[PPO],
    wrapper_cls: type[LearnedRewardWrapper] = LearnedRewardWrapper,
    wrapper_kwargs: dict[str, Any] | None = None,
    reward_model_factory: Callable[[BaselineConfig, int, int], PebbleRewardModel] | None = None,
) -> BaselineRunResult:
    base_env = make_env(cfg.env_id, seed=cfg.seed)
    shape = base_env.observation_space.shape
    obs_dim = int(np.prod(shape)) if shape is not None else 1
    action_dim = _action_dim(base_env)
    factory = reward_model_factory or _make_reward_model
    reward_model = factory(cfg, obs_dim, action_dim)
    env = wrapper_cls(base_env, reward_model, **(wrapper_kwargs or {}))

    if policy_cls is SAC:
        model: SAC | PPO = SAC(
            "MlpPolicy",
            env,
            learning_rate=cfg.policy_lr,
            buffer_size=min(cfg.buffer_size, max(cfg.num_train_steps * 2, 1000)),
            learning_starts=min(cfg.learning_starts, cfg.num_seed_steps),
            batch_size=cfg.batch_size,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            gamma=cfg.gamma,
            seed=cfg.seed,
            verbose=0,
        )
        chunk = 128
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.policy_lr,
            n_steps=128,
            batch_size=64,
            gamma=cfg.gamma,
            seed=cfg.seed,
            verbose=0,
        )
        chunk = 128

    t0 = time.perf_counter()
    steps_since_query = 0
    total_feedback = 0
    eval_history: list[EvalMetrics] = []
    final_acc = 0.0
    unsup_end = cfg.num_seed_steps + cfg.num_unsup_steps
    first_query_done = False
    last_eval_bucket = -1

    if cfg.num_seed_steps > 0:
        _random_steps(env, reward_model, cfg.num_seed_steps)

    while cfg.num_seed_steps + model.num_timesteps < cfg.num_train_steps:
        remaining = cfg.num_train_steps - cfg.num_seed_steps - model.num_timesteps
        learn_steps = min(chunk, remaining)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)
        effective_step = cfg.num_seed_steps + model.num_timesteps

        eval_bucket = effective_step // cfg.eval_frequency if cfg.eval_frequency > 0 else -1
        if cfg.eval_frequency > 0 and eval_bucket > last_eval_bucket and effective_step > 0:
            last_eval_bucket = eval_bucket
            metrics = _evaluate(base_env, model, reward_model, cfg.num_eval_episodes)
            eval_history.append(
                EvalMetrics(
                    effective_step,
                    metrics.mean_return,
                    metrics.mean_true_return,
                    metrics.total_queries,
                )
            )
            logger.log("eval", **eval_history[-1].__dict__)

        steps_since_query, total_feedback, acc, first_query_done = _maybe_query(
            effective_step,
            cfg,
            reward_model,
            unsup_end,
            steps_since_query,
            total_feedback,
            first_query_done,
            logger,
            model=model if policy_cls is SAC else None,
            base_env=base_env,
            learn_steps=learn_steps,
        )
        if acc:
            final_acc = acc

        if effective_step > 0 and effective_step % max(cfg.eval_frequency, 10_000) < learn_steps:
            logger.log(
                "train_progress",
                step=effective_step,
                total_feedback=total_feedback,
                wall_clock_sec=time.perf_counter() - t0,
            )

    elapsed = time.perf_counter() - t0
    total_env_steps = cfg.num_seed_steps + model.num_timesteps
    return BaselineRunResult(
        method=method,
        seed=cfg.seed,
        env_id=cfg.env_id,
        total_queries=reward_model.total_labels,
        total_env_steps=total_env_steps,
        wall_clock_sec=elapsed,
        eval_history=eval_history,
        final_train_acc=final_acc,
    )


def run_preference_baseline(
    cfg: BaselineConfig,
    method: str,
    *,
    wrapper_cls: type[LearnedRewardWrapper] = LearnedRewardWrapper,
    wrapper_kwargs: dict[str, Any] | None = None,
    reward_model_factory: Callable[[BaselineConfig, int, int], PebbleRewardModel] | None = None,
    policy_cls: type[SAC] | type[PPO] | None = None,
) -> BaselineRunResult:
    """PEBBLE (SAC), PrefPPO (PPO), RUNE, SURF, or APReL with shared query schedule."""
    cfg = apply_env_defaults(cfg)
    set_random_seed(cfg.seed)
    log_path = Path(cfg.log_path) if cfg.log_path else None
    logger = JsonlLogger(path=log_path)
    logger.log("baseline_start", method=method, config=cfg.to_dict())

    if policy_cls is None:
        policy_cls = PPO if method == "prefppo" else SAC

    result = _run_learned_policy(
        cfg,
        method,
        logger,
        policy_cls,
        wrapper_cls=wrapper_cls,
        wrapper_kwargs=wrapper_kwargs,
        reward_model_factory=reward_model_factory,
    )

    logger.log(
        "baseline_end",
        method=method,
        total_queries=result.total_queries,
        wall_clock_sec=result.wall_clock_sec,
    )
    logger.close()
    return result
