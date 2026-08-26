"""Integrated SPARC training loop — wires all four pillars."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from sparc.annotator_sim import SyntheticAnnotator
from sparc.drift_detector import DriftDetector, DriftDetectorConfig
from sparc.engine.config import TrainConfig
from sparc.engine.env_rollout import collect_rollouts, evaluate_random_policy, infer_env_dims
from sparc.engine.policy_trainer import SparcPolicyTrainer
from sparc.env.buffer import SegmentBuffer
from sparc.env.dataset import PreferenceDataset
from sparc.env.factory import make_env
from sparc.env.segments import TrajectorySegment
from sparc.env.specs import get_env_spec
from sparc.harness.metrics import EvalMetrics
from sparc.logging.jsonl import JsonlLogger
from sparc.policy import BoundingConfig, ConfidenceGate
from sparc.query_optimizer import CommsWindowOptimizer, QueryCandidate
from sparc.query_optimizer.batch import BatchSelectionResult, operator_for_window
from sparc.reward_model import RewardEnsemble


@dataclass
class TrainResult:
    windows_completed: int
    total_queries: int
    total_labels: int
    drift_triggers: int
    wall_clock_sec: float
    final_reward_loss: float
    total_env_steps: int = 0
    eval_history: list[EvalMetrics] = field(default_factory=list)


def _random_rollout(
    rng: np.random.Generator,
    state_dim: int,
    action_dim: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = rng.normal(size=(length, state_dim))
    speed_idx = min(3, state_dim - 1)
    states[:, speed_idx] = np.abs(states[:, speed_idx])
    actions = rng.normal(size=(length, action_dim)) * 0.1
    return states.astype(np.float64), actions.astype(np.float64)


def _fill_buffer(
    buffer: SegmentBuffer,
    rng: np.random.Generator,
    state_dim: int,
    action_dim: int,
    rollout_length: int,
    num_rollouts: int,
    env=None,
    policy=None,
) -> None:
    if env is not None:
        collect_rollouts(
            env,
            buffer,
            rng,
            total_steps=rollout_length * num_rollouts,
            policy=policy,
        )
        return
    for _ in range(num_rollouts):
        states, actions = _random_rollout(rng, state_dim, action_dim, rollout_length)
        buffer.add_trajectory(states, actions)


def _segments_to_candidates(segments: list[TrajectorySegment]) -> list[QueryCandidate]:
    return [
        QueryCandidate(segment_0=segments[i], segment_1=segments[i + 1])
        for i in range(len(segments) - 1)
    ]


class SparcTrainer:
    """One-engine orchestrator: Pillar 1–4 + comms windows + JSONL audit trail."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.logger = JsonlLogger(
            path=Path(config.log_path) if config.log_path else None,
        )
        self._init_components()

    def _init_components(self) -> None:
        cfg = self.config
        self.env = None
        feature_layout = "rover"

        if cfg.env_id:
            spec = get_env_spec(cfg.env_id)
            self.env = make_env(cfg.env_id, seed=cfg.seed)
            state_dim, action_dim = infer_env_dims(self.env)
            cfg = replace(cfg, state_dim=state_dim, action_dim=action_dim)
            self.config = cfg
            feature_layout = spec.layout if spec else "generic"

        if cfg.regime == "stress":
            self.annotator = SyntheticAnnotator.stress_regime(
                train_steps=cfg.train_steps_total,
                seed=cfg.seed,
            )
        else:
            self.annotator = SyntheticAnnotator.easy_regime(seed=cfg.seed)
        self.annotator.feature_layout = feature_layout

        reward_cfg = cfg.reward
        if cfg.share_operator_latents:
            # Pillar 1 ablation: single shared latent; multi-op labels remapped to op 1
            reward_cfg = replace(reward_cfg, num_operators=1)

        self.ensemble = RewardEnsemble.create(
            cfg.state_dim,
            cfg.action_dim,
            cfg.horizon,
            config=reward_cfg,
            seed=cfg.seed,
        )
        self.drift: DriftDetector | None
        if cfg.enable_drift_detector:
            self.drift = DriftDetector(
                ensemble=self.ensemble,
                config=DriftDetectorConfig(),
                operator_ids=list(range(1, reward_cfg.num_operators + 1)),
            )
        else:
            self.drift = None
        self.optimizer = CommsWindowOptimizer(
            ensemble=self.ensemble,
            config=cfg.query,
            drift_detector=self.drift,
            num_operators=cfg.num_operators,
        )
        self.gate = ConfidenceGate(
            ensemble=self.ensemble,
            config=BoundingConfig(mode=cfg.bounding_mode),
            operator_id=cfg.operator_id,
        )
        self.buffer = SegmentBuffer(horizon=cfg.horizon)
        self.dataset = PreferenceDataset()
        self.policy_trainer: SparcPolicyTrainer | None = None
        if cfg.env_id and cfg.train_policy and self.env is not None:
            self.policy_trainer = SparcPolicyTrainer(
                self.env,
                self.ensemble,
                self.gate,
                cfg,
            )

    def run(self) -> TrainResult:
        cfg = self.config
        t0 = time.perf_counter()
        self.logger.log("train_start", config=cfg.to_dict())

        total_queries = 0
        total_labels = 0
        drift_triggers = 0
        final_loss = 0.0
        env_step = 0
        windows_completed = 0
        eval_history: list[EvalMetrics] = []

        try:
            for window_id in range(cfg.num_windows):
                if cfg.regime == "easy":
                    op_id = 2  # EXPERIMENTS.md easy regime: single Op-2
                else:
                    op_id = operator_for_window(window_id, cfg.num_operators)
                model_op_id = 1 if cfg.share_operator_latents else op_id
                self.gate.operator_id = model_op_id
                if self.policy_trainer is not None:
                    self.policy_trainer.set_operator(model_op_id)
                self.buffer.clear()
                policy_model = self.policy_trainer.model if self.policy_trainer else None
                _fill_buffer(
                    self.buffer,
                    self.rng,
                    cfg.state_dim,
                    cfg.action_dim,
                    rollout_length=cfg.horizon * 4,
                    num_rollouts=8,
                    env=self.env,
                    policy=policy_model,
                )
                candidates = _segments_to_candidates(self.buffer.iter_segments())
                if len(candidates) < cfg.query.batch_size:
                    _fill_buffer(
                        self.buffer,
                        self.rng,
                        cfg.state_dim,
                        cfg.action_dim,
                        rollout_length=cfg.horizon * 6,
                        num_rollouts=12,
                        env=self.env,
                        policy=policy_model,
                    )
                    candidates = _segments_to_candidates(self.buffer.iter_segments())

                batch_result: BatchSelectionResult
                if cfg.max_queries is not None and total_queries >= cfg.max_queries:
                    batch_result = BatchSelectionResult(
                        queries=[],
                        wall_clock_sec=0.0,
                        operator_id=op_id,
                        window_id=window_id,
                    )
                else:
                    effective_batch = cfg.query.batch_size
                    if cfg.max_queries is not None:
                        remaining = cfg.max_queries - total_queries
                        effective_batch = min(effective_batch, remaining)
                    batch_result = self.optimizer.on_window_open(
                        window_id=window_id,
                        candidates=candidates,
                        env_step=env_step,
                        operator_id=model_op_id,
                        batch_size=effective_batch,
                    )
                total_queries += len(batch_result.queries)
                self.logger.log(
                    "window_open",
                    window_id=window_id,
                    operator_id=op_id,
                    num_queries=len(batch_result.queries),
                    select_sec=batch_result.wall_clock_sec,
                    env_step=env_step,
                )

                window_records = []
                for q in batch_result.queries:
                    for drift_ev in self.annotator.apply_drift_at(q.env_step):
                        self.logger.log(
                            "scripted_drift",
                            operator_id=drift_ev.operator_id,
                            env_step=drift_ev.env_step,
                            query_env_step=q.env_step,
                        )
                    # Annotator uses rotating op; shared-latent ablation remaps for RM train
                    label_q = replace(q, operator_id=op_id) if cfg.share_operator_latents else q
                    label = self.annotator.label_query(label_q)
                    rec = self.dataset.add_query(q, label)
                    window_records.append(rec)
                    self.logger.log(
                        "label_received",
                        window_id=window_id,
                        operator_id=op_id,
                        label=label,
                        env_step=q.env_step,
                    )
                    if label is not None:
                        total_labels += 1

                events = (
                    self.drift.update_with_labels(window_records) if self.drift is not None else []
                )
                for ev in events:
                    drift_triggers += 1
                    self.logger.log(
                        "drift_trigger",
                        operator_id=ev.operator_id,
                        regime=ev.regime,
                        log_lambda=ev.log_lambda,
                    )

                for _ in range(cfg.train_epochs_per_window):
                    final_loss = self.ensemble.train_epoch(self.dataset)
                self.logger.log(
                    "reward_train",
                    window_id=window_id,
                    loss=final_loss,
                    dataset_size=len(self.dataset.labeled()),
                )

                segments = self.buffer.iter_segments()
                tau = self.gate.update_threshold(segments)
                self.logger.log("gate_update", window_id=window_id, q90=self.gate.q90, tau=tau)

                if segments:
                    seg = segments[0]
                    r_hat = self.ensemble.pooled_reward(seg, model_op_id)
                    bound = self.gate.bound_reward(seg, r_hat)
                    self.logger.log(
                        "bounding_applied",
                        window_id=window_id,
                        r_hat=r_hat,
                        disagreement=self.gate.disagreement(seg),
                        action=bound.action.name,
                        reward_out=bound.reward,
                    )

                self.optimizer.on_labels_received()
                env_step += cfg.env_steps_per_window
                windows_completed += 1
                self.logger.log("window_close", window_id=window_id, env_step=env_step)

                if self.policy_trainer is not None:
                    self.policy_trainer.train(cfg.policy_steps_per_window)
                    self.logger.log(
                        "policy_train",
                        window_id=window_id,
                        steps=cfg.policy_steps_per_window,
                    )

                if (
                    self.env is not None
                    and cfg.eval_every_windows > 0
                    and windows_completed % cfg.eval_every_windows == 0
                ):
                    if self.policy_trainer is not None:
                        mean_r, mean_true = self.policy_trainer.evaluate(
                            cfg.num_eval_episodes,
                            cfg.seed + window_id,
                        )
                    else:
                        mean_r, mean_true = evaluate_random_policy(
                            self.env,
                            cfg.num_eval_episodes,
                            cfg.seed + window_id,
                        )
                    eval_history.append(
                        EvalMetrics(
                            env_steps=env_step,
                            mean_return=mean_r,
                            mean_true_return=mean_true,
                            total_queries=total_queries,
                        )
                    )
                    self.logger.log("eval", **eval_history[-1].__dict__)

                if env_step >= cfg.train_steps_total:
                    break

        finally:
            elapsed = time.perf_counter() - t0
            self.logger.log(
                "train_end",
                windows_completed=windows_completed,
                total_queries=total_queries,
                drift_triggers=drift_triggers,
                wall_clock_sec=elapsed,
            )
            self.logger.close()

        return TrainResult(
            windows_completed=windows_completed,
            total_queries=total_queries,
            total_labels=total_labels,
            drift_triggers=drift_triggers,
            wall_clock_sec=elapsed,
            final_reward_loss=final_loss,
            total_env_steps=env_step,
            eval_history=eval_history,
        )
