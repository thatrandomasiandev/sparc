# CHANGES

Versioned log of spec and behavior changes. STAR-bar discipline: nothing hidden between releases.

Format: `[version/date] file — summary`

---

## Spec revisions (pre-implementation)

### 2026-08-25 — Phase 0 scope lock

- **`PROBLEM.md`** — Locked primary scenario (URC rover, delayed multi-operator comms), falsifiable success criterion (≤20% PEBBLE queries), non-goals.
- **`docs/prior-art-matrix.md`** — Filled PbRL prior-art comparison (9 methods + SPARC gap row).
- **`docs/field-deployment-memo.md`** — Two-page memo on lab-vs-field PbRL failure modes; scenario #1 locked.
- **`docs/aprel-friction-log.md`** — APReL install deferred; expected friction documented from docs review.

### 2026-08-25 — Phase 1 initial design (v1)

- **`DESIGN.md` v1** — Feedback model, four pillars (pseudocode), APReL comparison, red-team notes.
- **`EXPERIMENTS.md` v1** — Environments, synthetic annotators, baselines, metrics, statistical tests.

### 2026-08-25 — Phase 1 sanity-check revision (v2)

- **`EXPERIMENTS.md` v2** — Fixed ΔT to sim mission time only (22.5 ep/window); rover Train steps 750,000; Drift-1 at fixed fraction \(f_{\mathrm{drift}}=0.5\) per env.
- **`DESIGN.md` v2** — Pillar 3 τ from rolling replay-buffer P90 (per-window cadence); EM-style alternation for \(z_i\); Pillar 2 EIG implementation note (Newton on \(w_i\), no retrain per candidate).

### 2026-08-25 — Phase 1 alignment pass (v3)

- **`DESIGN.md` v3** — Comms window locked to sim mission time; tie/skip handling aligned with `EXPERIMENTS.md`.
- **`PROBLEM.md`** — Cross-references updated; authority over downstream specs confirmed.
- **`docs/week6-external-review-packet.md`** — Reviewer handoff for LIRA/SLURM.

---

## Implementation (Phase 2+)

### 2026-08-25 — PEBBLE interact schedule fix + Week 25 ablations

- **`sparc/baselines/loop.py`** — `num_interact` now counted in **env steps** (was per 128-step learn chunk); eval bucket crossing; `train_progress` logs; `reward_batch` used as `mb_size`.
- **Week 25 ablations** — `share_operator_latents`, `query.selection_mode=random`, `bounding_mode=pass-through`, `enable_drift_detector=false` + configs under `experiments/configs/sparc_walker_ablate_p{1-4}*.json`.
- **`sparc/harness/stress_verify.py`** — scripted Drift-1 + SPRT triggers; trainer logs `scripted_drift`.
- **Phase 4 pilot** — restarted after abort (`phase4_walker_pilot`).

### 2026-08-25 — earlier Phase 4 / Week 19 notes

- **`sparc/harness/drift_regret.py`** — post-drift regret proxy on eval history; attached to all harness runs.
- **Tests** — 82 passing (+3 drift regret).

### 2026-08-25 — Phase 4 prep: metrics, manifests, RESULTS.md scaffold

- **`sparc/harness/log_metrics.py`** — wall-clock per query from JSONL (`sparc metrics --log ...`).
- **`sparc/harness/rm_accuracy.py`** — held-out pairwise reward-model accuracy (EXPERIMENTS.md §2).
- **`experiments/configs/phase4_stress_{walker,rover}_primary.json`** — 10-seed primary gate manifests.
- **`experiments/configs/pebble_rover_stress.json`** — PEBBLE reference for rover-nav stress.
- **`RESULTS.md`** — Phase 4 scaffold with reproduction commands and gate checklist.
- **Tests** — +3 log/accuracy metric tests.

### 2026-08-25 — Phase 3 Week 19 harness dry-run suite

- **`sparc/harness/dry_run.py`** — manifest-driven suite runner for all methods + `suite_summary.json`.
- **CLI** — `sparc suite --manifest ... --output-dir ... [--seeds 0,1]`
- **`experiments/configs/dry_run_{walker_stress_smoke,pendulum_ci}.json`** — Week 19 rehearsal manifests.
- Smoke configs for prefppo/surf/rune/aprel on walker-walk stress scale.
- **Tests** — 76 passing (+2 suite tests; pendulum CI ~39s, walker suite optional).

### 2026-08-25 — Phase 3 Week 17 SURF, RUNE, APReL baselines

- **`sparc/baselines/rune.py`** — RUNE exploration bonus (β · ensemble std) on PEBBLE SAC.
- **`sparc/baselines/surf_reward.py`**, **`surf.py`** — semi-supervised pseudo-label + segment augmentation.
- **`sparc/baselines/aprel_sampling.py`**, **`aprel.py`** — disagreement + greedy medoid batch (`feed_type=2`).
- **CLI** — `sparc benchmark --method surf|rune|aprel --config ...`
- **`experiments/configs/{surf,rune,aprel}_walker_walk.json`** — B-Pref-scale configs.
- **Tests** — 74 passing (+5 Week 17 smoke / sampling tests).

### 2026-08-25 — PEBBLE SAC replay-buffer relabeling + query schedule fix

- **`sparc/baselines/relabel.py`** — overwrite stale SAC replay rewards after each ensemble update.
- **`sparc/baselines/loop.py`** — relabel hook in `_maybe_query`; fix effective-step counting (seed + policy timesteps) and first-query trigger (`>= unsup_end`, not `==`).
- **`experiments/configs/{sparc,pebble}_walker_stress_smoke.json`** — CI-scale head-to-head stress dry run.
- **Tests** — 69 passing (+2 relabel / query-schedule regression).

### 2026-08-25 — EXPERIMENTS.md stress-regime schedule + query cap

- **`sparc/harness/stress_defaults.py`** — ΔT=900s comms windows, B=8, N_SPARC=200; `auto_stress_schedule` on `TrainConfig`.
- **`sparc/env/specs.py`** — per-env `sim_dt` for comms step math (rover 0.05s, DMC 0.025s).
- **SparcTrainer** — enforces `max_queries`; stops at `train_steps_total`.
- **`experiments/configs/sparc_{rover,walker}_stress*.json`**, **`pebble_walker_stress.json`** — Phase 4 primary comparison configs.
- **Tests** — 67 passing (+5 stress schedule / query cap).

### 2026-08-25 — SPARC SAC policy integration + compare CLI

- **`sparc/engine/reward_wrapper.py`** — rolling-segment SPARC rewards + Pillar 3 bounding for SB3.
- **`sparc/engine/policy_trainer.py`** — SAC trains on bounded SPARC rewards each comms window.
- **SparcTrainer** — policy rollouts for query buffer; deterministic policy eval.
- **`sparc/harness/compare.py`** + **`sparc compare --a ... --b ...`** — Welch t-test + query reduction.

### 2026-08-25 — SPARC wired into multi-seed harness

- **`sparc/harness/sparc_runner.py`** — `run_sparc()` returns `BaselineRunResult` for head-to-head comparison.
- **`sparc/engine/env_rollout.py`** — real Gymnasium rollouts into segment buffer.
- **SparcTrainer** — optional `env_id`; stress/easy regimes on rover-nav + DMC envs.
- **CLI** — `sparc benchmark --method sparc --config ... [--seeds all]`.
- **`experiments/configs/sparc_rover_stress_smoke.json`**.

### 2026-08-25 — Phase 3 multi-seed benchmark runner

- **`sparc/harness/multi_seed.py`** — `run_multi_seed_baseline()` across EXPERIMENTS.md master seeds.
- **`sparc/harness/stats.py`** — 95% CI, Welch t-test, seed summaries.
- **CLI** — `--seeds 0,1,2|all` and `--output-dir` on `sparc benchmark`.
- **`experiments/configs/pebble_walker_walk_10seed.json`** — full 10-seed reproduction config.
- **scipy** added to dev deps for statistical tests.

### 2026-08-25 — Phase 3 Week 16 DMControl + walker-walk

- **`sparc/env/dmc.py`**, **`sparc/env/specs.py`** — Shimmy DMControl wrapper (flattened state, 1000-step episodes) and EXPERIMENTS task registry.
- **`experiments/configs/pebble_walker_walk.json`** — Full B-Pref-scale PEBBLE config (500k steps, 1000 queries, H=50).
- **`experiments/configs/pebble_walker_walk_smoke.json`** — CI-scale smoke config.
- **Optional dep** `[dmc]` → `shimmy[dm-control]==2.0.0`.
- **Tests** — 53 passing (+5 DMC / walker-walk smoke).

### 2026-08-25 — Phase 3 Week 16 PEBBLE/PrefPPO baseline harness

- **`sparc/baselines/`** — B-Pref-style ensemble reward model, disagreement sampling, SAC (PEBBLE) and PPO (PrefPPO) trainers.
- **`sparc/harness/`** — Shared `run_baseline()` entry point and metrics types.
- **`sparc/env/factory.py`**, **`sparc/env/rover_nav.py`** — Env factory + minimal rover-nav proxy.
- **CLI** — `sparc benchmark --method pebble|prefppo --config ...`
- **Tests** — 48 passing (+5 baseline/harness smoke tests).

### 2026-08-25 — Phase 2 Week 15 hardening pass

- **Version** bumped to `0.1.0a0` (PEP 440 alpha; tag target `v0.1.0-alpha`).
- **mypy** — fixed 8 type errors across logging, annotator, reward model, EIG; added mypy step to CI.
- **No open TODOs** in `sparc/`; full suite green (43 tests).

### 2026-08-25 — Phase 2 Week 14 integration + CLI

- **`sparc/engine/trainer.py`** — `SparcTrainer` wires all four pillars + comms windows.
- **`sparc/logging/jsonl.py`** — structured JSONL audit log (queries, reward train, gate, drift).
- **`sparc/cli/main.py`** — `sparc train --config ... [--log ...]`.
- **`experiments/configs/smoke_easy.json`** — toy smoke config.
- **`tests/test_e2e_smoke.py`** — full pipeline < 2 min.

### 2026-08-25 — Phase 2 Week 13 drift detector (Pillar 4)

- **`sparc/drift_detector/`** — SPRT per operator (α=0.05, β=0.10), drift triggers, query history for re-query pool.
- **`sparc/query_optimizer/requery.py`** — `select_batch_with_requery` (⌈B/2⌉ from top-disagreement history).
- **`CommsWindowOptimizer`** — optional `drift_detector` integration on window open.
- **`tests/test_drift_detector.py`** — 5 tests.

### 2026-08-25 — Phase 2 Week 12 policy bounding (Pillar 3)

- **`sparc/policy/`** — `ApplyBounding` (pass-through/shrink/freeze), rolling P90 threshold, `ConfidenceGate`, minibatch skip at f_max.
- **`tests/test_policy_bounding.py`** — 9 analytic toy-case tests.

### 2026-08-25 — Phase 2 Week 11 query optimizer (Pillar 2)

- **`sparc/query_optimizer/`** — EIG scoring with Newton `PredictPrefProb`, greedy `select_batch`, `CommsWindowOptimizer`, wall-clock benchmark.
- **`tests/test_query_optimizer.py`** — 5 tests including 500-candidate benchmark (< 1 s gate).

### 2026-08-25 — Phase 2 Week 10 reward model (Pillar 1)

- **`sparc/reward_model/`** — `TrajectoryEncoder`, `EnsembleHead`, `RewardEnsemble` (M=7 bootstrap), EM-style E/M steps, disagreement/uncertainty APIs.
- **`tests/test_reward_model.py`** — golden-output regression test (seed 42, torch 2.5.1).

### 2026-08-25 — Phase 2 Week 9 environment & data layer

- **`sparc/env/segments.py`** — `TrajectorySegment`, rover feature extractor, `slice_segments`.
- **`sparc/env/buffer.py`** — `SegmentBuffer` for query candidate pools.
- **`sparc/env/dataset.py`** — `PreferenceQuery`, `PreferenceRecord`, `PreferenceDataset`.
- **`sparc/annotator_sim/synthetic.py`** — `SyntheticAnnotator` (noise, skip, rotation, Drift-1); stress/easy regime factories.
- **`tests/test_{segment_buffer,preference_dataset,synthetic_annotator}.py`** — 15 new unit tests (17 total).

### 2026-08-25 — Phase 2 Week 8 repo architecture (skeleton)

- **`sparc/{env,reward_model,query_optimizer,policy,annotator_sim,drift_detector,cli}/`** — Empty subpackages with phase/week pointers (no pillar logic yet).
- **`pyproject.toml`** — Pinned optional deps: `torch==2.5.1`, `gymnasium==1.0.0`, `stable-baselines3==2.4.0`; dev: pytest, ruff, black, mypy, pre-commit.
- **`.pre-commit-config.yaml`** — black, ruff, mypy hooks (draft for review).
- **`.github/workflows/ci.yml`** — ruff, black check, package layout smoke test on push/PR.
- **`tests/test_package_layout.py`** — Subpackage import smoke test.

_First implementation tag target: `v0.1.0-alpha` (ROADMAP Week 15)._
