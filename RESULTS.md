# SPARC — Results (Phase 4)

> **Status:** Scaffold — primary stress-regime gates **not yet validated** at full scale.  
> **Authority:** `PROBLEM.md` success criteria · `EXPERIMENTS.md` protocol · regenerable artifacts under `experiments/runs/`.

---

## Phase 4 pilot (in progress)

Single-seed full-scale walker-walk stress run (500k steps, PEBBLE vs SPARC):

```bash
sparc suite --manifest experiments/configs/phase4_stress_walker_pilot.json \
  --output-dir experiments/runs/phase4_walker_pilot --seeds 0
```

Artifacts: `experiments/runs/phase4_walker_pilot/` (running).

---

## Primary success gate (stress regime)

From `PROBLEM.md` — pass requires **all** of the following on `walker-walk` and `sparc-rover-nav-v0`, 10 seeds:

| Criterion | Target | Walker-walk | Rover-nav | Evidence |
|-----------|--------|-------------|-----------|----------|
| Query budget | SPARC ≤ 200 (PEBBLE ref. 1000) | _pending_ | _pending_ | `phase4_stress_*/*/summary.json` |
| Task quality | SPARC R_∞ not worse than PEBBLE (Δ_NI = 50) | _pending_ | _pending_ | Welch t-test via `sparc compare` |
| Stress active | ΔT=900s, K=3, Drift-1 @ 50% | _pending_ | _pending_ | configs + JSONL drift events |

**Rehearsal only (Week 19 smoke, 6k steps, 2 seeds):** see `experiments/runs/week19_dry_run/SUITE_REPORT.md` — not reportable as Phase 4 validation.

---

## How to reproduce (Phase 4 primary runs)

```bash
# Walker-walk stress — 10 seeds, ~500k steps (long-running)
sparc suite --manifest experiments/configs/phase4_stress_walker_primary.json \
  --output-dir experiments/runs/phase4_walker_stress --seeds all

# Rover-nav stress — 10 seeds, ~750k steps (long-running)
sparc suite --manifest experiments/configs/phase4_stress_rover_primary.json \
  --output-dir experiments/runs/phase4_rover_stress --seeds all

# Report + primary gate
sparc report --suite-summary experiments/runs/phase4_walker_stress/suite_summary.json
python experiments/scripts/plot_suite_curves.py experiments/runs/phase4_walker_stress
sparc compare --a experiments/runs/phase4_walker_stress/pebble/summary.json \
              --b experiments/runs/phase4_walker_stress/sparc/summary.json
```

---

## Metrics checklist (`EXPERIMENTS.md`)

| # | Metric | Implementation | Status |
|---|--------|----------------|--------|
| 1 | Policy return vs. queries | `eval_history` in run JSON; learning curve via `sparc.harness.report.learning_curve_points` | Implemented |
| 2 | Reward-model held-out accuracy | `sparc.harness.rm_accuracy` | Implemented |
| 3 | Regret under drift | `sparc.harness.drift_regret` (best-checkpoint proxy) | Implemented |
| 4 | Wall-clock per query | `sparc.harness.log_metrics` from JSONL | Implemented |
| 5 | Multi-seed CI + Welch | `sparc.harness.stats`, `sparc compare` | Implemented |
| 6 | Stress regime verification | `sparc.harness.stress_verify` from SPARC JSONL | Implemented |

---

## Baseline suite (Week 19 dry run)

Full harness rehearsal on `walker-walk` stress smoke (6 methods, 2 seeds):

- Artifacts: `experiments/runs/week19_dry_run/`
- Report: `experiments/runs/week19_dry_run/SUITE_REPORT.md`

---

## Ablations (Week 25)

Configs disable one pillar at a time under the stress regime:

| Config | Ablation |
|--------|----------|
| `sparc_walker_ablate_p1.json` | − Pillar 1 (`share_operator_latents`) |
| `sparc_walker_ablate_p2.json` | − Pillar 2 (`query.selection_mode=random`) |
| `sparc_walker_ablate_p3.json` | − Pillar 3 (`bounding_mode=pass-through`) |
| `sparc_walker_ablate_p4.json` | − Pillar 4 (`enable_drift_detector=false`) |

Smoke manifest: `experiments/configs/phase4_ablation_walker_smoke.json`.

---

## Honest limitations (to be updated after Phase 4)

- Week 19 smoke uses 6k env steps — variance is high; wide CIs expected.
- PEBBLE stress runs use oracle teacher, not multi-operator synthetic labels (fair for return, not for annotator modeling).
- Hardware validation (Phase 5) not started.
- Pre-fix pilot (aborted): PEBBLE `num_interact` was counted per 128-step learn chunk; fixed to env steps before restarting Phase 4 pilot.

---

## Revision log

| Date | Notes |
|------|-------|
| 2026-08-25 | Initial scaffold after Week 19 dry run; Phase 4 manifests added |
| 2026-08-25 | PEBBLE interact-schedule fix; Week 25 ablation configs + flags; pilot restarted |
