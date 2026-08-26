# SPARC

**S**parse-feedback **P**reference **A**ctive **R**eward **C**ore — a preference-learning engine for robots that receive sparse, delayed, and inconsistent human feedback.

Engineered to the discipline bar set by [STAR](https://github.com/alexdobin/STAR) (depth, governance, reproducibility) — not its domain.

---

## Status

| Phase | Weeks | Status |
|-------|-------|--------|
| **0** — Foundation | 1–3 | Complete (scope locked) |
| **1** — Design & protocol | 4–7 | Complete (external review recommended) |
| **2** — Core engineering | 8–15 | **Week 15 hardening complete (`0.1.0a0`)** |
| **3** — Baseline reproduction | 16–19 | **Complete** (Week 19 dry run passed) |
| **4** — Simulation validation | 20–26 | **In progress** — manifests + `RESULTS.md` scaffold |

**Current focus:** Phase 4 primary stress runs (10-seed walker + rover) per `RESULTS.md` manifests.

---

## Package layout

```
sparc/
├── env/              # Environment wrappers, rover-nav proxy (Week 9+)
├── reward_model/     # Pillar 1 (Week 10+)
├── query_optimizer/  # Pillar 2 (Week 11+)
├── policy/           # Pillar 3 bounding wrapper (Week 12+)
├── annotator_sim/    # Synthetic operators (Week 9+)
├── drift_detector/   # Pillar 4 SPRT (Week 13+)
├── baselines/        # PEBBLE, PrefPPO (Week 16+)
├── harness/          # Shared benchmark runner
├── engine/           # SparcTrainer integration (Week 14+)
├── logging/          # JSONL audit log
└── cli/              # sparc train / benchmark entry points
```

```bash
pip install -e ".[dev,rl]"
pip install -e ".[dmc]"   # MuJoCo + DMControl (walker-walk)
pytest tests/ -q
sparc train --config experiments/configs/smoke_easy.json --log /tmp/sparc.jsonl
sparc benchmark --method sparc --config experiments/configs/sparc_rover_stress_smoke.json --seeds 0,1
sparc suite --manifest experiments/configs/dry_run_pendulum_ci.json --output-dir /tmp/sparc_suite
sparc suite --manifest experiments/configs/dry_run_walker_stress_smoke.json --output-dir experiments/runs/week19_walker --seeds 0,1
sparc benchmark --method surf --config experiments/configs/surf_walker_walk.json --seeds 0
sparc benchmark --method aprel --config experiments/configs/aprel_walker_walk.json --seeds 0
sparc benchmark --method sparc --config experiments/configs/sparc_walker_stress_10seed.json --seeds all --output-dir experiments/runs/sparc_walker_stress
sparc compare --a experiments/runs/pebble_walker_stress/summary.json --b experiments/runs/sparc_walker_stress/summary.json
```

---

## Documentation (authority order)

| Document | Purpose |
|----------|---------|
| [`PROBLEM.md`](PROBLEM.md) | Locked scenario and falsifiable success criteria |
| [`DESIGN.md`](DESIGN.md) | Four pillars — math, pseudocode, red-team notes (v3) |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Locked evaluation protocol, baselines, metrics (v2) |
| [`ROADMAP.md`](ROADMAP.md) | 38-week build plan (sequencing source of truth) |
| [`CHANGES.md`](CHANGES.md) | Spec and behavior revision log |

### Phase 0 research

| Document | Purpose |
|----------|---------|
| [`docs/prior-art-matrix.md`](docs/prior-art-matrix.md) | PbRL method comparison |
| [`docs/field-deployment-memo.md`](docs/field-deployment-memo.md) | Why lab PbRL fails in the field |
| [`docs/aprel-friction-log.md`](docs/aprel-friction-log.md) | APReL install notes (deferred) |
| [`docs/week6-external-review-packet.md`](docs/week6-external-review-packet.md) | Reviewer handoff |

**Not yet validated:** Phase 4 primary gates — see [`RESULTS.md`](RESULTS.md) (scaffold; full 10-seed stress runs pending).

---

## Success criterion (primary)

Match **PEBBLE** asymptotic task return using **≤20% of its preference queries** under simulated comms delay, 3 disagreeing synthetic annotators, and scripted preference drift. Details in `PROBLEM.md` and `EXPERIMENTS.md`.

---

## License

MIT — see [`LICENSE`](LICENSE). Governance: [`CONTRIBUTING.md`](CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
