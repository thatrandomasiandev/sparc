# APReL friction log (Phase 0, Week 1)

> **Status:** Install and end-to-end run **deferred** — not blocking Phase 1 design lock. Revisit at Phase 3 baseline reproduction (Weeks 16–19) when adapting APReL's batch active learner.  
> **Source reviewed:** [APReL documentation](https://aprel.readthedocs.io/en/latest/overview.html), [Stanford-ILIAD/APReL](https://github.com/Stanford-ILIAD/APReL) README.

---

## Planned install (not yet executed)

```bash
git clone https://github.com/Stanford-ILIAD/APReL.git vendor/APReL
cd vendor/APReL && pip install -e .
python -m aprel.examples.driving  # bundled example (path may vary)
```

---

## Expected friction points (from docs / prior integration reports)

Documented here so SPARC Pillar 2 design choices are justified even before install.

| # | Friction | Impact on SPARC | Design response |
|---|----------|-----------------|-----------------|
| 1 | Legacy `gym` vs `gymnasium` API drift | Harness must wrap envs consistently for all baselines | Pin gymnasium in Phase 2; single env adapter layer |
| 2 | Feature-based / low-dim reward models in examples | MetaWorld/DMControl need trajectory encoders | SPARC Pillar 1 uses learned trajectory encoder (Phase 2 Week 10) |
| 3 | Batch optimizers (medoids, greedy) assume **sequential belief updates** between queries | Incompatible with comms blackout | Pillar 2: full batch at window open, no mid-window re-opt (`DESIGN.md` v3) |
| 4 | Single implicit `true_user` — no multi-operator rotation | Cannot model 3 disagreeing ops out of box | Pillar 1 per-operator latents; annotator sim in Phase 2 Week 9 |
| 5 | SAC / policy training external to APReL | Integration glue is project-owned | Single `sparc train` CLI (Phase 2 Week 14) |
| 6 | Volume removal / disagreement heuristics differ from EIG under latency | Baseline adapter, not SPARC core | Reproduce APReL batch learner per `EXPERIMENTS.md` baseline row |

---

## Post-install log (fill when run)

_When APReL is installed during Phase 3 baseline work, append:_

- Python / torch versions used
- Actual example command and runtime
- Errors encountered and fixes
- Diff vs expected friction above

---

## Revision log

| Date | Notes |
|------|-------|
| 2026-08-25 | Initial log — deferred install; expected friction from docs review |
