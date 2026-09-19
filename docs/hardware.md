# Hardware & human-study track — ACTIVE

**Status:** un-parked (2026-09-18). Lab robot arm access is available; we will use it for
trajectory comparison / human preference sessions alongside the sim track.

Sim queue items remain the path to RSS numbers (exact regret stays on the gridworld).
Hardware is a complementary validation track, not a substitute for powered sim claims.

---

## Current plan

- **Platform:** robot arm (exact model / driver stack TBD — record once named)
- **Role:** show trajectory pairs (or short segments), collect preference answers, log
  sessions via `plr/hardware/`
- **Keep:** offline video / pre-recorded clip studies as a fallback when live teleop
  is unavailable
- **Do not break:** I3 (`QueryCosts` raises on missing modality); exact-regret oracle
  remains sim-only

## Useful now

- `plr/hardware/` session logger + segment featurization (no robot SDK required for
  logging / offline clips)
- Wire arm SDK / teleop only when the platform is named and a trained operator is set

## Still required before live human data

Named platform + operator (Josh trained, or lab mate), IRB/safety path with Prof. Bıyık.
See git history for the previous pilot plan (N≈8–12, VOI vs BALD).
