# Week 6 external review packet (Phase 1)

> **Purpose:** Handoff for a sanity-check conversation with LIRA (Prof. Erdem Bıyık) and/or SLURM (Prof. Daniel Seita).  
> **Send:** `PROBLEM.md`, `DESIGN.md` v3, `EXPERIMENTS.md` v2, this file.  
> **Status:** Ready to send — no pillar implementation yet (Phase 2 Week 8+).

---

## One-paragraph pitch

SPARC learns robot rewards from **sparse, delayed, batched** human preferences when **three rotating operators disagree** and preferences may **drift** mid-mission — a setting mainstream PbRL (PEBBLE, SURF, RUNE, APReL) does not target. Four integrated mechanisms: per-operator latent reward model, comms-window batch query optimizer, confidence-gated policy bounding, and online drift detection. **Success:** match PEBBLE asymptotic return using **≤20% of its preference queries** under simulated comms delay + 3 synthetic annotators + scripted drift.

---

## Design commits to sanity-check

These are explicit v2/v3 decisions that affect implementation structure — feedback welcome.

| # | Commit | Location | Question for reviewer |
|---|--------|----------|----------------------|
| 1 | **EM-style alternation** for operator latents \(z_i\): MAP E-step (`InferOperatorLatent`), Adam M-step on shared/adapter params — **not** joint SGD on \(z_i\) | `DESIGN.md` § Pillar 1 pseudocode | Is alternating MAP/Adam standard enough here, or should we joint-optimize with a variational bound? |
| 2 | **EIG scoring** uses one Newton step on linear adapter \(w_i\) on fixed embeddings — **no** full retrain per candidate query | `DESIGN.md` § Pillar 2 implementation note | Is rank-1 Laplace on \(w_i\) sufficient for batch selection, or do we need full ensemble refit? |
| 3 | **Bounding threshold** \(\tau_t = \tau_0 + \kappa \cdot q_{90,t}\) from rolling replay-buffer P90, updated **once per comms window** | `DESIGN.md` § Pillar 3 | Adaptive percentile vs fixed \(\tau\): risk of gate always-on in obstacle-heavy nav? |
| 4 | **Drift detection** via SPRT; re-query overrides half the next window batch | `DESIGN.md` § Pillar 4 | Will operator **rotation** (not drift) trigger false positives? (Red-team flagged.) |
| 5 | **Comms window** = 900 s **simulated mission time**, \(B=8\), 22.5 episodes/window on rover env | `EXPERIMENTS.md`, `PROBLEM.md` | Reasonable stress vs real URC comms cadence? |
| 6 | **Drift-1** at \(f_{\mathrm{drift}}=0.5\) of each env's train steps (not universal step count) | `EXPERIMENTS.md` | Fair cross-env comparison? |

---

## Red-team falsifiable checks (already in `DESIGN.md`)

Ask reviewers whether these failure modes are the right ones to test in Phase 3:

1. **Pillar 1:** With ≤200 total queries and 3 operators, \(\|w_i - \bar w\|\) variance collapses → SPARC silently becomes single-reward PEBBLE.
2. **Pillar 2:** Labels arrive for stale policy snapshots → reward NLL on received labels vs current rollouts diverges.
3. **Pillar 3:** In narrow corridors, \(D > \tau\) on >50% of batch → freeze mode blocks all learning.
4. **Pillar 4:** Operator rotation triggers SPRT false alarms → re-query storms burn query budget.

---

## Locked success criterion (from `PROBLEM.md`)

**Pass (sim, Phase 4):** \(N_q^{\mathrm{SPARC}} \leq 200\) vs PEBBLE 1000; non-inferior \(\bar{R}_\infty\); stress regime active (windows, 3 annotators, Drift-1). Welch's t-test, 10 seeds — details in `EXPERIMENTS.md`.

---

## What we are NOT asking reviewers to endorse

- Implementation correctness (no code yet)
- Hardware validation plan (Phase 5)
- Hyperparameter defaults marked provisional in `EXPERIMENTS.md`

---

## Suggested review format

30-minute conversation or async bullet feedback on:

1. Is the **problem gap** real and non-strawman?
2. Are the **six design commits** above the biggest risks?
3. Any **missing baseline** or **metric** before Phase 3 training runs?
4. Should `DESIGN.md` bump to v4 after this review?

---

## Revision log (this packet)

| Date | Notes |
|------|-------|
| 2026-08-25 | Initial packet — post v2 revision pass and `PROBLEM.md` lock |
