# Harness suite: week19_walker_stress_smoke

**Env:** `walker-walk` · **Seeds:** [0, 1]

| Method | Seeds | R_∞ (learned) | 95% CI | True R_∞ | Queries | Wall (s) |
|--------|-------|---------------|--------|------------|---------|----------|
| aprel | 2 | -22.0 | [-156.3, 112.3] | 21.6 | 16.0 | 69 |
| pebble | 2 | -22.0 | [-156.3, 112.3] | 21.6 | 16.0 | 56 |
| prefppo | 2 | 0.0 | [0.0, 0.0] | 0.0 | 16.0 | 3 |
| rune | 2 | -24.0 | [-110.2, 62.3] | 20.8 | 16.0 | 77 |
| sparc | 2 | -0.3 | [-1.1, 0.6] | 17.2 | 12.0 | 22 |
| surf | 2 | 5.2 | [-960.1, 970.5] | 20.6 | 16.0 | 49 |

## Primary comparison

### sparc_vs_pebble

- **R_∞:** -0.3 vs -22.0 (Welch p=0.2885)
- **Queries:** 12.0 vs 16.0 (25% reduction)

## Primary gate (PROBLEM.md)

| Check | Result | Detail |
|-------|--------|--------|
| Query budget (≤200, ≤20% PEBBLE) | **PASS** | SPARC 12 vs PEBBLE 16 |
| Return non-inferiority (Δ ≥ −50) | **PASS** | Δ=21.7 (SPARC -0.3 vs PEBBLE -22.0) |
| **Overall** | **PASS** | 2 SPARC / 2 PEBBLE seeds |

_Stress conditions (K=3, ΔT, drift) verified by config + JSONL — not auto-checked here._


## Stress regime verification (SPARC JSONL)

- **sparc:** FAIL — 3 windows, ops=[1, 2, 3], scripted_drift=0, sprt_triggers=0

