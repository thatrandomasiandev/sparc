# Experiments protocol

**Authority:** [`docs/CLAIMS.md`](docs/CLAIMS.md) is the claim↔experiment contract.
This file is the short index. **I8:** no paper number without a Claim ID + run artifact.

Standing order: be creative, thorough, and back every manuscript number with an experiment.

---

## Status board

| ID | Question | Tier | Status |
|----|----------|------|--------|
| E1 | Factored latent vs global-b? | 1 | NEGATIVE |
| E1-strat | Difficulty-stratified rescue? | 1 | planned |
| E2 | Is b recoverable / difficulty? | 1 | done (+ caveat) |
| E3-ablate | Particle / lookahead VOI? | 1 | done — partial kill |
| E3-power | VOI vs BALD vs random (powered) | 1 | **needed** (abstract blocked) |
| E-eq / E-eq-d | Policy equivalence stats | 2 | **next cheap** |
| E-decouple | Irrelevant features ⇒ VOI≻BALD? | 2 | queue #2 flagship |
| E-prior / E-prior-shift | Population prior necessary? | 2 | planned |
| E-act | Better *action* closes oracle gap? | 2 | planned |
| E-calib / E-cover / E-neg | Calibration, coverage, neg. control | 2 | planned |
| E-budget | Full regret vs t curves | 2 | planned |
| E-diag-bald | High BALD, low decision value | 2 | planned |
| E-ternary / E-mismatch | K=3; non-PL users | 2 | planned |

Hardware: **PARKED** (`docs/hardware.md`).

---

## Global rules

- Primary metric for elicitation: **final-query regret** (pre-specify in RESULTS before launch)
- Paired bootstrap 10k; ≥3 seeds exploratory; powered claims ≥10×15 when possible
- Oracle = **true-regret** ceiling (`oracle_true_regret_score`)
- Every figure caption lists Claim IDs from `docs/CLAIMS.md`

---

## Commands (spine)

```bash
pytest -q
python experiment_1.py --seeds 0 1 2
python experiment_2.py --seeds 0 1 2
python experiment_3.py --seeds 0 1 2 3 4 5 6 7 8 9 --d 8 --n-users 15 \
  --strategies random bald voi oracle \
  --out experiments/runs/exp3_power_d8.jsonl
python aggregate_exp3.py experiments/runs/exp3_power_d8.jsonl
python experiments/scripts/plot_exp3.py experiments/runs/exp3_power_d8.jsonl
```

New Tier-2 scripts land as `experiment_eq.py`, `experiment_decouple.py`, … when implemented;
register them in `docs/CLAIMS.md` the same day.
