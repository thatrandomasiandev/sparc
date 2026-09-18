# Results log

Source of truth for narrative detail: [`PROJECT.md`](PROJECT.md) §5. Do not re-headline
retracted claims (§5.4).

## Experiment 1 — factored vs VPL-style (NEGATIVE)

No win at any consistency spread; corr(b) ≈ 0 under random context queries.

## Experiment 2 — consistency recovery

| Regime | corr(b̂,b) | median abs err |
|--------|-----------|----------------|
| Easy | 0.556 | **23.09** |
| Hard | **0.867** | 1.07 |
| Mixed | 0.608 | 0.54 |
| Random | **0.753** | 0.63 |

Caveat: random is better than expected — always cite alongside hard.

## Experiment 3 — decision-relevant elicitation (PRELIMINARY POSITIVE)

Powered d=8, 150 user-runs (10×15):

| Metric | random | BALD | VOI | oracle |
|--------|--------|------|-----|--------|
| Queries to threshold | 6.08±0.34 | 5.77±0.33 | 5.14±0.33 | 2.21±0.21 |
| % reaching threshold | 66% | 69% | 75% | 95% |
| Final-query regret | 3.666±0.340 | 3.390±0.333 | **2.365±0.264** | 0.249±0.065 |

Paired bootstrap (queries-to-threshold): BALD−VOI **+0.63 [+0.02, +1.25]** (fragile);
random−BALD CI contains 0. Prefer **final-regret** as primary going forward.

Oracle gap (2.21 vs 5.14 queries; 0.249 vs 2.365 regret) is algorithmic headroom — queue #1.

## Primary metric decision (pre-specified before next run)

**Primary:** final-query regret  
**Secondary:** queries-to-threshold (with tie + censoring rates)

### E-eq / E-eq-d — PRE-SPECIFIED 2026-09-17 (before launch)

- **Claims:** C-mot-1, C-mot-2
- **Primary (amended before re-run):** `local_stable_rate` — P(π* unchanged under ε-perturbation);
  exact pairwise matches on dense random terrain are near-zero (straw man) and reported only as secondary
- **Secondary:** pairwise policy-match rate / compression among random unit rewards
- **Protocol:** 200 samples × dims {2,4,8,12,16} × seeds {0,1,2}, 6×6 terrain grid;
  local: 40 centers × 25 perturbations, ε=0.1
- **Command:** `python experiment_eq.py --out experiments/runs/exp_eq.jsonl`
- **Verdict:** _pending_

### E-decouple — PRE-SPECIFIED 2026-09-17 (before launch)

- **Claims:** C-mech-1 (BALD−VOI gap grows with decoy fraction ρ)
- **Primary:** paired BALD − VOI final-query regret as a function of ρ
- **Prediction:** slope of mean gap vs ρ > 0
- **Kill:** flat or decreasing slope
- **Protocol:** ρ ∈ {0, 0.25, 0.5, 0.75}, 3 seeds × 8 users, 8 queries, d_total=8
- **Command:** `python experiment_decouple.py --out experiments/runs/exp_decouple.jsonl`
- **Verdict:** _pending_

### E3-power — POP-VOI evaluation (PRE-SPECIFIED 2026-09-17, before launch)

- **Claims:** C-main-1 (VOI < random), C-main-2 (VOI < BALD), C-main-3 (oracle gap)
- **Primary:** final-query regret (paired bootstrap 10k)
- **Secondary:** queries-to-threshold (+ tie rate, censoring/reach rate)
- **Protocol:** d=8, 10 seeds × 15 users, 10 queries, 48 particles, 24 candidates
- **Strategies:** random, bald, voi (`Acquisition.VOI` / POP-VOI), oracle (true-regret)
- **Algorithm:** `plr.algorithm.POPVOI`
- **Command:**
  `python experiment_3.py --seeds 0 1 2 3 4 5 6 7 8 9 --d 8 --n-users 15 --strategies random bald voi oracle --out experiments/runs/exp3_power_d8.jsonl`
- **Verdict:** _pending run_

### Queue #1 run — VOI estimator ablation (2026-09-17)

**Pre-specified before seeing results:** primary = final-query regret.

Setup: d=8, 3 seeds × 8 users (n=24 paired), 10 queries, 32 particles, 16 candidates.

| Strategy | Final regret | Queries to thresh | Reach % |
|----------|-------------:|------------------:|--------:|
| oracle (true-regret) | **1.061 ± 0.259** | 4.08 | 75% |
| bald | 2.749 ± 0.628 | 5.75 | 67% |
| voi (particle loss) | 2.921 ± 0.661 | 5.63 | 58% |
| voi_mean (legacy) | 3.507 ± 0.719 | 6.04 | 54% |
| random | 3.882 ± 0.903 | 6.46 | 58% |

Paired bootstrap (final regret):

- **voi − oracle: +1.86 [+0.60, +3.40]** — oracle gap still large and significant
- voi_mean − voi: +0.59 [−1.24, +2.37] — particle helps on point estimate, **not established**
- bald − voi: −0.17 [−0.53, +0.20] — no VOI win over BALD in this underpowered ablation

**Bug fixed during run:** legacy `oracle_score` optimized posterior loss under true answer
probs, which is **not** a ceiling for true mean-decision regret (it could lose to random).
New `oracle_true_regret_score` is the eval-metric ceiling; use that going forward.

**Reading for queue #1:** particle loss alone does not close the oracle gap. Point estimate
vs `voi_mean` moves the right way but CI includes zero at n=24.

**Lookahead probe** (2 seeds × 6 users, n=12): `voi_look2` **worse** than one-step `voi`
on final regret (3.18 vs 2.80; diff NS). Greedy depth-2 does not close the oracle gap.

**Queue #1 verdict:** acquisition-side tweaks (particle loss, myopic lookahead) do not
close the true-regret oracle gap. Gap is likely posterior / decision-rule limited
(particles may miss `w_true`; we still *act* with posterior mean). Next: queue #2
(structural decoupling — tests whether VOI≻BALD when irrelevant dims exist) and/or
change the **decision** rule (act with sampled/minimax policy), not just acquisition.
