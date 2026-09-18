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

## Experiment 3 — decision-relevant elicitation

### UNREPRODUCED — produced outside this repo, no provenance

Powered d=8, 150 user-runs (10×15), labeled "120 particles" in older notes.
**Not produced by any code in this repository.**

| Metric | random | BALD | VOI | oracle |
|--------|--------|------|-----|--------|
| Queries to threshold | 6.08±0.34 | 5.77±0.33 | 5.14±0.33 | 2.21±0.21 |
| % reaching threshold | 66% | 69% | 75% | 95% |
| Final-query regret | 3.666±0.340 | 3.390±0.333 | **2.365±0.264** | 0.249±0.065 |

Paired bootstrap (queries-to-threshold): BALD−VOI **+0.63 [+0.02, +1.25]** (fragile);
random−BALD CI contains 0. Prefer **final-regret** as primary going forward.

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
- **Verdict: NULL on the primary metric.** Final regret (n=150, 48 particles): random 3.356, BALD 2.813, VOI 2.683, oracle 0.854. Paired: BALD−VOI +0.13 [−0.59, +0.83]; random−VOI +0.67 [+0.004, +1.36] (just clears zero); VOI−oracle +1.83 [+1.32, +2.38]. Secondary (queries-to-threshold): BALD−VOI +0.35 [−0.24, +0.93], tie rate 0.56; reach rates random 56%, BALD 59%, VOI 64%, oracle 77%.
- C-main-2 (VOI < BALD): **not supported**. C-main-1 (VOI < random): fragile.
- **Provenance note:** this run's code differs from baseline commit `00363ab` (`pre-rerun-baseline`) in `plr/domains.py` and `plr/mdp.py` (edited during/after the run). Aggregate: `experiments/runs/exp3_power_d8_agg.txt`.

### E3-power-120 — PRE-SPECIFIED 2026-09-17, before launch

- **Baseline tag:** `pre-rerun-baseline` / `00363ab`; **pre-registration:** tag `e3-power-120-prereg`
- **Question:** does raising particles from 48 to 120 recover VOI < BALD? This is a new experiment, not a rescue of E3-power.
- **Claims:** C-main-1, C-main-2 (re-tested at 120 particles)
- **Protocol:** identical to E3-power except `--n-particles 120`. Same seeds 0–9, 15 users, d=8, 10 queries, 24 candidates, same candidate generation (pool drawn once per seed, single-state pairs). Change nothing else. One variable changes, so the result is directly comparable with the 48-particle null.
- **Primary:** mean regret over queries 1–3 (from `regret_curve[0:3]`).
- **Secondary:** area under the regret curve over queries 1–5.
- **Tertiary:** final-query regret (the E3-power primary, kept for continuity).
- All reported as paired bootstrap with 10k resamples, with tie rate and reach rate. Report all three whichever way they come out.
- **Rationale:** VOI's early lead is visible in the curves, and the claim is about low budgets. This metric is pre-registered after E3-power used final regret as primary.
- **Kill condition:** if the BALD−VOI CI on the primary metric contains zero, record C-main-2 as unsupported at 120 particles too.
- **Command:**
  `python experiment_3.py --seeds 0 1 2 3 4 5 6 7 8 9 --d 8 --n-users 15 --n-particles 120 --strategies random bald voi oracle --out experiments/runs/exp3_power_d8_p120.jsonl`
- **Output:** `experiments/runs/exp3_power_d8_p120.jsonl` (log alongside)
- **Verdict:** _pending launch_

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
