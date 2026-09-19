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

### E-eq / E-eq-d — 2026-09-18 — claims C-mot-1, C-mot-2

- **Primary (amended before re-run):** `local_stable_rate` — P(π* unchanged under ε-perturbation);
  exact pairwise matches on dense random terrain are near-zero (straw man) and reported only as secondary
- **Secondary:** pairwise policy-match rate / compression among random unit rewards
- **Protocol:** 200 samples × dims {2,4,8,12,16} × seeds {0,1,2}, 6×6 terrain grid;
  local: 40 centers × 25 perturbations, ε=0.1
- **Command:** `python experiment_eq.py --out experiments/runs/exp_eq.jsonl`
- **Aggregate:** `experiments/runs/exp_eq_agg.txt`
- **Figure:** `paper/figures/exp_eq_match_vs_d.{png,pdf}`
- **Numbers:**

| d | local_stable | pairwise match | compression |
|--:|-------------:|---------------:|------------:|
| 2 | **0.457 ± 0.047** | 0.026 ± 0.004 | 0.672 |
| 4 | 0.107 ± 0.010 | 0.0005 | 0.050 |
| 8 | 0.047 ± 0.004 | 0.000 | 0.000 |
| 12 | 0.022 ± 0.001 | 0.000 | 0.000 |
| 16 | 0.014 ± 0.003 | 0.000 | 0.000 |

- **Verdict:** C-mot-1 **supported at low d** (compression 0.67 + local stability 0.46 at d=2; weak at d≥8).
  C-mot-2 **killed as stated** — equivalence / local stability **falls** with `d`. Paper must say
  “equivalence collapses with dimension,” not “irrelevance grows.”

### E-decouple — 2026-09-18 — claim C-mech-1

- **Primary:** paired BALD − VOI final-query regret as a function of ρ
- **Prediction:** slope of mean gap vs ρ > 0 · **Kill:** flat or decreasing
- **Protocol:** ρ ∈ {0, 0.25, 0.5, 0.75}, 3 seeds × 8 users, 8 queries, d_total=8
- **Command:** `python experiment_decouple.py --out experiments/runs/exp_decouple.jsonl`
- **Aggregate:** `experiments/runs/exp_decouple_agg.txt`
- **Figure:** `paper/figures/exp_decouple_gap_vs_rho.{png,pdf}`
- **Paired BALD−VOI (n=24/ρ):**

| ρ | BALD−VOI | 95% CI |
|--:|---------:|--------|
| 0.00 | +0.45 | [−0.88, +2.01] |
| 0.25 | +0.13 | [−1.49, +1.99] |
| 0.50 | **+1.48** | **[+0.05, +3.29]** |
| 0.75 | +0.39 | [−0.56, +1.42] |

- Mean gap vs ρ slope ≈ **+0.46** (prediction direction)
- **Verdict:** **preliminary** — slope positive but non-monotone; only ρ=0.5 CI excludes 0.
  Do not lock abstract language on C-mech-1 until powered.

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
- **Output:** `experiments/runs/exp3_power_d8_p120.jsonl` · agg `experiments/runs/exp3_power_d8_p120_agg.txt`
- **Figures:** `paper/figures/exp3_power_d8_p120_*.{png,pdf}`
- **Commit at aggregate:** `2b05c57`
- **Verdict (n=150):**

| Metric | random | BALD | VOI | oracle | bald−voi CI |
|--------|-------:|-----:|----:|-------:|-------------|
| **early_mean_q1_3 (primary)** | 4.070 | 3.497 | 3.415 | 1.980 | +0.08 [−0.28, +0.44] |
| early_auc_q1_5 | 15.45 | 13.54 | 12.92 | 6.82 | +0.61 [−0.85, +2.07] |
| final_regret | 3.471 | 2.822 | 2.567 | 1.163 | +0.26 [−0.29, +0.79] |

  - random−VOI primary: **+0.655 [+0.060, +1.256]** (VOI < random)
  - **Kill hit:** BALD−VOI CI contains 0 on the pre-registered primary → C-main-2 **unsupported at 120 particles too**
  - Raising particles 48→120 does **not** recover VOI < BALD; final-regret point estimate improves slightly (2.68→2.57) but BALD gap stays NS

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

### E-prior — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-2
- **Primary:** final-query regret (VOI acquisition fixed)
- **Priors:** population | uniform sphere | cheat (true user)
- **Prediction:** uniform − population > 0 (CI excludes 0)
- **Kill:** CI contains 0 or negative
- **Protocol:** d=8, seeds 0–9, 12 users, 10 queries, 48 particles, 24 candidates
- **Command:** `python experiment_prior.py --seeds 0 1 2 3 4 5 6 7 8 9 --out experiments/runs/exp_prior.jsonl`

### E-act — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-4
- **Primary:** final-query regret under VOI acquisition
- **Rules:** mean | map | sample | softminimax (+ oracle/mean ceiling)
- **Prediction:** mean − softminimax > 0 on final regret
- **Kill:** no rule beats mean
- **Protocol:** d=8, seeds 0–5, 12 users, 10 queries, 48 particles
- **Command:** `python experiment_act.py --seeds 0 1 2 3 4 5 --out experiments/runs/exp_act.jsonl`

### E-mismatch — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-10
- **Primary:** paired bald−voi and random−voi final regret per mismatch ∈ {pl, lex, satisficing, fatigue}
- **Kill:** VOI worse than random under any mismatch (random−voi CI entirely negative)
- **Protocol:** d=8, seeds 0–5, 10 users, 10 queries, 48 particles
- **Command:** `python experiment_mismatch.py --seeds 0 1 2 3 4 5 --out experiments/runs/exp_mismatch.jsonl`

### E-neg — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-7
- **Primary:** random − VOI final regret under orthogonal-label answers
- **Kill:** VOI significantly beats random (claims decision-relevance while labels are orthogonal)
- **Command:** `python experiment_neg.py --seeds 0 1 2 3 4 5 --out experiments/runs/exp_neg.jsonl`

### E-prior-shift — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-3
- **Primary:** shifted − matched final regret under VOI
- **Prediction:** shifted > matched (CI excludes 0)
- **Command:** `python experiment_prior_shift.py --seeds 0 1 2 3 4 5 --out experiments/runs/exp_prior_shift.jsonl`

### E-decouple-power — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-mech-1 / path to C-main-2 on decoupled domain
- **Primary:** paired BALD−VOI final regret vs ρ; slope of mean gap vs ρ
- **Protocol:** ρ ∈ {0,0.25,0.5,0.75}, seeds 0–9, 12 users, 10 queries, 48 particles, 24 candidates
- **Command:** `python experiment_decouple.py --seeds 0 1 2 3 4 5 6 7 8 9 --n-users 12 --n-queries 10 --n-particles 48 --n-candidates 24 --out experiments/runs/exp_decouple_power.jsonl`

### Results — mechanism suite 2026-09-18 (commit `2b05c57`)

**E-prior (C-mech-2) — SUPPORTED.** n=120. Final regret: pop 2.90, uniform 4.64, cheat ≈0.
uniform−population **+1.74 [+0.88, +2.62]**. Artifact: `exp_prior.jsonl`.

**E-prior-shift (C-mech-3) — SUPPORTED.** n=72. matched 3.06, shifted 5.24, uniform 4.65.
shifted−matched **+2.18 [+1.05, +3.36]**. Artifact: `exp_prior_shift.jsonl`.

**E-act (C-mech-4) — KILLED.** n=72, VOI acq. mean 2.80 < softminimax 3.10 (NS) ≪ sample 4.50
(mean−sample **−1.70 [−2.87, −0.52]**). Oracle gap voi/mean−oracle **+2.03 [+1.24, +2.91]**.
Artifact: `exp_act.jsonl`.

**E-decouple-power (C-mech-1) — KILLED as stated.** n=120/ρ. No ρ with BALD−VOI CI excluding 0.
Slope ≈+0.15. At ρ=0.75 only: random−voi **[+0.06, +1.11]**. Still no VOI≺BALD.
Artifact: `exp_decouple_power.jsonl`.

**E-mismatch (C-mech-10) — supported (no kill).** n=60/type. No mismatch with random−voi entirely
negative. PL bald−voi still NS. Artifact: `exp_mismatch.jsonl`.

**E-neg (C-mech-7) — SUPPORTED.** n=60. random−voi **+0.18 [−1.19, +1.62]** (CI∋0).
Artifact: `exp_neg.jsonl`.

### L6 — B-Pref / PEBBLE transfer (assessment 2026-09-18)

- **Status:** not started as an experiment; interface stub in `plr/deep_bridge.py`.
- **Why blocked:** exact VI VOI does not transfer to deep control; needs rollout-based
  approx VOI + B-Pref/MuJoCo stack. That is a separate engineering milestone.
- **Env note:** MuJoCo 3.12 + `dm_control` present; `gym` missing; B-Pref cloned to `/tmp/B-Pref`.
- **Minimum bar when unblocked:** walker-walk (or one B-Pref task), swap acquisition only,
  primary = return @ feedback budget; kill if approx-VOI ≱ disagreement.
- **Do not claim L6 in the paper until that run exists.**

### Maturity after mechanism suite

| Level | Status |
|------:|--------|
| L2 | still here on VOI vs BALD |
| L3 | **not reached** (C-main-2 NS everywhere tested) |
| L4 | **partial** — prior necessary; decouple growth killed |
| L5 | **not reached** — E-act killed |
| L6 | **assessed + proxy run** — NS on cartpole (C-l6-1 kill); legacy B-Pref blocked |
| L7 | parked |

### E-L6-dmc — PRE-SPECIFIED 2026-09-18 (before launch)

- **Claim:** C-l6-1 — approx_voi final CEM true-return **>** disagreement (paired)
- **Why not stock B-Pref:** conda pin is Python 3.6 / PyTorch 1.4; unusable here.
  This is the minimum credible continuous-control acquisition bake-off.
- **Primary:** `true_return` (CEM on learned mean reward, evaluated under env reward)
- **Secondary:** Spearman(pred, true) on holdout segments
- **Strategies:** random | disagreement | approx_voi
- **Protocol:** dm_control cartpole/balance; seeds 0–5; 20 queries; ensemble 5
- **Kill:** disagreement−approx_voi CI on primary contains 0 or is negative
- **Command:**
  `python3 experiment_l6_dmc.py --seeds 0 1 2 3 4 5 --out experiments/runs/exp_l6_dmc.jsonl`

### E-L6-dmc — RESULTS 2026-09-18

- **Claim C-l6-1: NS / kill hit**
- n=6; Primary true_return: approx_voi **37.87**, disagreement **38.01**, random **37.96**
- disagreement−approx_voi: **+0.14 [−1.35, +1.57]**
- Secondary spearman: approx_voi 0.28 vs disagreement 0.09 (NS)
- Artifact: `experiments/runs/exp_l6_dmc.jsonl` · `exp_l6_dmc_agg.txt`
- Continuous-control acquisition transfer is implemented and tested; **no supported L6 win**.
  Do not claim field-facing VOI superiority over disagreement.

