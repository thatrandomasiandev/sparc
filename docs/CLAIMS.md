# Claim ↔ experiment contract

**Standing order (Josh, 2026-09-17):** Every number and empirical sentence in the paper
must be backed by a named experiment that produces that number. Be creative. Be
thorough. Prefer an extra negative/diagnostic experiment over an unsupported claim.

If you are a coding agent: treat this file + `AGENTS.md` **I8** as permanent. Do not
“remember” only in chat.

---

## Hard rule (I8)

| | |
|--|--|
| **I8** | No paper claim without a **Claim ID** below, a reproducing command, a run artifact under `experiments/runs/`, and a row in `RESULTS.md`. Figures cite Claim IDs in captions. |

**Workflow for every manuscript edit that asserts a number:**

1. Find or add a Claim ID in §A.
2. Point it at an Experiment ID in §B (or design a new one).
3. Run it (≥3 seeds; paired bootstrap for comparisons).
4. Paste numbers into `RESULTS.md` with seed list + git commit.
5. Only then put the number in `paper/`.

If the experiment fails or is NS: **change the paper**, do not keep the number.

---

## A. Claim inventory (paper-facing)

Status: `supported` · `preliminary` · `retracted` · `planned` · `blocked`

### A1. Motivating facts (must be evidenced, not vibes)

| ID | Claim (paraphrase) | Experiment | Status |
|----|--------------------|------------|--------|
| C-mot-1 | Many distinct unit rewards induce the **same** optimal policy on our domain | E-eq | supported at low `d` (local_stable≈0.46, compression≈0.67 at d=2); weak at d≥8 |
| C-mot-2 | Fraction of reward-space volume (or particle pairs) that is decision-irrelevant grows with `d` | E-eq-d | **killed as stated** — local stability and pairwise match **fall** with `d`; rewrite to “equivalence collapses with `d`” |
| C-mot-3 | BALD can select queries with high param IG but near-zero true-regret reduction | E-diag-bald | planned |

### A2. Method / formal claims (tests + small probes)

| ID | Claim | Experiment / test | Status |
|----|-------|-------------------|--------|
| C-form-1 | `‖w‖=1` removes BT scale confound; doubling `w` does not change probs if `b` fixed via `rewards()` | `test_unit_norm_invariant_i1` | supported |
| C-form-2 | BT ≡ Plackett–Luce at K=2 | `test_bradley_terry_equals_plackett_luce_at_k2` | supported |
| C-form-3 | Trivial (identical) query has BALD = 0 | `test_trivial_query_has_zero_information` | supported |
| C-form-4 | Difficulty = reward gap, independent of `b` | `test_difficulty_independent_of_b` | supported |
| C-form-5 | `V^π(w)` is linear in `w` for fixed `π` (enables exact batch regret) | `test_policy_value_linearity` | supported |
| C-form-6 | POP-VOI implements select→update→decide under population particles | `tests/test_algorithm.py` | supported |

### A3. Consistency / encoder (negative + explanatory — keep honest)

| ID | Claim | Experiment | Status |
|----|-------|------------|--------|
| C-enc-1 | Factored latent does **not** beat VPL-style global-`b` on random contexts | E1 | supported (NEGATIVE) |
| C-enc-2 | `b` is recoverable in principle when `w` known, if queries are not all easy | E2 | supported |
| C-enc-3 | Easy-only queries make MLE `b` catastrophic (median abs err ≫ true scale) | E2-easy | supported |
| C-enc-4 | Random contexts are *not* strictly worse than hard for `corr(b)` — caveat mandatory | E2-random | supported |
| C-enc-5 | Difficulty-stratified contexts rescue joint `(w,b)` recovery | E1-strat | planned |

### A4. Main elicitation results (Exp 3 family)

| ID | Claim | Experiment | Status |
|----|-------|------------|--------|
| C-main-1 | VOI final regret **<** random (paired bootstrap CI excludes 0) | E3-power / E3-power-120 | **fragile supported** — 48p final +0.67 [+0.004,+1.36]; 120p early_mean +0.66 [+0.06,+1.26] |
| C-main-2 | VOI final regret **<** BALD (paired CI excludes 0) | E3-power / E3-power-120 | **NS** — do not claim; 48p final and 120p early_mean_q1_3 both CI∋0 |
| C-main-3 | True-regret oracle ≪ VOI (oracle gap) | E3-power / E3-ablate | **supported** — power gap ~1.83 final regret |
| C-main-4 | BALD not significantly better than random at d=8 (legacy claim weakened) | E3-power | **supported (NS win)** — random−BALD CI contains 0 |
| C-main-5 | Queries-to-threshold: VOI ≤ BALD — **secondary only**; report ties + censoring | E3-power | fragile / demoted |
| C-main-6 | Particle-loss VOI beats mean-loss VOI | E3-ablate | **NS** — do not claim |
| C-main-7 | Depth-2 lookahead closes oracle gap | E3-look2 | **killed** — no help |

### A5. Mechanistic / creative claims (make the paper deep)

| ID | Claim | Experiment | Status |
|----|-------|------------|--------|
| C-mech-1 | When ≥ρ fraction of features are **unreachable** (decision-irrelevant), VOI−BALD gap **grows** in ρ | E-decouple-power | **killed as stated** — powered n=120/ρ; no ρ with BALD−VOI CI excluding 0; slope ≈+0.15 weak |
| C-mech-2 | Population prior beats uniform particle prior on final regret (same budget) | E-prior | **supported** — uniform−pop +1.74 [+0.88, +2.62] (n=120) |
| C-mech-3 | Misspecified prior (train modes ≠ test modes) increases final regret vs matched prior | E-prior-shift | **supported** — shifted−matched +2.18 [+1.05, +3.36] (n=72) |
| C-mech-4 | Acting with minimax-regret / sampled particle policy reduces oracle gap vs mean action | E-act | **killed** — mean best; sample significantly worse; softminimax NS vs mean |
| C-mech-5 | VOI score is calibrated: higher predicted Δregret ⇒ higher realized Δregret | E-calib | planned |
| C-mech-6 | Posterior coverage: distance of `w_true` to particle set predicts residual regret | E-cover | planned |
| C-mech-7 | Negative control: labels from reward orthogonal to task ⇒ VOI ≉ better than random | E-neg | **supported** — random−voi CI∋0 (n=60) |
| C-mech-8 | Budget curves: regret vs #queries for all methods, one figure, all Claim IDs marked | E-budget | planned |
| C-mech-9 | Ternary vs binary at matched wall-clock *or* matched query count | E-ternary | planned |
| C-mech-10 | Model mismatch (satisficing / lex / fatigue): graceful degradation (VOI ≱ worse than random) | E-mismatch | **supported (no kill)** — no mismatch with random−voi CI entirely negative |

| ID | Claim | Experiment | Status |
|----|-------|------------|--------|
| C-l6-1 | On dm_control cartpole, approx_voi CEM return **>** disagreement (paired) | E-L6-dmc | **NS / kill** — disagreement−approx_voi +0.14 [−1.35, +1.57] (n=6); legacy B-Pref unusable (py3.6) |

Map each abstract sentence to IDs; **rewrite abstract** until every empirical clause is `supported` or clearly `preliminary`.

| Abstract clause | Needs | Action |
|-----------------|-------|--------|
| “lower final policy regret than random and BALD” | C-main-1, C-main-2 | Keep fragile random win; **drop BALD win** (NS) |
| “large gap to an oracle” | C-main-3 | Keep; cite E3-power / E3-ablate / E-act |
| “population prior makes criterion usable” | C-mech-2 | **Keep** — E-prior supported 2026-09-18 |
| “irrelevant features ⇒ VOI≻BALD” | C-mech-1 | **Drop** — E-decouple-power killed |

---

## B. Experiment catalog (creative + thorough)

Every experiment lists: **question**, **prediction**, **kill**, **primary metric**, **factors**, **n**, **artifact path**, **figure**.

### Tier 0 — Property tests (CI on every PR)

Already in `tests/test_core.py`. Claim IDs C-form-*. No paper numbers from smoke tests alone.

### Tier 1 — Core scientific spine

#### E1 — Factored vs global-`b` encoder
- **Question:** Does per-user `b` help preference prediction?
- **Status:** NEGATIVE (C-enc-1)
- **Command:** `python experiment_1.py --seeds 0 1 2`
- **Artifact:** `experiments/runs/exp1_*.jsonl`

#### E1-strat — Difficulty-stratified contexts
- **Question:** Does C-enc-1 flip if context queries span easy/hard/mixed?
- **Prediction:** corr(b) rises; held-out logp may beat baseline at high spread
- **Kill:** corr(b) still ≈0 with oracle-stratified difficulty
- **Primary:** corr(b), held-out logp (pre-specify which is primary before run)

#### E2 / E2-easy / E2-random — Consistency recovery
- **Supports:** C-enc-2,3,4
- **Command:** `python experiment_2.py`
- **Rule:** Any paper mention of hard-query corr must cite random caveat in same paragraph

#### E3-power — Powered elicitation bake-off
- **Question:** Does VOI beat random and BALD on **final regret**?
- **Protocol lock:** true-regret oracle; paired bootstrap; ≥10 seeds × 15 users; d∈{4,8,12}
- **Primary:** final-query regret  
- **Secondary:** queries-to-threshold (+ tie + censor rates)
- **Strategies:** random, bald, voi (particle), oracle
- **Artifact:** `experiments/runs/exp3_power_d{d}.jsonl`
- **Figure:** regret bars + curves (Claim IDs in caption)
- **Until this lands `supported`:** abstract must say preliminary / omit BALD win

#### E3-ablate — Estimator ablation (done)
- **Supports:** C-main-3, C-main-6 (NS), C-main-7 (killed)
- **Artifact:** `experiments/runs/exp3_voi_ablation2.jsonl`

### Tier 2 — Creative mechanistic suite (paper depth)

#### E-eq — Policy equivalence audit
- **Question:** How often do two unit rewards share an optimal policy?
- **Method:** Sample M unit `w` on the grid; cluster by `π*(w)` fingerprint; report #policies / M and mean cluster size
- **Output:** histogram + “decision-irrelevant radius” sketch
- **Why creative:** turns the intro slogan into a measured domain statistic reviewers can check

#### E-eq-d — Equivalence vs dimension
- Sweep `d`; plot fraction of random pairs with identical `π*`
- **Prediction:** non-monotone or increasing irrelevance with `d` — cite next to C-mot-2

#### E-decouple — Structural decoupling (queue #2)
- **Domain:** split features into **task** (affect transitions/rewards on reachable set) vs **decoy** (only appear in query segments / unreachable cells)
- **Factor:** decoy fraction ρ ∈ {0, 0.25, 0.5, 0.75}
- **Prediction:** (final regret BALD − VOI) **increases** in ρ
- **Kill:** flat or decreasing → gap-5 premise weak on this domain
- **Primary:** paired BALD−VOI final regret vs ρ (interaction test / bootstrap on slopes)
- **Figure:** the paper’s “why VOI” plot

#### E-prior — Population prior vs alternatives
- Priors: (a) population mixture particles, (b) uniform on sphere / Gaussian, (c) particles = true user only (cheat upper prior)
- Same acquisition (VOI particle) + same eval
- **Supports** abstract “population prior” sentence only if (a)≺(b) on regret

#### E-prior-shift — Prior misspecification
- Fit modes on population A; evaluate on users from shifted modes B
- **Prediction:** VOI degrades toward BALD; report honesty table

#### E-act — Decision rule ablation
- After each query, act with: mean `w` | MAP particle | sample particle | softminimax over top-k particles
- Acquisition fixed (VOI or BALD)
- **Prediction:** better action rule shrinks oracle gap more than acquisition tweaks (follows queue #1 verdict)

#### E-calib — Is VOI calibrated?
- For each asked query, record predicted Δloss vs realized true-regret drop
- Scatter + Spearman ρ; reliability bins
- **Kill:** ρ≈0 → stop calling the score “value of information” without caveat

#### E-cover — Posterior coverage diagnostic
- Metric: cosine distance from `w_true` to nearest particle / to posterior mean
- Regress residual final regret on coverage
- Explains oracle gap without hand-waving

#### E-neg — Negative control
- Generate labels from `w_junk` orthogonal to task-relevant subspace
- **Prediction:** VOI ≤ random (should not help)
- Stops us from shipping a method that “always looks good”

#### E-budget — Full budget curves
- Fix seeds/users/candidates; record regret after t=1…T for all methods
- One multi-panel figure → every “after 10 queries” number is a slice of this

#### E-diag-bald — BALD pathology showcase
- Pick queries maximizing BALD; plot their true-regret VOI and human-difficulty (gap)
- Anecdotal but figure-worthy: high BALD, low decision value

#### E-ternary — Ranking width
- K=2 vs K=3 at matched query count; optional matched information
- Primary: final regret / cos(w) identification error

#### E-mismatch — Non-PL users
- Answer models: PL (control), satisficing, lexicographic feature tie-break, fatigue (`b` decay)
- Table: degradation of each acquisition method

### Tier 3 — Stretch (only if Tier 1–2 green)

- Continuous control proxy (simple PointMass) with approximate regret
- Offline human clip study and/or live robot-arm trajectory comparisons (`docs/hardware.md`)

---

## C. Paper figure plan (each figure = claims)

| Figure | Content | Claim IDs |
|--------|---------|-----------|
| Fig 1 | Method schematic + policy-equivalence cartoon with E-eq statistic | C-mot-1 |
| Fig 2 | E-decouple: BALD−VOI vs ρ | C-mech-1, C-main-2 |
| Fig 3 | E3-power bars + curves (final regret) | C-main-1,2,3 |
| Fig 4 | E-prior prior ablation | C-mech-2 |
| Fig 5 | E-act decision-rule vs oracle gap | C-mech-4, C-main-3 |
| Fig 6 | E2 difficulty recovery + E1 negative | C-enc-* |
| Fig App | E-calib, E-cover, E-neg, E-mismatch | C-mech-5,6,7,10 |

---

## D. Execution order (shipping-aware)

1. **E-eq + E-eq-d** — cheap; hardens intro (do first)
2. **E3-power** — settle C-main-1/2 with true-regret oracle or rewrite abstract
3. **E-decouple** — flagship mechanistic result
4. **E-prior + E-act** — explain usability + oracle gap
5. **E-calib + E-cover + E-neg** — reviewer-proofing
6. **E1-strat, E-ternary, E-mismatch** — secondary

Long runs: background + log. Pre-specify primary metric in `RESULTS.md` **before** launch.

---

## E. Template for RESULTS.md entries

```
### <Experiment ID> — <date> — claim <Claim IDs>
- Pre-specified primary: ...
- Seeds: ...
- Commit: ...
- Command: ...
- Numbers: ...
- Paired bootstrap: ...
- Verdict: supported | NS | kill | rewrite paper
```
