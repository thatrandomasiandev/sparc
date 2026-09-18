# Project context — decision-relevant elicitation in preference-based RL

**Audience:** a coding agent (Cursor, Claude Code, or similar) that will write and run
experiments in this repository. You have no memory of how the project got here. This file
is the whole history.

**Companion files:** `AGENTS.md` holds the short version — invariants, repo map, standards.
`docs/CLAIMS.md` holds the claim↔experiment matrix (every paper number must trace here).
If documents disagree: `AGENTS.md` wins on invariants; this file wins on scientific
context and reported numbers; `docs/CLAIMS.md` wins on which experiment backs which claim.

**Last updated:** 17 September 2026, after the powered run of experiment 3.

---

## 1. What we are trying to publish

### 1.1 The one-paragraph version

A robot learns what a person wants by showing them pairs of behaviors and asking which is
better. Every method in this field chooses the *next* question by asking which comparison
reveals the most about the person's reward function. We claim that is the wrong target,
because a large fraction of what a reward function encodes never changes what the robot
actually does — two quite different reward estimates frequently induce the same policy. So
the robot spends a person's limited patience resolving uncertainty that has no behavioral
consequence. We select queries instead by *expected reduction in policy regret*, computed
under a prior learned from a population of previous users, and we test whether that reaches
good behavior in fewer questions.

### 1.2 Target venue and timeline

RSS 2027, Athens, 6–11 July 2027. The submission deadline is still listed as TBA;
historically it lands in late January or early February, so plan against roughly 19 weeks
from mid-September 2026. CoRL 2027 (call usually drops around April) is the fallback.
IROS 2027 (deadline 1 March 2027) is the floor.

The author is a USC undergraduate applying to PhD programs in fall 2027. That constrains
things in a way worth stating: **a finished, honest, modest paper is worth more than an
ambitious unfinished one.** Optimize for shipping.

### 1.3 What is and is not novel — read this before claiming anything

This project has narrowed four separate times under literature search. That is normal, and
the current claim is the survivor. Do not overstate it.

**Not novel.** The concept of eliciting only the reward information that changes the
optimal policy is from 2009: Regan & Boutilier, *Regret-Based Reward Elicitation for
Markov Decision Processes* (UAI 2009). They state the goal explicitly — minimize the
precision with which a reward must be specified while still producing near-optimal
policies — and select queries by minimax regret reduction. *Model-Free Preference
Elicitation* (IJCAI 2024) is the modern instantiation, scoring queries by expected value
of information toward recommendation quality, but in recommender systems with no policy
and no MDP. Alizadeh et al. continue the same line with approximate regret-based
elicitation in MDPs.

**What remains open, and is our contribution.** The decision-theoretic criterion has never
been combined with (a) human trajectory-segment comparisons rather than bound queries on
reward values, (b) deep or continuous control rather than small tabular and factored MDPs
with exact minimax-regret optimization, or (c) **a learned prior over a population of
previous users**. The third is the real opening. Regan & Boutilier search all of reward
space; we search only the part real users occupy, which is what makes the computation
affordable and the query count small.

**Related work that must be cited in the first paragraph**, not buried: Regan & Boutilier
2009, Model-Free Preference Elicitation (IJCAI 2024), and the acquisition lineage below.
A reviewer who knows the 2009 paper and sees it cited late will assume we did not know it.

---

## 2. The field, compressed

### 2.1 Roots

Christiano et al. 2017 (*Deep RL from Human Preferences*) is the origin of the modern
subfield. The motivating problem: how do you communicate a complex goal to an RL system
when you cannot write the reward down? The answer: show non-expert humans pairs of short
trajectory clips, ask which is better, fit a reward model. It worked with feedback on
under one percent of the agent's interactions, roughly an hour of human time. That number
is why the field exists.

The robotics branch started separately with a different concern. Sadigh et al. 2017
(*Active Preference-Based Learning of Reward Functions*, RSS) asked not only how to learn
from comparisons but *which* comparisons to request, formulating query selection as
maximum volume removal over reward-weight space. Physical robots cannot afford thousands
of queries the way a simulated agent can.

The statistical backbone is Bradley-Terry: the probability a human prefers A over B is a
sigmoid of the scaled reward difference. Nearly every paper either uses it or explains why
it is insufficient.

### 2.2 The feedback-efficiency line

Every paper here answers the same complaint — human feedback is the bottleneck.

- **PEBBLE** (ICML 2021) is the reference baseline. Two ideas: relabel the entire replay
  buffer whenever the reward model updates (because the reward keeps changing, so past
  experience becomes mislabeled), and pre-train the agent with unsupervised exploration so
  early queries compare meaningfully different behaviors rather than noise against noise.
- **SURF** (ICLR 2022) pseudo-labels confident unlabeled pairs and augments by temporal
  cropping.
- **Few-shot preference learning** (Hejna & Sadigh, CoRL 2022) meta-learns a reward prior
  across tasks with MAML and reports roughly 20× fewer queries than PEBBLE. It also
  selects queries by ensemble disagreement, and the authors note this produces comparisons
  "too difficult for humans to answer" — a recurring failure we must avoid.

Pattern: each finds free information outside the human's answers — unlabeled pairs, other
tasks, old demonstrations — to reduce how many answers are needed. Our population prior is
the same move applied to previous users.

### 2.3 The acquisition lineage — the thread we sit inside

This is the most important section for anyone writing acquisition code. It is a sequence
of corrections, each fixing a pathology in the previous objective.

| Criterion | Work | What it fixed | What broke |
| --- | --- | --- | --- |
| Volume removal | Sadigh et al. 2017 | First principled query selection for robot reward learning | The trivial query (two identical options) is a *global optimum* |
| Batch volume removal | Bıyık & Sadigh 2018 | Per-query optimization too slow for live interaction | Greedy batches are redundant; needs diversity heuristics |
| Information gain | Bıyık et al. 2019 (*Asking Easy Questions*, CoRL) | Trivial query becomes a global *minimum*; produces human-answerable questions | Optimizes parameter entropy, not behavior |
| Generalized / alignment | Bıyık et al. 2024 | Parameter entropy is wasteful — many parameters give one reward, many rewards give one behavior | Single user only |
| Ensemble disagreement | Hejna & Sadigh 2022 | Cheap, works with deep ensembles | Selects near-identical clips humans cannot judge |
| Query synthesis | 2026 | Generates queries in continuous space instead of scoring a pool | Single user only |
| When/how to ask (SPARQ) | 2026 | Chooses feedback modality and timing under human effort cost | Discrete states; single user; one simple task |

Two findings from this thread bear directly on our code:

1. **Information gain automatically produces human-answerable queries**, because the
   objective subtracts the human's expected answer uncertainty. This is a documented
   property verified in user studies, not speculation. It is why our BALD implementation
   subtracts the mean per-particle entropy.
2. **Parameter-space information gain is the wrong target.** Two independent literatures
   say so — Bıyık et al. 2024 for reward learning, and *Prediction-Oriented Bayesian
   Active Learning* (EPIG, AISTATS 2023) for active learning generally, which shows BALD
   can be *worse than random* when parameter uncertainty decouples from predictive
   uncertainty. This is the intellectual foundation of our VOI approach and should be
   cited as such.

### 2.4 The pluralism turn

Around 2023–2024 the field stopped treating annotator disagreement as noise and started
treating it as structure.

- **MaxMin-RLHF** (2024) Theorem 1 proves a single reward model's alignment gap grows with
  preference diversity and shrinks with minority representation. Cite this in the intro —
  it converts "the average annotator is nobody" from an intuition into a theorem.
- **DPL** (ICLR 2024) detects that hidden context exists via a distribution over utilities,
  but never recovers *whose*.
- **PAL** (2024) models K prototypical ideal points; each user is a weighted mixture.
  Roughly 20 samples per unseen user, sample complexity scaling in K.
- **VPL** (NeurIPS 2024) is the closest architectural precedent: a variational encoder maps
  a user's annotations to a latent `z`, and a latent-conditioned reward predicts
  preferences. Adapts from 2–8 **fixed survey queries**. Its authors explicitly name
  relaxing the fixed-query assumption as future work.
- **PREC** (2026, titled "Deployable") clusters users with a leaky-EM algorithm and trains
  one policy per cluster. MuJoCo only, scripted users.

Critically: every one of these evaluates on synthetic or semi-synthetic populations. VPL's
authors state the reason — realistic preference datasets with diverse users do not exist
at scale.

### 2.5 The five gaps we surveyed, and why we chose gap 5

A survey of roughly 210 papers by title and abstract, 31 read closely, produced five
candidate gaps:

1. **Nobody checks whether a preference exists.** An audit of two preference datasets
   found 57% non-attitudes and only 2% genuine stable preferences. No personalization
   method can output "this person has no preference here" — a posterior over `z` always
   integrates to one, so some `z` always wins.
2. **Personalized rewards are never tested by retraining on them.** *On the Fragility of
   Learned Reward Functions* shows frozen learned rewards often fail when a fresh agent
   trains on them, and that a larger sampler budget can make relearners worse. Every
   personalization paper grades on preference-prediction accuracy instead.
3. **The subfield runs on scripted users.** No preference-based personalization paper
   evaluates real humans on real robot hardware.
4. **Individual-level non-stationarity.** NS-DPO handles population drift but assumes
   everyone shifts together and does not model individuals. Personalization assumes each
   person is a fixed point. One tired user produces statistics resembling two users, and
   clustering methods would split them.
5. **Elicitation ignores what the robot will do.** ← **chosen**

Gap 5 was chosen because it is the most native to preference-based RL specifically rather
than to LLM alignment. Gaps 5 and 2 both *require a policy*, which is what separates PbRL
from RLHF on text. Gaps 1 and 4 are preference-modeling problems that work fine with no
agent anywhere; their evidence bases are LLM annotation datasets and a DPO paper
respectively.

---

## 3. Formal setup

### 3.1 Notation

- `phi(s)` ∈ R^d — feature vector of a state. Reward is linear: `r(s) = w · phi(s)`.
- `w` ∈ R^d — a user's preference direction, **always unit norm** (see invariant I1).
- `b` > 0 — a user's consistency / rationality. Higher means more reliable answers.
- A trajectory segment is summarized by its discounted feature count
  `f = sum_t gamma^t phi(s_t)`.
- A binary query is a pair of segments; `delta = f_A - f_B`.

### 3.2 Answer models

Binary (Bradley-Terry):

```
P(A preferred) = sigmoid( b * (w · f_A - w · f_B) ) = sigmoid( b * w · delta )
```

Ranking over K options (Plackett-Luce) — pick the best remaining repeatedly:

```
log P(order) = sum_{k=0}^{K-2} [ b*r_{order[k]} - logsumexp_{j>=k} b*r_{order[j]} ]
```

The two must agree exactly at K=2; there is a test asserting this.

### 3.3 Why `w` is normalized — the single most important design decision

Under Bradley-Terry only the **product** `b * ||w||` affects any answer probability. Double
`w` and halve `b` and every prediction is identical. If both are free parameters, then
"consistency" is not a real quantity — it is whatever scale the optimizer happens to leave
over, and every result about `b` is an artifact of our own parameterization.

Pinning `||w|| = 1` assigns all scale to `b` and makes consistency identifiable, **provided
queries span more than one difficulty level.** That proviso is why difficulty is a design
variable and not an afterthought. `likelihood.unit()` exists for this; never compute
`w @ phi` directly.

### 3.4 Acquisition functions

**BALD** — information about the reward parameters:

```
score(q) = H[ E_w p(y|w) ] - E_w H[ p(y|w) ]
```

High when particles disagree about the answer, with per-particle noise subtracted. That
subtraction is what makes the trivial query (two identical options) score exactly zero,
and there is a test asserting it.

**EPIG** — information about future predictions rather than parameters:

```
score(q) = E_{target x*} I( y_q ; y_{x*} )
         = E_{x*} sum_{a,c} p(a,c) log[ p(a,c) / (p(a)p(c)) ],  p(a,c) = E_w[p_w(a)p_w(c)]
```

Uses the fact that both answers are conditionally independent given the latent — which is
exactly right, because the person *is* the latent.

**VOI (ours)** — expected reduction in policy regret:

```
Loss(posterior) = E_{w~posterior}[ V*_w - V^{decision(posterior)}_w ]
score(q)        = Loss(current) - sum_y P(y) Loss(posterior | y)
```

where `decision(posterior)` is the optimal policy for the posterior mean reward. Cost: two
value iterations per candidate query. That expense is precisely why the 2009 work stayed in
tabular MDPs, and why our population prior matters — it shrinks the space we must search.

**Effort normalization.** When modalities differ in cost, divide by measured seconds:
`score(q) / cost(q)`. Costs are measured in a pilot study, never assumed. `QueryCosts`
raises `KeyError` rather than defaulting, deliberately.

---

## 4. Repository

```
plr/
  likelihood.py   Bradley-Terry + Plackett-Luce answer models. Holds invariant I1.
  users.py        Simulated populations with ground-truth (w, b).
  encoder.py      q(w, b | answers) — the factored variational encoder.
  acquisition.py  BALD, EPIG, effort normalization, difficulty scoring.
  mdp.py          Gridworld with exact value iteration and exact regret.
tests/
  test_core.py    Property tests asserting the PAPER'S CLAIMS, not "does it run".
experiment_1.py   Factored latent vs VPL-style single latent.
experiment_2.py   Is consistency recoverable, and under which query difficulty?
experiment_3.py   Decision-relevant elicitation: random vs BALD vs VOI vs oracle.
aggregate_exp3.py Multi-seed aggregation with paired bootstrap CIs.
```

Dependencies: PyTorch, NumPy, pytest. Nothing else. A reviewer reproducing this should not
need a conda solve.

### 4.1 Module contracts

**`likelihood.py`** — `unit(w)` normalizes along the last axis with a clamped norm.
`rewards(phi, w)` normalizes internally rather than trusting callers. `binary_logp` uses
`logsigmoid` because `log(sigmoid(x))` underflows once `b * gap` exceeds about 30, which
happens routinely for confident users on easy queries. `reward_gap` defines difficulty as
the smallest gap between consecutive sorted option rewards — small means hard.

**`users.py`** — `sample_population` takes `consistency_spread`, which is the experiment's
independent variable: 0.0 makes everyone equally reliable (the world VPL implicitly
assumes, where our contribution should vanish), higher values make reliability vary.
`answer()` samples from the same Plackett-Luce model the likelihood assumes, which is
deliberately generous to us — model mismatch is tested separately.

**`encoder.py`** — a DeepSets-style set encoder: embed each annotation, mean-pool (which
makes the encoder order-invariant, correct because a person is the same whether you asked
question 3 or 7 first), then two heads for `w` and `log b`. Per-annotation features are
direction (`best - mean(rest)`, the sufficient statistic for preference direction), context
(mean of options), and **spread plus option count** (the difficulty signal). Removing the
spread term is a valid ablation and a fatal bug — label it correctly. Log-std is clamped to
[-5, 2] to prevent posterior collapse (KL explodes) and noise explosion (likelihood becomes
meaningless); both failures present as "training diverged."

**`acquisition.py`** — `predictive` enumerates all K! outcomes (K stays small; nobody ranks
eight clips) and renormalizes in log space via softmax for stability at large `b`.

**`mdp.py`** — the key trick is `policy_value_matrix`: because reward is linear in `w`, the
value of a *fixed* policy is too, so `V^pi(w) = (I - gamma P^pi)^-1 Phi w`. Invert once per
policy, then evaluate under thousands of candidate rewards with one matmul. Only `V*(w)`
needs value iteration, and particle values are precomputed once because particles are
fixed — only their weights change.

---

## 5. Results so far — all of them, including the failures

### 5.1 Experiment 1 — factored latent vs single latent (NEGATIVE)

Trains the factored encoder against a VPL-style baseline whose consistency is one global
learned constant, on randomly generated queries, sweeping population consistency spread.

| spread | model | cos(w) | corr(b) | held-out logp |
| --- | --- | --- | --- | --- |
| 0.0 | baseline | 0.767 | n/a | −0.5146 |
| 0.0 | factored | 0.764 | n/a | −0.5227 |
| 0.4 | baseline | 0.766 | −0.073 | −0.5141 |
| 0.4 | factored | 0.756 | −0.036 | −0.5356 |
| 0.8 | baseline | 0.728 | −0.050 | −0.5618 |
| 0.8 | factored | 0.718 | −0.059 | −0.5856 |

No win at any spread, and consistency recovered at correlation ≈ 0. Taken alone this reads
as a dead premise.

### 5.2 Experiment 2 — is consistency recoverable at all? (EXPLAINS IT)

Estimates `b` by maximum likelihood with the **true** `w` supplied, under controlled query
difficulty. Supplying `w` isolates the question: if `b` cannot be recovered even when the
direction is known exactly, it certainly cannot be recovered when it is not.

| Query regime | corr(b̂, b) | median abs error |
| --- | --- | --- |
| Easy (large gaps) | 0.556 | **23.09** |
| Hard (near threshold) | **0.867** | 1.07 |
| Mixed (staircase-like) | 0.608 | **0.54** |
| Random (uncontrolled) | 0.753 | 0.63 |

Three readings. (1) The information exists — consistency is recoverable in principle.
(2) Easy queries are catastrophic: median error 23 against a true median of 3, because
someone who answers every easy question correctly is indistinguishable from a perfect
respondent, and the MLE runs to the top of the grid. (3) The bottleneck in experiment 1 is
therefore *joint estimation* of `w` and `b` from a small uncontrolled context set, not
identifiability.

**The caveat that must never be dropped:** random queries scored 0.753, better than
expected, which undercuts any claim that hard queries are strictly necessary. The
defensible claim is narrow — easy queries are catastrophic and *some* hard queries are
required.

### 5.3 Experiment 3 — decision-relevant elicitation (PRELIMINARY POSITIVE)

6×6 gridworld, linear reward over terrain features, exact optimal policy and exact regret.
Users drawn from a three-mode population prior, which is also the elicitor's particle
prior. Four strategies share the same posterior machinery, the same users, and the same
candidate queries.

**Small run, d=4, 48 user-runs.** Regret after ten queries: random 1.52, BALD 0.91, VOI
0.70, oracle 0.004. Queries to threshold: random 3.83, BALD 2.90, VOI 2.48, oracle 1.31.
Paired bootstrap: random−VOI +1.35 [+0.46, +2.23]; random−BALD +0.94 [+0.04, +1.88];
**BALD−VOI +0.42 [−0.38, +1.17] — not established.**

**Dimension sweep, 24 runs per cell.** BALD−VOI difference: +0.42 at d=4, +1.42 at d=8,
+1.00 at d=12. All point estimates favor VOI; all three CIs contain zero.

**Powered run, d=8, 150 user-runs (10 seeds × 15 users).**

| Metric | random | BALD | VOI | oracle |
| --- | --- | --- | --- | --- |
| Queries to threshold | 6.08 ± 0.34 | 5.77 ± 0.33 | 5.14 ± 0.33 | 2.21 ± 0.21 |
| % of runs reaching it | 66% | 69% | 75% | 95% |
| Final-query regret | 3.666 ± 0.340 | 3.390 ± 0.333 | **2.365 ± 0.264** | 0.249 ± 0.065 |

Paired bootstrap on queries-to-threshold:

- **BALD − VOI: +0.63, 95% CI [+0.02, +1.25]** — excludes zero, barely.
- random − VOI: +0.94, CI [+0.18, +1.71] — VOI beats random.
- random − BALD: +0.31, CI [−0.43, +1.03] — **BALD does not beat random at d=8.**

**How to report this.** The queries-to-threshold result is real but fragile: the CI lower
bound is +0.02, 43% of runs tie, and only 66–75% of runs reach the threshold at all, so
censoring at 11 may bias the estimate. The **final-regret comparison is much cleaner** —
VOI 2.365 ± 0.264 against BALD 3.390 ± 0.333 is roughly a three-standard-error gap — and
should probably become the primary metric, with queries-to-threshold secondary. Decide
this and write it down *before* the next run, not after seeing results.

**The oracle gap is the other headline.** Oracle reaches the threshold in 2.21 queries
against VOI's 5.14, and ends at regret 0.249 against 2.365. Our acquisition captures maybe
half of what is available even with perfect information about the user. That is
algorithmic headroom entirely independent of the comparison to BALD, and it is probably
the most promising place to improve.

### 5.4 Retracted claims — do not repeat these

- **"BALD performs worse than random."** The first single-seed run of experiment 3 showed
  this, and it would have echoed the published EPIG finding nicely. Across three seeds at
  d=4 it does not hold. At d=8 with 150 runs, BALD is statistically indistinguishable from
  random — which is a *different and weaker* claim, and is the only version supportable.
- **"The factored latent beats VPL."** Experiment 1 says no. Any future version of this
  claim needs difficulty-stratified context sets and a fresh run.

---

## 6. Invariants — breaking these silently invalidates results

**I1. `w` is always normalized before computing rewards.** See §3.3. Never `w @ phi`.

**I2. Difficulty is defined without reference to `b`.** It is a property of the question,
not the person, which is what lets us schedule difficulty before knowing the user.

**I3. Query cost is measured, never assumed.** `QueryCosts` raises rather than defaulting.
Do not add a default.

**I4. The encoder's per-annotation features must include option spread.** Without it the
consistency head is guessing.

**I5. Held-out scoring.** Context set and target set are disjoint, always.

**I6. Every result reproducible from one integer seed.** If `--seed N` cannot regenerate
it, it does not go in the paper.

**I7. No claim from a single seed.** This project has already produced one retraction from
violating it. Minimum three seeds for exploratory claims, and a paired bootstrap for
anything that goes in the paper.

---

## 7. Statistical methodology — non-negotiable

The strategies see the same users and the same candidate queries, so **runs are paired**.
Comparing independent standard errors throws that structure away and badly understates the
evidence; it is also how you end up reporting a "significant" result that is an artifact.

- Report paired differences with a bootstrap CI (10,000 resamples), not unpaired SEs.
- Report the tie rate. At d=8 it is 43%, which matters for interpreting the mean difference.
- Report the censoring rate when using queries-to-threshold. Runs that never reach the
  threshold are counted as `n_queries + 1`, which is a lower bound on their true value and
  biases the comparison toward whichever method reaches it more often.
- Pre-specify the primary metric before running. Choosing it afterwards is how a +0.02 CI
  lower bound becomes a headline.
- When a result is marginal, say "preliminary" and run more seeds. Compute is cheap here;
  a retraction is not.

---

## 8. Experiment queue

Run in order. Each has a stated kill condition. Report it honestly when hit.

1. **Improve the VOI estimator** (highest value). ~~The current version uses only the
   posterior-mean policy…~~ **Done (2026-09-17) — partial kill.** Implemented particle-policy
   loss + greedy depth-2 lookahead + true-regret oracle. Particle helps vs mean on point
   estimate (NS at n=24); lookahead does not help; **oracle gap remains** (~1.9 regret on
   primary metric). See RESULTS.md. Treat further acquisition tweaks as low priority;
   try decision-rule changes or queue #2 next.
2. **Structural decoupling domain.** Build a domain where reward dimensions provably cannot
   affect the optimal policy (e.g. terrain the agent cannot reach, or features visible in
   query segments but absent from the task region). *Prediction:* VOI's advantage over
   BALD grows sharply. *Kill:* no growth — the premise of gap 5 does not hold and we should
   reconsider.
3. **Difficulty-stratified context sets in experiment 1.** Does consistency become
   recoverable when the encoder sees a difficulty-spread context set instead of a random
   one? *Kill:* still ≈0 with oracle-stratified difficulty; the architecture needs
   rethinking, not tuning.
4. **Ternary vs binary at matched query count.** Does adding three-option rankings improve
   identification of `w`? (Cost comparison waits for the human study.)
5. **Model mismatch.** Sample answers from a response model that is *not* Plackett-Luce —
   satisficing, lexicographic tie-breaking, fatigue-driven. *Kill:* estimates collapse,
   meaning the method only works on its own generative assumptions. A reviewer will test
   this, so we should first.
6. **Non-stationary consistency.** Let `b` drift downward within a session. Exploratory,
   highest-risk, most novel. Do not let it block 1–5.

### 8.1 Hardware / human pilot — PARKED

Lab robots are available in principle, but the author does not currently operate them.
**Do not pursue live-robot experiments for RSS** unless this section is explicitly
un-parked. Code under `plr/hardware/` may support a future offline clip study; it is not
on the critical path. Sim results are the paper.

Details: `docs/hardware.md`.

---

## 9. Code standards

This code will be released with the paper and read by reviewers trying to break it.

- **Architectural stability over features.** One module, one job, contract in the
  docstring. A boring correct implementation beats a clever one.
- **Shape comments on every tensor operation**, in the form `# (B, N, K, d)`. Most bugs
  here are silent broadcasting errors; shapes in comments are the cheapest defense.
- **Validate at the boundary.** Raise on impossible inputs (non-positive `b`, mismatched
  user counts) rather than letting them become a plausible wrong number.
- **Docstrings explain WHY.** The reader can see what the line does; they cannot see which
  failure mode it prevents. Match the voice of the existing files.
- **Tests assert claims.** `test_trivial_query_has_zero_information` exists because
  trivial-query degeneracy sank volume-removal selection. Every test should trace to
  something the paper asserts.
- **Long runs go to background** with output to a log file. The powered d=8 run took about
  25 minutes.

---

## 10. Things not to do

- Do not add a deep reward network yet. Linear reward keeps identifiability analysis
  tractable and makes exact regret computable. Non-linear is a later ablation.
- Do not tune hyperparameters to rescue a negative result before understanding it.
  Experiment 1's failure was informative precisely because it was not tuned away.
- Do not report a result without its caveat. The 0.753 random-query number belongs in
  every summary that cites the 0.867 one.
- Do not cite Regan & Boutilier's or MaxMin-RLHF's theorems as ours.
- Do not claim novelty for the decision-relevant concept. Claim it for the instantiation:
  human comparisons, continuous control, population prior.
- Do not let the oracle out of the experiments. Without it, you cannot tell whether a
  method is good or the problem is easy.

---

## 11. Open questions for the advisor (Prof. Bıyık, LIRA Lab)

Bring these in priority order. The first is the one that could invalidate a whole branch.

1. Is the preference/consistency confound real, or is there a normalization that dissolves
   it? (Our answer: `||w|| = 1` resolves it, and difficulty variation identifies `b`. He
   will know in thirty seconds whether that holds.)
2. Is contradicting *Asking Easy Questions* defensible as a boundary condition — easy
   queries are right for a single user with no consistency parameter, wrong when you must
   also estimate consistency — rather than as a refutation?
3. Given Regan & Boutilier 2009, is the population-prior instantiation a sufficient
   contribution for RSS, or does this need theory of its own?
4. Is a ternary-query interface realistic in a human study, or does showing three clips
   break the interaction?
5. RSS in February or CoRL in spring, given human-study timing?

---

## 12. Glossary

- **BALD** — Bayesian Active Learning by Disagreement. Information gain about model
  parameters. Houlsby et al. 2011, which also applied it to preference learning.
- **EPIG** — Expected Predictive Information Gain. Information about predictions on a
  target distribution rather than about parameters.
- **VOI** — Value of Information. Here: expected reduction in policy regret.
- **Regret** — expected shortfall in return from acting on an estimated reward instead of
  the true one. The quantity this project is ultimately about.
- **PbRL** — preference-based reinforcement learning. Distinguished from RLHF-on-text by
  having a policy.
- **Plackett-Luce** — ranking model; repeatedly pick the best remaining option under a
  softmax over scaled rewards. Reduces to Bradley-Terry at two options.
- **Consistency / rationality (`b`)** — how reliably a user reports their own preference.
  High `b` means sharp, near-deterministic answers.
- **Oracle** — a cheating strategy that scores queries against true regret. Used as a
  ceiling, never as a baseline to beat.
