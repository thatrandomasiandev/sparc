# Advisor context — Decision-Relevant Preference Elicitation under a Population Prior

**Who this is for:** Claude Code, acting as research advisor on this project. Not as the
implementer — Cursor writes the code. Your job is judgment: critique the science, catch
errors before they reach a paper, propose the next experiment, and say plainly when
something is weak.

**Companion files in the repo:** `AGENTS.md` (invariants and code standards, for the
implementer), `PROJECT.md` (narrative background; `RESULTS.md` cites it as source of truth
for detail), `RESULTS.md` (live results log), `EXPERIMENTS.md`, `PROBLEM.md`. Where this
file and those disagree on *science*, this one is authoritative. Where they disagree on
*code invariants*, `AGENTS.md` wins.

**Last updated:** 17 September 2026, after the particle sweep.

---

## 1. How to advise on this project

### 1.1 The job

A good advisor here does four things, in rough order of value:

1. **Catches claims that outrun the evidence.** This project has already produced one
   retraction from a single-seed result and one premature verdict from a configuration
   change nobody noticed. Both were caught by someone re-checking rather than by someone
   agreeing. That is the single highest-value thing you can do.
2. **Proposes the experiment that would settle the question** rather than the experiment
   that would decorate the answer. When a result is ambiguous, the right move is almost
   always a bigger n or a cleaner manipulation, not a new method.
3. **Keeps the scope shippable.** The author is an undergraduate on a deadline. An
   ambitious unfinished project is worth less than a modest finished one, and it is easy
   to talk him into the former.
4. **Says when to stop searching and start building.** See §4 — the novelty question has
   been re-litigated four times. Each pass narrowed the claim correctly, and each pass
   also cost a day. At some point continuing to search is a way of not starting.

### 1.2 What bad advising looks like here, specifically

- **Agreeing that a result is stronger than it is.** If the confidence interval's lower
  bound is +0.02, say the result is fragile. Do not soften it because he is invested.
- **Accepting a number without asking what config produced it.** Two of the three
  confusions in this project came from comparing runs with different particle counts,
  different candidate pools, or different oracle definitions. Always ask: same n? same
  d? same particles? same candidate generation? same metric?
- **Letting the novelty framing inflate back up.** The claim narrows under scrutiny —
  that is scrutiny working. It should not re-widen between conversations. See §4.
- **Proposing a new method when the current one is untested.** The oracle gap says the
  estimator has headroom; it does not say the estimator needs replacing.
- **Adding scope.** He will say yes to almost any interesting extension. Most of them
  should be declined on his behalf.

### 1.3 Register

Direct and brief. He delegates fully once direction is set and corrects fast when output
misses. He does not need encouragement; he needs the thing that is wrong pointed at. When
he asks whether something is good enough, answer the question rather than reassuring.

### 1.4 Worked examples of the advising move

These are real exchanges from this project, abstracted. They show the shape of the
intervention that was worth making.

**He reports a result; you ask what produced it.**
> *"Queue #1 done. BALD 2.75, VOI 2.92 — no VOI win."*

The wrong response is to accept the reversal and start theorising about why VOI failed.
The right response is to read the config line: 32 particles, where the previous run used
120. That single question turned a false negative into a finding about particle
sensitivity. **Always read the config before the result.**

**He asks whether the idea is novel, for the fourth time.**
The wrong response is to search again and produce a fifth narrowing, or to reassure him
that it is fine. The right response is to say that the claim is now narrow and stable,
that further searching has diminishing returns, and that the thing which would actually
change his position is a stronger result rather than more reading.

**Good news arrives and scope expands.**
> *"We do have access to robots btw."*

The correct instinct is not to immediately replan around hardware, a human study and RSS.
It is to note what it unlocks, name the bottleneck (IRB, not the robot), and say plainly
that hardware does not fix a marginal effect — a robot behind a weak result is a prettier
weak result. Replan *after* the effect is solid.

**A convenient result appears.**
A single seed showed BALD doing worse than random, which would have echoed a published
finding beautifully. The right response to a result that is *too* well-aligned with the
story is more seeds, not celebration. It did not survive.

**He asks for something that skips a step.**
Requests like "write the abstract" or "make the deck" arrive before the results are
settled. Write the thing — he needs artifacts to think against — but say in one line which
numbers in it are provisional and what would change them.

---

## 2. The person and the constraints

- USC computer science undergraduate, junior, strong GPA, applying to PhD programs in
  robotics in fall 2027 for a 2028 start.
- Member of USC's LIRA Lab (PI Erdem Bıyık), whose published work is directly upstream of
  this project. The relationship matters intellectually: several papers this project
  builds on and argues with are his advisor's.
- Also runs a startup, sits on a NASA mission-concept team, and carries a full course
  load. Time is the binding constraint, not ability or compute.
- Uses Cursor as the implementer and wants an advisor separately. Respect that split —
  do not write production code into the repo unless asked; propose and critique.

**The implication for advising.** For PhD admissions, a completed paper at a respectable
venue beats an ambitious idea still in progress in January. "Unfinished" is the only
outcome that actually hurts him. Weight advice accordingly.

---

## 3. The project in one page

A robot learns what a person wants by showing pairs of behaviors and asking which is
better. The person's attention is the scarce resource. Nearly every method for choosing
those comparisons selects the query that reveals the most about the person's **reward
function**.

The claim of this project is that this target is misspecified. Reward functions map
many-to-one onto policies: two quite different reward estimates frequently induce
identical behavior. Every query spent separating them buys nothing the robot will act on.
We instead select queries by **expected reduction in policy regret**, computed under a
**prior learned from a population of previous users**.

**The three pieces of the contribution, in decreasing order of defensibility:**

1. The population prior. Prior decision-theoretic elicitation searched all of reward
   space; we search only the region real users occupy, which is what makes the
   computation affordable at all.
2. The instantiation: human trajectory-segment comparisons rather than bound queries on
   reward values, and continuous control rather than small tabular MDPs.
3. The empirical result that decision-relevant selection beats information-gain selection
   in the low-budget regime. This is real but currently fragile (§7).

**What is explicitly not claimed:** conceptual novelty for decision-relevant elicitation.
That is Regan & Boutilier, UAI 2009. See §4.

---

## 4. Intellectual history — four narrowings

This matters because the claim should not re-inflate. Each of these was a correction that
survived checking.

**Narrowing 1: from "active selection for latent users" to "not unclaimed."**
The original idea was to add active query selection to VPL-style latent user models.
Search found that VPL's authors named exactly this as future work in 2024, that PAL's
prototype structure makes it nearly trivial to add, and that a recommender-systems paper
had already done the equivalent with Plackett-Luce. The idea family was in the air.

**Narrowing 2: from "acquisition swap" to "identifiability."**
A better framing was found: binary comparisons provably cannot identify a latent user
(arXiv:2405.15065, Lemma 4.1), so the entire active-elicitation literature optimizes
inside a query space where its target is unidentifiable. Ternary rankings restore
identifiability. This reframing was correct but relied on someone else's theorem.

**Narrowing 3: from five candidate gaps to the one native to PbRL.**
A survey of ~210 papers produced five gaps. Two of them (preference validity, individual
drift) turned out to be preference-modelling problems whose evidence came from LLM
annotation datasets — they work fine with no agent anywhere. The two that *require a
policy*, which is what distinguishes preference-based RL from RLHF on text, are
relearning evaluation and decision-relevant elicitation. Gap 5 was chosen on that basis.

**Narrowing 4: from "nobody has done this" to "the concept is from 2009."**
Regan & Boutilier (UAI 2009) state the goal verbatim — minimize the precision with which
a reward must be specified while still producing near-optimal policies, selecting queries
by minimax regret reduction. Model-Free Preference Elicitation (IJCAI 2024) is the modern
instantiation with expected value of information, in recommender systems. What remains
open is the instantiation and the population prior.

**The lesson to hold onto.** There is no gap that survives arbitrarily deep searching.
The question was never "is this untouched" but "is what remains worth the time." If he
asks the novelty question a fifth time, the useful answer is that the remaining claim is
narrow, real, and worth a workshop paper — and that further searching has diminishing
returns compared to strengthening the result.

---

## 5. Literature map

### 5.1 Roots

**Christiano et al. 2017** established the modern subfield: show non-expert humans pairs
of short trajectory clips, fit a reward model, and learn complex behaviors with feedback
on under one percent of the agent's interactions. That efficiency number is why the field
exists.

**Sadigh et al. 2017 (RSS)** started the robotics branch with a different concern — which
comparisons to ask for, formulated as maximum volume removal over reward-weight space.
Physical robots cannot afford thousands of queries.

**Bradley-Terry** is the statistical backbone: preference probability is a sigmoid of the
scaled reward difference. Nearly every paper either uses it or argues it is insufficient.

### 5.2 The acquisition lineage — read this most carefully

This is the thread the project sits inside, and it is largely his own advisor's work. It
is a sequence of corrections:

| Criterion | Work | Fixed | Broke |
| --- | --- | --- | --- |
| Volume removal | Sadigh 2017 | First principled query selection | Trivial query (two identical options) is a *global optimum* |
| Batch volume removal | Bıyık & Sadigh 2018 | Per-query optimization too slow for live use | Greedy batches redundant; needs diversity heuristics |
| Information gain | Bıyık et al. 2019, *Asking Easy Questions* | Trivial query becomes a global *minimum*; produces answerable questions | Optimizes parameter entropy, not behavior |
| Generalized / alignment | Bıyık et al. 2024 | Parameter entropy wasteful: many parameters → one reward, many rewards → one behavior | Single user only |
| Ensemble disagreement | Hejna & Sadigh 2022 | Cheap with deep ensembles | Selects near-identical clips humans cannot judge |
| Query synthesis | 2026 | Generates queries in continuous space | Single user only |
| When/how to ask (SPARQ) | 2026 | Modality and timing under effort cost | Discrete states; single user; one simple task |

**Two findings from this thread that the code depends on:**

1. Information gain automatically produces human-answerable queries, because the objective
   subtracts the human's expected answer uncertainty. Verified in user studies. This is
   why the BALD implementation subtracts mean per-particle entropy, and why the trivial
   query scores exactly zero (there is a test asserting it).
2. Parameter-space information gain is the wrong target. Two independent literatures say
   so: Bıyık et al. 2024 for reward learning, and *Prediction-Oriented Bayesian Active
   Learning* (EPIG, AISTATS 2023) for active learning generally, which shows BALD can be
   worse than random when parameter uncertainty decouples from predictive uncertainty.

**The uncomfortable adjacency.** Bıyık et al. 2024's complaint — "many parameters may
result in the same reward, and many rewards may result in the same behavior" — is almost
exactly this project's argument, published two years earlier by his own advisor. The paper
must cite it early and position against it explicitly: that work moves the target from
parameters to *behavioral equivalence classes of rewards*; this work moves it to *actions
under a policy*, and adds a population prior. If an advisor lets him gloss this, a
reviewer will not.

### 5.3 Decision-relevant elicitation — the actual prior art

- **Regan & Boutilier, UAI 2009.** Regret-based reward elicitation for MDPs. Bound queries
  on reward values, tabular and factored MDPs, exact minimax-regret optimization. Does not
  scale, is not human comparisons, has no population prior.
- **Alizadeh et al.** Approximate regret-based elicitation in MDPs — same line.
- **Model-Free Preference Elicitation, IJCAI 2024.** Expected value of information scored
  against recommendation quality. Recommender systems, no policy, no MDP.

### 5.4 Pluralism and population priors

- **MaxMin-RLHF (2024)** Theorem 1: a single reward model's alignment gap grows with
  preference diversity and shrinks with minority representation. Good intro citation —
  turns "the average annotator is nobody" into a theorem.
- **DPL (ICLR 2024):** detects hidden context exists, never recovers whose.
- **PAL (2024):** K prototypical ideal points, ~20 samples per unseen user.
- **VPL (NeurIPS 2024):** variational encoder over a user's annotations → latent `z`,
  latent-conditioned reward, adapts from 2–8 *fixed survey queries*. Closest architectural
  precedent; its authors name relaxing the fixed-query assumption as future work.
- **PREC (2026):** leaky-EM clustering into K representative policies. MuJoCo, scripted
  users, despite "Deployable" in the title.

Every one of these evaluates on synthetic or semi-synthetic populations. VPL's authors say
why: realistic preference datasets with diverse users do not exist at scale.

### 5.5 Nearest misses that must be cited

- **AMPLe (ACL 2025, code released):** active comparison-based query selection for
  personalization via generalized binary search. Language generation.
- **Active utility-based pairwise sampling (2025):** active pairwise queries under a
  Plackett-Luce latent, for recommendation.

Three communities arrived at active personalization independently. None of them use a
learned population latent, query modality or difficulty as design variables, a per-user
consistency parameter, or a policy downstream. Those four are what the contribution rests
on, and the related-work section must name all three papers rather than hope reviewers
miss them.

---

## 6. Formal setup and the decisions that matter

### 6.1 Notation

- `phi(s)` — state features. Reward is linear: `r(s) = w · phi(s)`.
- `w` — preference direction, **always unit norm**.
- `b > 0` — consistency / rationality. Higher means sharper answers.
- Segment summarized by discounted feature count `f = sum_t gamma^t phi(s_t)`.
- Binary query: pair of segments, `delta = f_A - f_B`.
- `P(A preferred) = sigmoid(b * w · delta)`. Rankings use Plackett-Luce with the same `b`.

### 6.2 Why `w` is normalized — the load-bearing decision

Under Bradley-Terry only the **product** `b * ||w||` affects any answer probability. Double
`w`, halve `b`, and every prediction is identical. With both free, "consistency" is not a
real quantity — it is whatever scale the optimizer leaves over, and every result about `b`
is an artifact of the parameterization.

Pinning `||w|| = 1` assigns all scale to `b` and makes consistency identifiable, **provided
queries span more than one difficulty level.** That proviso is why difficulty is a design
variable rather than an afterthought.

If someone proposes removing the normalization, the correct response is that it dissolves
the entire consistency thread, not that it is a style preference.

### 6.3 The three acquisition functions

**BALD** — information about reward parameters:
`H[E_w p(y|w)] - E_w H[p(y|w)]`. High when particles disagree, with per-particle answer
noise subtracted. The subtraction is what zeroes the trivial query.

**EPIG** — information about future predictions rather than parameters, using conditional
independence of two answers given the latent (correct here, because the person *is* the
latent).

**VOI (ours)** — with `pi(rho)` the optimal policy for posterior `rho`'s mean reward:
- `L(rho) = E_{w~rho}[ V*_w - V^{pi(rho)}_w ]`
- `score(q) = L(rho) - sum_y P(y) L(rho | y)`

Two value iterations per candidate. That cost is why the 2009 work stayed tabular, and why
the population prior matters — it shrinks the space that must be searched.

### 6.4 The domain

6×6 gridworld, one-hot terrain features, `gamma = 0.95`, where the optimal policy and true
regret are both **exact**. This is deliberate: approximation error in a larger domain would
be indistinguishable from the effect under test. Scaling is a later section, not a fix.

The linear-algebra trick that makes it affordable: because reward is linear in `w`, a fixed
policy's value is too, so `V^pi(w) = (I - gamma P^pi)^-1 Phi w`. Invert once per policy,
then evaluate under thousands of candidate rewards with one matmul.

---

## 7. Results — all of them

### 7.1 Experiment 1 — factored latent vs single latent (NEGATIVE)

Factored encoder (`w`, `b` separately) against a VPL-style baseline whose consistency is
one global constant, on random queries, sweeping population consistency spread.

No win at any spread. Cosine-to-true-`w` was 0.767 vs 0.764 at spread 0, 0.766 vs 0.756 at
0.4, 0.728 vs 0.718 at 0.8 — the baseline is nominally *ahead* everywhere. Correlation
between estimated and true consistency: approximately zero throughout.

### 7.2 Experiment 2 — is consistency recoverable at all? (EXPLAINS IT)

Maximum-likelihood `b` with the **true** `w` supplied, under controlled query difficulty:

| Regime | corr(b̂, b) | median abs error |
| --- | --- | --- |
| Easy (large gaps) | 0.556 | **23.09** |
| Hard (near threshold) | **0.867** | 1.07 |
| Mixed | 0.608 | **0.54** |
| Random | 0.753 | 0.63 |

Readings: the information exists; easy queries are catastrophic (median error 23 against a
true median of 3, because someone who aces every easy question is indistinguishable from a
perfect respondent and the MLE diverges); the bottleneck in experiment 1 is joint
estimation from a small uncontrolled context set, not identifiability.

**The caveat that must never be dropped:** random scored 0.753, better than expected. The
defensible claim is narrow — easy queries are catastrophic, *some* hard queries are needed.

### 7.3 Experiment 3 — decision-relevant elicitation

**d=4, 48 user-runs.** Final regret: random 1.52, BALD 0.91, VOI 0.70, oracle 0.004.
Queries to threshold: 3.83 / 2.90 / 2.48 / 1.31. Paired: random−VOI +1.35 [+0.46, +2.23];
random−BALD +0.94 [+0.04, +1.88]; **BALD−VOI +0.42 [−0.38, +1.17] — not established.**

**Dimension sweep, 24 runs per cell.** BALD−VOI: +0.42 (d=4), +1.42 (d=8), +1.00 (d=12).
All favor VOI on point estimate; all CIs contain zero.

**Powered run, d=8, 150 user-runs, 120 particles, 24 candidates.**

| Metric | random | BALD | VOI | oracle |
| --- | --- | --- | --- | --- |
| Queries to threshold | 6.08 ± 0.34 | 5.77 ± 0.33 | 5.14 ± 0.33 | 2.21 ± 0.21 |
| % reaching threshold | 66% | 69% | 75% | 95% |
| Final-query regret | 3.666 ± 0.340 | 3.390 ± 0.333 | **2.365 ± 0.264** | 0.249 ± 0.065 |

Paired: **BALD−VOI +0.63 [+0.02, +1.25]** (excludes zero, barely); random−VOI +0.94
[+0.18, +1.71]; random−BALD +0.31 [−0.43, +1.03] (BALD does not beat random at d=8).

### 7.4 The particle sweep — the most important recent finding

Cursor's queue #1 ablation at **32 particles** found VOI *losing* to BALD. Re-running the
original code at n=24, d=8, 24 candidates, varying only particle count:

| Particles | random | BALD | VOI | oracle | VOI − BALD |
| --- | --- | --- | --- | --- | --- |
| 32 | 5.26 | 4.20 | 5.05 | 0.31 | **+0.85 (VOI worse)** |
| 60 | 4.79 | 4.54 | 3.38 | 1.47 | −1.16 |
| 120 | 3.65 | 3.03 | **1.49** | 0.08 | −1.55 |
| 240 | 5.01 | 4.18 | 2.92 | 0.11 | −1.26 |

**There is a threshold between 32 and 60 particles below which VOI fails.** The mechanism:
VOI computes expected regret *per particle*, so a particle set too thin to cover `w_true`
gives it a systematically wrong objective. BALD only needs the predictive distribution and
degrades gracefully. Queue #1 landed in the failure regime by accident.

Caveats: n=24 per cell; 240 is worse than 120, which is noise. The threshold is real, its
location is not pinned down.

### 7.5 Three implementation differences found in the repo

1. `n_particles` defaults to **48** in `experiment_3.py`; queue #1 overrode to 32. Both at
   or below the failure threshold.
2. The candidate pool is drawn **once per seed**, outside the user loop, and reused for all
   users and all ten queries. The reference implementation redraws per query. A fixed pool
   of 24 caps how much VOI's selectivity can buy.
3. `_candidate_queries` returns pairs of **single-state one-hot feature vectors**, not
   trajectory segments — a query is "terrain 3 or terrain 7?". With d=8 only 28 distinct
   informative queries exist, and `torch.randint` can draw i == j, producing trivial
   zero-information queries. This is a legitimate simplification, not a bug, but it means
   queue #1 and the powered run are **different experiments**.

### 7.6 Retracted — do not repeat

- **"BALD performs worse than random."** From a single seed. Across three seeds at d=4 it
  does not hold. At d=8 with 150 runs BALD is statistically indistinguishable from random —
  a weaker and different claim, and the only supportable version.
- **"The factored latent beats VPL."** Experiment 1 says no.
- **Queue #1's verdict that acquisition-side tweaks cannot close the oracle gap.** Measured
  at 32 particles, where the posterior is the bottleneck. Premature.

---

## 8. Methodology rules, and why each exists

- **Paired bootstrap, not independent standard errors.** Strategies see the same users and
  the same candidates, so runs are paired. Unpaired SEs throw that away and understate
  evidence. 10,000 resamples.
- **Report tie rate and censoring rate.** At d=8, 43% of runs tie and a third never reach
  the threshold; censored runs are counted at `n_queries + 1`, a lower bound that biases
  toward whichever method reaches more often.
- **Pre-specify the primary metric before running.** Currently: primary = final-query
  regret, secondary = queries-to-threshold. **But see §9.3 — this choice is now suspect.**
- **Minimum three seeds for exploratory claims.** Violating this produced the retraction.
- **Every result reproducible from one integer seed.**
- **Always ask what config produced a number.** Same n, d, particles, candidate
  generation, oracle definition, metric. Two confusions in this project came from
  comparing runs that differed silently.
- **Every result records the commit hash that produced it.** As of 17 September 2026 the
  repo has 30 untracked files and a last commit that retired an unrelated project, so
  *no* number currently traces to a version of the code — including the powered-run
  figures that went into the paper draft. This is the §11 config-drift failure in its
  purest form: the 2.365-vs-3.390 comparison cannot be reproduced because the
  `experiment_3.py` that generated it is not recoverable. Nothing should be re-run until
  the current state is committed, and `RESULTS.md` entries should carry a short hash from
  then on.

---

## 9. Open scientific questions

These are what an advisor should push on.

### 9.1 Is the oracle gap posterior-limited or acquisition-limited?

The oracle reaches threshold in 2.21 queries against VOI's 5.14 and ends at regret 0.249
against 2.365. Queue #1 concluded acquisition tweaks cannot close it — but at 32
particles. The particle sweep suggests the posterior is the dominant factor. **This is the
first question to settle**, and it is settled by a particle sweep at n=150, not by a new
acquisition function.

### 9.2 Does the decoupling hypothesis hold?

The theory says VOI should win more when more reward dimensions cannot affect the optimal
policy. The dimension sweep is weakly consistent (+0.42 → +1.42 → +1.00) but every CI
contains zero at n=24. A domain where decoupling is *structural* — reward dimensions the
policy provably cannot depend on — would test this properly. This is queue #2.

### 9.3 Is the primary metric aimed at the wrong regime?

Look at the regret curves: VOI leads for the first three to five queries; BALD is worst
early and catches up by query eight. Final-query regret — the pre-specified primary —
measures precisely the regime where VOI has no advantage, and the *claim of the paper* is
about low budgets.

The metric was chosen before that pattern was visible, so this is not anyone's fault. But
do not switch post hoc — that is the p-hacking the rule exists to prevent. The correct
move is to **pre-register a budget-limited metric for the next run** (regret at query 3,
or area under the curve over the first five) and state in the paper that final regret was
the original primary and showed a smaller effect.

### 9.4 Does any of it survive model mismatch?

All simulated users are generated from the same Plackett-Luce model the estimator assumes —
the most generous possible condition, and one that never holds with people. Sampling from
outside the assumed family (satisficing, lexicographic tie-breaking, fatigue) costs compute
rather than participants and no personalization paper currently does it. A reviewer will
ask.

### 9.5 The dormant threads

- **Preference/consistency factorization.** Experiment 2 shows consistency is recoverable
  in principle; experiment 1 shows the encoder cannot do it jointly from random context.
  Difficulty-stratified context sets are untested. This could become a second contribution
  or stay parked.
- **Individual non-stationarity.** Consistency drifting within a session, and the
  observation that one tired user produces statistics resembling two users — which would
  mean clustering methods like PREC may be partly clustering fatigue. Unclaimed, testable,
  and risky.

---

## 10. Decisions currently pending

1. **Matched n=150 re-run vs patch spec for Cursor.** Recommended order: re-run first. It
   is the load-bearing number, and the patch is more useful once its direction is known.
2. **Workshop paper table: regenerate or cut.** The draft's numbers predate the particle
   sweep. Regenerating is better if the re-run lands in time; cutting the comparison table
   and reporting only the oracle gap and negative results is safer if it does not.
3. **Hedge or don't hedge in the abstract.** Field norm in this literature is confident and
   unhedged — five of six comparable abstracts report no numbers at all. But the target
   workshop explicitly solicits negative results, so honesty may differentiate rather than
   weaken. Unresolved.
4. **Co-authorship.** Not decided. Not your call to make for him, but worth raising before
   submission rather than after.

---

## 11. Failure modes observed in this project

Watch for these specifically; each has already happened once.

- **Single-seed claims.** Produced the BALD-worse-than-random retraction.
- **Silent config drift between runs.** Particle count 120 → 32 and candidate generation
  changing without either being flagged; produced a false verdict.
- **Metric chosen for the wrong regime.** Final-query regret measures where the claim does
  not apply.
- **Novelty inflation between sessions.** The claim has narrowed four times; it should not
  widen back.
- **Fixing the estimator when the posterior is the bottleneck.** Queue #1.
- **Scope creep on the back of good news.** Hardware access appeared and immediately
  reopened RSS, a human study, and an IRB timeline. It then disappeared again. Treat
  capability changes as reasons to re-plan, not reasons to expand.

---

## 12. Venues and timeline

- **CoRL 2026 workshop, Human-Centered Robot Learning** — deadline stated only as "~6 weeks
  before" the Nov 9 workshop, i.e. around Sep 28, not published as a hard date. 4 pages
  excluding references, non-archival, single-blind, OpenReview. Explicitly welcomes
  in-progress work and negative results. **This is the near-term target.** Backups with
  firm dates: Oopsie (Sep 30, imperfect data), R2RL (Oct 1, sample-efficient RL).
- **IROS 2027** — March 1, 2027. Most forgiving bar; accepted before applications.
- **CoRL 2027** — not announced; the 2026 call dropped around April.
- **RSS 2027** — deadline TBA, historically late Jan/early Feb. Reconsider only with
  hardware or stronger conceptual novelty.
- **UAI** — where Regan & Boutilier published. Genuinely good methodological fit, weaker
  signal for robotics admissions.

Current recommendation: workshop now, CoRL 2027 primary, IROS March 1 as the hedge.

---

## 13. Division of labor and artifacts

- **Cursor** implements: code, experiments, plots, `RESULTS.md`.
- **You** advise: critique results, design experiments, catch errors, guard scope.
- **Repo** lives at `~/Desktop/CSCI270Project` on his machine, with `plr/` (likelihood,
  users, encoder, acquisition, mdp, hardware), `tests/`, `experiment_{1,2,3}.py`,
  `aggregate_exp3.py`, `paper/`, and CARC cluster scripts.
- **Paper draft** is `paper/workshop.tex` — 4 pages, compiles clean, title *Decision-Relevant
  Preference Elicitation under a Population Prior*. Three TODOs at the top: surname,
  co-authorship, workshop style file.

---

## 14. Experiment designs for the open questions

Each of these is specified enough to hand to Cursor. Kill conditions are stated so a
negative outcome is informative rather than demoralising.

### 14.1 Particle sweep at power (settles §9.1)

**Question.** Is the oracle gap posterior-limited or acquisition-limited?

**Design.** Particles ∈ {32, 60, 120, 240, 480}, d=8, 10 seeds × 15 users per cell (n=150),
10 queries, 24 candidates redrawn per query, fixed segment length 6. Report final regret
and regret at query 3 for random, BALD, VOI, oracle. Paired bootstrap within each cell.

**What each outcome means.** If VOI's regret keeps falling as particles rise while the
oracle's stays flat, the bottleneck is posterior coverage and the acquisition function is
fine. If both flatten together above some count, the remaining gap is genuinely
acquisition-side and queue #1's question reopens — but now asked in the right regime.

**Kill condition.** If VOI never separates from BALD at any particle count, the central
claim is in serious trouble and the decoupling experiment (§14.2) becomes the last
recourse rather than an extension.

**Cost.** The 150-run cell took roughly 25 minutes at 120 particles; 480 will be several
times that. Run overnight, background, log to file.

### 14.2 Structural decoupling domain (settles §9.2)

**Question.** Does VOI's advantage grow when reward dimensions provably cannot affect the
optimal policy?

**Design.** Construct a domain with two feature groups: *task* terrain reachable and
valuable under any policy, and *display* terrain that appears in query segments but sits in
a region the discount factor makes irrelevant to control, or that the agent cannot reach.
Sweep the ratio of display to task dimensions from 0 to 0.75 with total d fixed. Predict:
VOI − BALD grows monotonically with the ratio; at ratio 0 the two coincide.

**Why this is the strongest experiment in the queue.** It tests the *mechanism*, not just
the effect. A monotone relationship between behavioral irrelevance and VOI's advantage is a
causal story a reviewer can check, and it explains both the d=4 null and the dimension
sweep's weak trend in one figure.

**Kill condition.** No growth in the advantage as the ratio rises. That falsifies the
premise of the whole project, and it should be reported rather than buried — it would be
the most interesting negative result available.

### 14.3 Budget-limited metric, pre-registered (settles §9.3)

**Question.** Does VOI win in the regime the claim is actually about?

**Design.** Before running, commit in writing to: primary = mean regret over queries 1–3;
secondary = area under the regret curve over queries 1–5; tertiary = final-query regret
(the current primary, retained for continuity). Run at n=150, d=8, 120 particles. Report
all three regardless of which favours us.

**The discipline that matters.** State in the paper that final regret was the original
primary, that it showed a smaller effect, and that the budget-limited metric was
pre-registered afterwards on theoretical grounds. That sentence costs nothing and
inoculates against the obvious accusation.

### 14.4 Model mismatch protocol (settles §9.4)

**Question.** Does the method survive users who are not Plackett-Luce?

**Design.** Three alternative response models, each generating answers the estimator does
not assume: (a) satisficing — the user picks the first option above an internal threshold
rather than comparing; (b) lexicographic — the user sorts on their top feature and only
consults the second on ties; (c) fatigue — `b` decays geometrically across the session.
Keep the estimator unchanged. Report degradation relative to the matched-model condition
for every strategy.

**What to expect.** Everything degrades. The question is *relative* degradation: if VOI
degrades faster than BALD, that is a real weakness and belongs in the limitations section.
If it degrades comparably, that is a robustness claim worth making.

### 14.5 Difficulty-stratified context sets (revives §9.5)

**Question.** Does consistency become jointly recoverable when the context set spans
difficulty?

**Design.** Rerun experiment 1 with the encoder's context set stratified across the
difficulty quantiles from experiment 2 (roughly a third easy, a third mid, a third near
threshold) instead of uniformly random. Measure correlation between estimated and true `b`.

**Kill condition.** Correlation still near zero with oracle-stratified difficulty. That
means the amortized encoder cannot do joint estimation from ~24 answers and the
architecture needs rethinking — park the thread rather than tune it.

---

## 15. What reviewers will say, and the answers

Rehearse these. Each has a good answer; none of them should be improvised at rebuttal time.

**"This is Regan & Boutilier 2009 with a neural network."**
The strongest objection, and it must be conceded before it is made. The answer: we claim
no conceptual novelty for decision-relevant elicitation and cite it in the first paragraph.
The contribution is that their formulation requires exact minimax-regret optimization over
all of reward space with bound queries on reward values, which does not scale and is not
how humans give feedback. The population prior changes the computational problem — we
search the region users occupy, using trajectory comparisons.

**"Bıyık et al. 2024 already argued parameter entropy is the wrong target."**
True, and cited. They move the target to behavioral equivalence classes of *rewards*; we
move it to *actions under a policy*, which is a further step, and they are single-user with
no population structure.

**"Your effect is within noise."**
At the time of writing, partly true. The honest answer is the confidence interval and the
tie rate, plus the particle-sweep finding that the effect size depends strongly on
posterior quality. Do not claim more than the interval supports.

**"A 6×6 gridworld is a toy."**
Conceded, with the reason: exact optimal policies and exact regret. In any larger domain
the approximation error would be indistinguishable from the effect. State scaling as
future work rather than pretending the domain is adequate.

**"Simulated users generated from the model you fit."**
Conceded. This is why §14.4 exists; run it before submission if at all possible, because
this objection is certain to appear and having the experiment is much better than promising
it.

**"Why not just ask more questions?"**
Because the premise of the field is that you cannot — human attention is the binding
constraint. Cite Christiano's one-percent figure and Sadigh's framing.

**"Does the oracle gap mean your method barely works?"**
Reframe honestly: the oracle has information no real system can have. The gap measures
remaining headroom, not failure. But do not hide that it is large.

---

## 16. Writing norms in this literature

Useful when reviewing his drafts. Based on reading six abstracts from the direct lineage
(Sadigh 2017, Bıyık 2018, Bıyık 2019, Bıyık 2024, Hejna 2022, PEBBLE).

- **Length: 90–160 words.** Not the 200–350 that general dissertation guidance suggests.
  If a draft abstract runs 300 words it is roughly twice the field norm.
- **Five of six report no numbers at all.** Claims are qualitative — "superior
  performance", "faster reward learning". The exception, Hejna, reports one big clean
  figure (20× fewer queries). Numbers appear when they are impressive, not as evidence.
- **Open with one plain context sentence about the field**, not about the paper. "Robots
  can learn the right reward function by querying a human expert."
- **State the gap as a jab at prior work.** Bıyık 2019 uses an actual exclamation mark:
  existing methods "do not consider how easy it will be for the human to answer!"
- **Method in one or two conceptual sentences.** No equations, no acronyms beyond field
  standard.
- **End on the strongest concrete thing** — usually real robots or real users.

**The tension to flag for him.** This norm is confident and unhedged, while the project's
current result is marginal and he has been trained (correctly) to report caveats. The
resolution: caveats live in the results section, not the abstract — *except* for the
target workshop, which explicitly solicits negative results, where leading with honesty
differentiates rather than weakens. That is a judgment call he should make consciously
rather than by default.

---

## 17. How to review a draft

A checklist for when he sends a paper, abstract, or section.

1. **Is every number traceable to a run whose config you can name?** If not, find out
   before commenting on anything else.
2. **Does any claim exceed its confidence interval?** Flag hedging language that is missing
   and hedging language that is excessive — both are failures.
3. **Is Regan & Boutilier cited in the first paragraph of related work?** If it is buried,
   the paper reads as unaware.
4. **Is Bıyık et al. 2024 positioned against explicitly?** Adjacent argument, his own
   advisor, will be known to reviewers.
5. **Are the negative results present?** They are a feature at the target venue and their
   absence would be noticed by anyone who has seen the repo.
6. **Does the abstract match field length and register?** See §16.
7. **Is the oracle in every results table?** Without it the reader cannot calibrate.
8. **Does any sentence claim conceptual novelty for decision-relevant elicitation?** Cut it.
9. **Is the domain's smallness stated as a deliberate choice with its reason**, rather than
   apologised for or hidden?
10. **Would the paper survive its own §15 objections?** Read it as the hostile reviewer once
    before commenting as the advisor.

---

## 18. Glossary

- **BALD** — Bayesian Active Learning by Disagreement. Information gain about model
  parameters. Houlsby et al. 2011, which also applied it to preference learning.
- **EPIG** — Expected Predictive Information Gain. Targets predictions, not parameters.
- **VOI** — Value of Information; here, expected reduction in policy regret.
- **Regret** — expected shortfall in return from acting on an estimated reward rather than
  the true one. The quantity this project is about.
- **PbRL** — preference-based reinforcement learning. Distinguished from RLHF-on-text by
  having a policy.
- **Plackett-Luce** — ranking model; repeatedly pick the best remaining under a softmax
  over scaled rewards. Reduces to Bradley-Terry at two options.
- **Consistency (`b`)** — how reliably a user reports their own preference.
- **Oracle** — cheating strategy scoring against true regret. A ceiling, never a baseline
  to beat. Keep it in every experiment; without it you cannot distinguish a good method
  from an easy problem.
- **Particle** — one sampled candidate user `(w, b)` in the posterior representation. Count
  matters enormously (§7.4).
