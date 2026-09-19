# Conference paper outline: Decision-Relevant Preference Elicitation under a Population Prior

**Targets:** IROS 2027 (Mar 1, 6 pages + refs), used as the hedge; CoRL 2027 (≈8 pages + refs), the primary target.
**Status as of 2026-09-17:** outline only. Every result below is marked **[have]**, **[pending]** or **[new work]**.
Numbers from the unreproduced 120-particle table (VOI 2.365 / BALD 3.390) appear nowhere in this outline and must not appear in the paper.

---

## 0. The paper in five sentences

1. Robots that learn rewards from comparisons usually pick queries by information about reward parameters.
2. Much of that information never changes what the robot does, because many rewards induce policies with near-identical regret.
3. We select queries by expected reduction in policy regret, as Regan & Boutilier (2009) proposed, and make this affordable with a particle prior *learned from previous users' comparisons*.
4. Across two domains with pre-registered metrics, we measure when this beats information-gain selection at low query budgets and when it does not.
5. We trace the difference to two mechanisms: how much reward uncertainty is decision-irrelevant, and whether the prior covers the new user.

Sentence 4 is written in the neutral form on purpose. It becomes "beats" or "matches" only after §6.2 lands.

---

## Claims → evidence map

Nothing enters the paper unless it has a row here.

| ID | Claim | Section | Evidence | Status |
|---|---|---|---|---|
| C-mot | A substantial fraction of posterior uncertainty is regret-irrelevant in our domains | 6.1 | ε-regret equivalence rate | **[new work]**. Exact-match E-eq exists and argues *against* the claim (local stability 4.7% at d=8) |
| C-main | VOI < BALD on low-budget regret | 6.2 | E3-power-120 (grid), segment + domain-2 replications | **[pending]**. 48p pre-registered run is **null** (+0.13 [−0.59, +0.83]) |
| C-prior-1 | A learned prior is close to the true population prior | 6.3 | learned vs true vs uniform | **[new work]** |
| C-prior-2 | The prior is what makes VOI work at a feasible particle count | 6.3 | uniform-prior VOI vs population-prior VOI at matched particles | **[new work]** |
| C-prior-3 | Prior shift shrinks the advantage, and we can say by how much | 6.3 | shifted-mode test users | **[new work]** |
| C-mech | The VOI − BALD gap grows with the decision-irrelevant fraction ρ | 6.4 | decoupling sweep | exploratory run in progress; **[new work]** at n=150 |
| C-cov | VOI needs sufficient posterior coverage; BALD degrades gracefully | 6.5 | particle sweep | **[have]** at n=24 (exploratory); **[new work]** at n=150 |
| C-rob | Relative degradation under non-BT users is comparable across methods | 6.6 | mismatch suite | **[new work]** |

---

## Title

- Branch A (C-main holds): *Decision-Relevant Preference Elicitation under a Learned Population Prior*
- Branch B (C-main null or mixed): *When Does Decision-Relevant Preference Elicitation Help? Regret-Targeted Queries under a Population Prior*

## Abstract (~150 words, written last)

1. Field context sentence: robots learn rewards by asking people to compare behaviors, and each question costs human attention.
2. The jab: information-gain methods spend that attention separating rewards the robot would act on identically.
3. Method, in one sentence and with no equations: pick queries by expected regret reduction under a particle prior learned from previous users.
4. Evaluation: two domains, pre-registered low-budget metric, simulated users, including users who violate the model.
5. Finding: branch A or B phrasing (see §0).
6. End on the strongest concrete thing, whichever of the mechanism figure or the prior result is cleanest.

No numbers unless one is large and clean.

---

## 1. Introduction (1.0 page; IROS 0.8)

**Figure 1 (top of page 2 or right column of page 1): the concept.**
Two rewards w₁ and w₂ that differ in parameter space but produce the same path in a small grid. Panel (a): BALD's chosen query separates w₁ from w₂, so it is informative about parameters but has zero regret value. Panel (b): VOI's chosen query separates the policy-relevant alternative. This must be a *real* example taken from the gridworld, not a cartoon, and ideally one pulled from the §6.1 data.

**¶1 Context.** Preference-based reward learning; human attention is the budget. Cite Christiano 2017 (feedback on under 1% of interactions) and Sadigh 2017.

**¶2 How queries are chosen today.** The lineage runs from volume removal through information gain to the generalized alignment objective (Sadigh 2017 → Bıyık 2019 → Bıyık 2024). All of them target knowledge about the reward.

**¶3 The gap.** The robot acts through a policy. Bıyık et al. 2024 already observed that many rewards produce the same behavior and moved the target to behavioral equivalence classes. We move it one step further, to the regret of the policy the robot will actually execute. Regan & Boutilier 2009 formulated exactly this target (minimax regret, bound queries, tabular MDPs), and it has not been used with comparison queries because it requires searching all of reward space.

**¶4 Our idea.** Robots rarely meet their first user. Previous users' comparisons define a prior that concentrates on the region real users occupy, which turns regret-based selection from intractable into a particle computation. The linear-value trick makes each candidate cost two matrix products.

**¶5 What we find.** Branch-dependent; one paragraph. Name the mechanism result either way.

**¶6 Contributions (bulleted, 4 items):**
1. A population prior learned from previous users' comparisons, and evidence of what it buys (§4.1, §6.3).
2. Regret-based VOI for trajectory-segment comparisons, with an efficient evaluation (§4.2–4.3).
3. A pre-registered low-budget evaluation against random, BALD and an oracle ceiling in two domains (§6.2).
4. Mechanism analyses: when regret-irrelevant uncertainty exists (§6.1, §6.4), and how much prior coverage VOI needs (§6.5).

**Do not write:** "we are the first," "novel paradigm," or any sentence implying conceptual novelty for decision-relevant elicitation.

---

## 2. Related work (0.75 page; IROS 0.5)

Four paragraphs, each ending with how this work differs.

**2.1 Decision-relevant elicitation.** This paragraph comes first. Regan & Boutilier 2009 (minimax-regret reward elicitation in MDPs); Alizadeh et al. (approximate regret-based elicitation); Model-Free Preference Elicitation (IJCAI 2024, EVOI for recommendations).
*Difference:* comparison queries rather than bound queries, a learned population prior rather than all of reward space, and a robotics setting with trajectories.

**2.2 Active preference-based reward learning.** Sadigh 2017, Bıyık & Sadigh 2018, Bıyık 2019 (information gain yields easy questions), Hejna & Sadigh 2022 (ensembles), 2026 query synthesis, SPARQ 2026.
Bıyık et al. 2024 gets its own positioning sentence, since it is the closest argument: their target is reward equivalence classes for a single user, while ours is the regret of the executed policy under a population prior.
EPIG (AISTATS 2023) supplies the general active-learning version of "parameter information is the wrong target."

**2.3 Population and pluralistic preference models.** MaxMin-RLHF (Thm 1 motivates why an average user is nobody), DPL, PAL, VPL (closest architecture; fixed survey queries; names active selection as future work), PREC 2026.
*Difference:* we use the population to *choose queries*, not only to model users.

**2.4 Active personalization elsewhere.** AMPLe (ACL 2025) and active utility-based pairwise sampling (2025).
*Difference:* neither has a downstream policy or a regret target.

---

## 3. Problem formulation (0.75 page; IROS 0.6)

**3.1 Reward model.** Features φ(s), reward r(s) = w·φ(s) with ‖w‖ = 1, and consistency b > 0. One sentence on why: under Bradley-Terry only b‖w‖ is identifiable, so normalizing puts all scale into b. Segments are summarized by discounted feature counts f = Σₜ γᵗ φ(sₜ).

**3.2 Response model.** P(A ≻ B) = σ(b · w·(f_A − f_B)). Plackett-Luce goes in a footnote if rankings are not used.

**3.3 Population setting.** Previous users u₁…u_N each provided a small number of comparisons. A new user arrives with unknown (w, b). The robot asks T queries, then acts.

**3.4 Objective.** Regret of the executed policy under the true reward: Reg(ρ) = V*_{w} − V^{π(ρ)}_{w}. The low-budget objective is regret after t ≤ 3 queries, pre-registered; final regret is secondary.

**3.5 Oracle.** An acquisition that scores candidates with true regret. It is a ceiling, never a baseline to beat, and appears in every table.

---

## 4. Method (1.0 page; IROS 0.8)

**4.1 Learning the population prior. [new work]**
Current code samples particles from the *true* generating mixture (`experiment_3.py:114-119`), so the prior is an oracle. The conference version replaces this with:

- N_train previous users (sweep N_train ∈ {10, 50, 200}), each answering K random comparisons (K ≈ 20, or match what VPL/PAL assume).
- Per-user MAP estimate of w on the sphere (normalized), with b fixed to the population median or jointly estimated. Experiment 2 says joint b is hard, so fix it and say so.
- Fit a mixture over the per-user estimates (von Mises–Fisher, or a Gaussian followed by renormalization; choose the number of components by held-out likelihood).
- Sample P particles from the fitted mixture.

Algorithm box 1 covers the prior-learning procedure. The key design choice to justify is using per-user point estimates rather than a hierarchical posterior: it is simple and cheap, and hierarchical inference is future work.

**4.2 Posterior and VOI acquisition.**
- Particle posterior: weights updated by the Bradley-Terry likelihood; resample when ESS falls below a threshold (state it).
- Loss L(ρ) = E_{w∼ρ}[V*_w − V^{π(ρ)}_w], and score(q) = L(ρ) − Σ_y P(y) L(ρ | y).
- Efficient evaluation: for a fixed policy, V^π(w) = (I − γP^π)⁻¹ Φ w. Invert once per candidate policy, then evaluate every particle with one matrix product. Give the per-query cost in O-notation.

Algorithm box 2 is the elicitation loop.

**4.3 Segment queries. [new work]**
Candidates are pairs of rollout segments of length L (current exploratory value 6) from a diverse behavior set: random, and optimal policies for sampled particles. Candidates are redrawn every query; state the pool size. This replaces the single-state one-hot pairs the workshop version used, and the paper should say plainly that the workshop version used single states.

**4.4 Acting.** Execute π(posterior mean). One sentence noting that acting with the minimax-regret policy over particles is a natural alternative, left to future work unless C-mech-4 gets run.

**4.5 Baselines.** Random; BALD (with the per-particle entropy subtraction, so the trivial query scores 0); EPIG (optional, one line, but include it if cheap because reviewers from the active-learning side will ask); oracle.

---

## 5. Experimental setup (0.75 page; IROS 0.6)

**5.1 Domains.**
- **D1: 6×6 gridworld.** One-hot terrain features, d = 8, γ = 0.95. Optimal policy and regret are exact. This choice is deliberate: approximation error cannot be confused with the effect under test.
- **D2: a larger domain. [new work]** Recommended option: a trajectory-library domain in the style of the lab's driving and reaching tasks, where the "policy" is the best trajectory in a large fixed library. Regret is then exact *over the library*, which keeps the D1 advantage while adding continuous dynamics and trajectory-level features. Check whether APReL (the lab's library) already provides this; if it does, D2 costs about a week instead of a month.
- If D2 slips, drop "continuous control" from every claim. Do not substitute a domain with approximate regret late in the schedule.

**5.2 Simulated users.** Population: a mixture of preference modes, with consistency b drawn from a log-normal (median 3, spread 0.3). Test users are disjoint from prior-training users. Model-mismatch users are covered in §6.6.

**5.3 Protocol.** T = 10 queries, 10 seeds × 15 users = 150 paired runs per cell. The same users and the same candidate draws are shared across strategies within a seed.

**5.4 Metrics, pre-registered.**
- Primary: mean regret over queries 1–3.
- Secondary: AUC of regret over queries 1–5.
- Tertiary: final-query regret, retained because it was the original primary. The paper says this in one sentence.
- Also reported: queries to threshold (with censoring rate), tie rate, reach rate.

**5.5 Statistics.** Paired bootstrap with 10,000 resamples and 95% CIs on paired differences. No independent-SE comparisons. Every number in the paper links to a commit hash in Appendix C.

---

## 6. Results (2.5 pages; IROS 1.9)

For every subsection: the question, the pre-registered prediction, the result, and one sentence on what it means. Report nulls in the same format as positives.

### 6.1 Is posterior uncertainty regret-irrelevant in these domains? [new work]
- **Measure:** for pairs of rewards drawn from the population prior, the cross-regret V*_{w₁} − V^{π(w₂)}_{w₁}. Report the fraction below ε for ε at 1%, 5% and 10% of the typical value range, as a function of d.
- **Why this replaces the exact-match E-eq:** exact policy identity over all 36 states is stricter than the argument needs. Two policies can differ in unimportant states at negligible regret. Report exact match in the appendix, including that it is low.
- **Prediction (pre-register):** the ε-equivalence fraction is substantial and grows with d.
- **If it fails:** the motivation is weak *in these domains*. §6.4, where decision-irrelevance is built in, then carries the mechanism argument, and the introduction must be written accordingly.
- Small figure or inline numbers.

### 6.2 Main comparison at low budget
- **Table 1:** random, BALD, (EPIG), VOI and oracle; rows D1-single-state, D1-segments and D2; columns primary, secondary, tertiary and reach rate. Paired differences with CIs, in the table or directly below it.
- **Figure 2:** regret vs query index for D1 and D2, with 95% bands. The low-budget window is shaded.
- **History to report honestly, in one or two sentences:** the first pre-registered powered run (48 particles, final regret primary) found no VOI − BALD difference (+0.13 [−0.59, +0.83]). The particle sweep (§6.5) motivated 120 particles, and the budget-limited primary was pre-registered before the re-run.
- **Evidence status:** E3-power 48p **[have, null]**; E3-power-120 **[pending]**; segments and D2 **[new work]**.

### 6.3 What does the population prior buy? [new work] (most important section)
Four prior conditions, all with VOI and BALD, at matched particle count on D1 (and D2 if time allows):
1. **True prior** (the current code; an upper reference).
2. **Learned prior**, with N_train ∈ {10, 50, 200}.
3. **Uniform prior** on the sphere at the same particle count.
4. **Uniform prior at 10× particles**, which answers "is the prior just a compute saving?"

Plus **prior shift:** test users drawn from a population with one mode moved or removed.

- **Figure 3:** primary metric vs N_train for learned-prior VOI and BALD, with horizontal lines for the true prior and the uniform prior.
- **Predictions:** learned approaches true as N_train grows; uniform is much worse at matched particles; shift shrinks VOI's advantage more than BALD's, since VOI trusts the prior more. Pre-register all of these.
- **If uniform ≈ population prior:** contribution #1 is weakened and the paper says so. That is still publishable as part of branch B.

### 6.4 Mechanism: decision-irrelevant dimensions
- A decoy-feature construction: some features appear in query segments but are unreachable or discount-irrelevant, so they provably cannot change the optimal policy. Sweep ρ ∈ {0, 0.25, 0.5, 0.75} with total d fixed.
- **Figure 4:** VOI − BALD gap vs ρ with CIs, at n = 150 per ρ.
- **Prediction:** a monotone increase, with a gap near 0 at ρ = 0. **Kill:** flat or decreasing. That falsifies the premise and gets reported as such.
- The current exploratory run (3 seeds × 8 users) is **not** the paper result. Its figure was also generated from a partial run and must be regenerated.

### 6.5 How much prior coverage does VOI need?
- Particles ∈ {32, 60, 120, 240, 480} at n = 150, on D1 with segments.
- Report VOI − BALD by particle count, plus the oracle's regret, to separate posterior-limited from acquisition-limited.
- The exploratory n = 24 sweep suggests a threshold between 32 and 60 particles; the paper result is the n = 150 version.
- For IROS, this moves to the appendix, with one sentence in the main text.

### 6.6 Robustness to users who are not Bradley-Terry
- Satisficing (picks the first option above an internal threshold), lexicographic (top feature, ties broken by the second), and fatigue (b decays geometrically within the session). The estimator is unchanged.
- **Table 2:** degradation of the primary metric relative to matched-model users, per strategy.
- **The claim is about relative degradation.** If VOI degrades more than BALD, that goes in limitations, stated plainly.

---

## 7. Discussion and limitations (0.4 page)
- **The oracle gap** is headroom, not failure, but give its size.
- **Simulated users throughout.** §6.6 is the partial answer; a human study with the lab's robots is future work.
- **Domain scale.** Exact or library-exact regret was a deliberate choice; scaling to learned policies means VOI becomes approximate, which is an open question.
- **The prior assumes population structure.** A truly novel user falls outside the prior, and §6.3's shift result quantifies the cost.
- **Consistency is fixed to the population median** in prior learning. Joint estimation failed in our own experiments (Appendix A).

## 8. Conclusion (0.15 page)
Three sentences: the target, what the prior enables, and the conditions under which it pays off.

---

## Appendix (not page-limited at CoRL; check the IROS rules)
- **A. Consistency identification.** Experiment 1 (the factored latent did not beat the single latent) and Experiment 2 (b is recoverable when w is known; easy queries are catastrophic; median error 23 vs 1.07 on hard queries). This is honest negative work that shows the normalization is load-bearing.
- **B. Exact-match policy equivalence** (the original E-eq), including that it is low.
- **C. Pre-registration log.** Every experiment with its registration date, commit hash, and primary metric. Include the 48-particle null and the reason for changing the primary metric.
- **D. Full tables** with tie rates, censoring, and every strategy × condition.
- **E. Hyperparameters and compute.** Particle counts, candidate pool size, segment length, ESS threshold, and runtime per cell.
- **F. The n = 150 particle sweep** (if moved from §6.5).

---

## Figure and table budget

| # | Content | Section | Must-have for IROS? |
|---|---|---|---|
| Fig 1 | Concept: BALD query vs VOI query on a real grid example | 1 | yes |
| Fig 2 | Regret curves, D1 + D2 | 6.2 | yes |
| Fig 3 | Prior ablation vs N_train | 6.3 | yes |
| Fig 4 | Gap vs ρ | 6.4 | yes |
| Fig 5 | Particle sweep | 6.5 | appendix |
| Tab 1 | Main comparison, both domains, with oracle | 6.2 | yes |
| Tab 2 | Mismatch degradation | 6.6 | compress to text if needed |
| Alg 1 | Prior learning | 4.1 | yes |
| Alg 2 | Elicitation loop | 4.2 | merge with Alg 1 if short of space |

---

## Reviewer objections and where the paper answers them

| Objection | Answered in |
|---|---|
| "This is Regan & Boutilier with particles" | ¶3 of intro + §2.1 first sentence; the contribution is the learned prior (§6.3) and comparison queries |
| "Bıyık 2024 already made this argument" | §2.2 positioning sentence; the target differs (executed-policy regret vs reward classes) |
| "Your prior is the true distribution" | §4.1 + §6.3: learned prior, N_train sweep, prior shift |
| "Effect is within noise" | Pre-registration, paired CIs, the 48p null reported openly, §6.5 coverage explanation |
| "Toy domain" | D2 + the stated reason for exact regret |
| "Users are generated from your model" | §6.6 |
| "Is the prior just a compute trick?" | §6.3 condition 4 (uniform at 10× particles) |
| "Oracle gap means it barely works" | §7 first bullet |

---

## Build schedule to Mar 1

| Dates | Work | Output |
|---|---|---|
| Sep 18–28 | Workshop paper: E3-power-120, ε-regret measure (D1), finished exploratory decoupling | 4-page submission |
| Sep 29 – Oct 20 | §4.1 learned prior + §6.3 on D1 (single-state queries are fine for this) | Fig 3 draft |
| Oct 21 – Nov 10 | §4.3 segment queries; re-run §6.2 and §6.1 on D1 with segments | Table 1 rows 1–2 |
| Nov 11 – Dec 15 | D2 (check APReL first); §6.2 and §6.3 on D2. **Application season: highest slip risk** | Table 1 row 3 |
| Dec 16 – Jan 5 | §6.4 at n = 150, §6.6, §6.5 at n = 150 (overnight runs, low attention) | Figs 4–5, Tab 2 |
| Jan 6 – 25 | Full draft | Draft to Bıyık / lab ~Jan 25 |
| Jan 26 – Feb 22 | Revisions, figure polish, freeze Feb 22 | Frozen PDF |
| Mar 1 | IROS submission | |

**What to cut if time runs short, in order:** (1) §6.5 at n = 150, keeping the n = 24 version as exploratory in the appendix; (2) §6.6 reduced to fatigue only; (3) D2, and with it the "continuous control" wording. **Never cut §6.3.**

---

## Decisions to make now
1. **Does D2 survive the application season?** Decide by Oct 15, based on whether APReL gives you a library domain for free.
2. **Is EPIG in or out?** Include it if it is under a day of work.
3. **Co-authorship and the lab's involvement.** Needs deciding before the January draft goes to Bıyık, not after.
4. **Workshop → conference continuity.** The workshop version is non-archival, so reusing its content is fine. Say so in the conference submission if the venue asks.
