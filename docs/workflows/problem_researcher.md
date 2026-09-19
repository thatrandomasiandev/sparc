# Problem Formulation — Researcher outline

**Scope:** Subsection outline + notation for Problem Formulation only. No `paper/*.tex` edits.
**Lock (exact units):** `docs/workflows/problem_user_lock.md` — five units; section title **Problem Formulation**.
**Code / claims:** `docs/workflows/problem_engineer.md`; C-form-* in `docs/CLAIMS.md`; I1 in `plr/likelihood.py`.
**Placement:** This section **precedes** Method. Existing `paper/sections/method.tex` §setup may later slim down; do not duplicate POP-VOI scoring here.
**Hygiene:** No empiric bake-off wins. No claiming Regan’s regret criterion as new.

---

## Global constraints

- **Unit order = lock order (1→5).** Do not reorder to put oracle or population first.
- **No numbers, CIs, seeds, or “VOI beats BALD.”** Pre-registration of the primary metric is allowed as a design choice; results are not.
- **Novelty boundary:** Decision-relevant / regret-based elicitation is classical (`regan2009regret`). PF states *our* objective and interface (BT segments, population prior setting, low-budget regret). Instantiation depth belongs in Method / Related.
- **Cross-refs (mandatory):**
  - **Method** presents POP-VOI acquisition (score = expected reduction in policy loss / regret under population particles).
  - **Experiments** include the true-regret **oracle in every table** (ceiling, not a competitor to beat).

---

## Notation plan

Introduce once, early in Unit 1–2; reuse everywhere.

| Symbol | Meaning |
|--------|---------|
| $\phi(s)\in\mathbb{R}^d$ | State features |
| $w\in\mathbb{R}^d$, $\|w\|=1$ | User preference direction (I1) |
| $b>0$ | Consistency / rationality |
| $r(s)=w\cdot\phi(s)$ | Linear state reward |
| $f$ | Segment summary features (see Engineer FLAG on discounted vs single-state) |
| $\delta=f_A-f_B$ | Binary query difference |
| $\sigma$ | Logistic sigmoid |
| $V^*_w$, $V^\pi_w$ | Optimal / policy value under reward $w$ (mean start-state on the grid) |
| $\mathrm{Reg}(\rho)$ | Policy regret of acting from belief/decision $\rho$ |
| $\rho$ / $\beta$ | Belief over user parameters (particle weights in code) |
| $\pi(\rho)$ | Policy induced by deciding from $\rho$ (posterior-mean $w$, then optimal policy) |
| $T$ | Query budget before acting |
| $u_1,\ldots,u_N$ | Previous users (population) |

**Avoid in PF:** acquisition score formulas, BALD entropy, particle-loss vs mean-loss — those are Method.

**Align with Engineer on $f$:** Either (a) define $f$ as experimental segment features and mention discounted counts as intended semantics, or (b) write $f=\sum_t\gamma^t\phi(s_t)$ and note Exp 3 uses single-state pairs ($T=1$). Do not leave the gap unspoken.

---

## Subsection outline (lock 1–5)

Suggested LaTeX scaffolding (Writer): `\section{Problem Formulation}` then five short `\subsection`s or unnumbered paragraphs matching units — keep tight; Method owns the algorithm.

### 1. Reward model

**Content (lock):**
- Features $\phi(s)$; reward $r(s)=w\cdot\phi(s)$ with $\|w\|=1$; consistency $b>0$.
- **One sentence why normalize:** under Bradley–Terry only $b\|w\|$ is identifiable, so pinning $\|w\|=1$ puts all scale into $b$ (C-form-1 / I1).
- Segments summarized by features $f$ (discounted counts in the ideal statement; honesty note per Engineer FLAG).

**Do not:** Derive POP-VOI; cite empiric recovery of $b$ (that is E2 / encoder).

**Claim hooks:** C-form-1; optionally point to C-form-5 later when regret needs linearity.

---

### 2. Response model

**Content (lock):**
$$
P(A\succ B)=\sigma\big(b\cdot w\cdot(f_A-f_B)\big)
$$
with $\|w\|=1$ already stated (matches code $\sigma(b\cdot\mathrm{unit}(w)\cdot\Delta f)$).

**Footnote — Plackett–Luce (mandatory if rankings are not main experiments):**
- Main tables use **binary** comparisons ($K=2$).
- Footnote: full rankings follow Plackett–Luce; BT is the $K=2$ special case (C-form-2).
- Do not promote PL to a main-equation unless ternary/ranking experiments are in-scope for the submission.

**Do not:** Expand BALD or difficulty scheduling here (I2/C-form-4 can stay Method or a one-liner later).

---

### 3. Population setting

**Content (lock):**
- Previous users $u_1,\ldots,u_N$ each provide a small number of comparisons → induce a **population prior** over $(w,b)$.
- A **new user** arrives with unknown $(w,b)$.
- The robot asks $T$ queries, then **acts** (deploys a policy on the MDP).

**Tone:** Setting only — “we will maintain a particle belief drawn from that prior” can preview Method without giving the score.

**Cross-ref:** Formal acquisition and select→update→decide → Method / Alg. POP-VOI (C-form-6).

---

### 4. Objective

**Content (lock):**
$$
\mathrm{Reg}(\rho)=V^*_w - V^{\pi(\rho)}_w.
$$
- Interpret: regret of the **executed** policy under the **true** reward $w$, when the robot decides using belief $\rho$.
- Matches `Gridworld.regret` / `POPVOI.true_regret` (Engineer).

**Metrics (pre-registration language only — no results):**
- **Primary:** low-budget regret after $t\le 3$ queries — **pre-registered** for E3-power-120 (`RESULTS.md`).
- **Secondary:** final regret (lock). Do not invent means or CIs; do not imply the powered run is done.

**Do not:** Headliner claim VOI < BALD; that is Experiments / C-main-* after RESULTS.

**Optional one sentence:** Because value of a fixed policy is linear in $w$, regret is exact for tabular linear rewards (C-form-5) — bridges to Method’s cheap batch evaluation without stealing Method’s complexity paragraph.

---

### 5. Oracle

**Content (lock):**
- An acquisition that scores candidates using **true** regret (code: `oracle_true_regret_score`).
- Role: **ceiling** on achievable low-budget / regret reduction under the same candidate pool and decision rule.
- **Never** a baseline to beat; **appears in every** elicitation table (Experiments cross-ref).

**Do not:** Describe oracle as “our method”; do not confuse with legacy `oracle_score` (posterior-loss under true answers).

---

## Cross-reference map

| PF unit | Points forward to |
|---------|-------------------|
| 1–2 formal model | Method uses same BT likelihood + linear MDP |
| 3 population | Method: fixed particles, weight updates only |
| 4 $\mathrm{Reg}$ + $t\le 3$ primary | Experiments: paired curves; pre-reg metric; no orphan numbers (I8) |
| 5 oracle ceiling | Experiments: oracle column in every table; Method: `Acquisition.ORACLE` evaluation-only |

| Do not put in PF | Home section |
|------------------|--------------|
| $\mathrm{score}(q)=\mathrm{Loss}(\beta)-\sum_y\cdots$ | Method |
| POP-VOI pseudocode | Method |
| Random / BALD / VOI bake-off | Experiments |
| Regan as “our criterion” | Related / Intro (cite; instantiation only) |

---

## Novelty / wording guardrails

- **Allowed:** “We take policy regret after a small number of segment comparisons as the elicitation objective, under a population prior over users.”
- **Forbidden:** “We propose regret-based elicitation” / “novel decision-theoretic criterion” without locating Regan (and Model-Free for non-MDP VOI) as prior art — Intro already does; PF should not re-claim the criterion.
- **Allowed:** Naming the **primary metric** as pre-registered $t\le 3$ regret.
- **Forbidden:** Any implication that E3-power-120 has settled C-main-1/2.

---

## Writer beat (suggested)

Open Unit 1 with $\phi,w,b$ and the one-sentence I1 justification → Unit 2 BT display equation + PL footnote → Unit 3 short population narrative → Unit 4 regret definition + low-budget primary / final secondary → Unit 5 oracle as ceiling with Experiments forward-ref. Keep Method’s current §setup free to shrink to “notation as in §Problem Formulation.”
