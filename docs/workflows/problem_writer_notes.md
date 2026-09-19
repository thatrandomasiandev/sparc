# Problem Formulation — Writer notes (2026-09-17)

**Deliverable:** Created `paper/sections/problem.tex`; inserted
`\input{sections/problem}` in `paper/main.tex` after Related and before Method.
Sources: `problem_user_lock.md`, `problem_engineer.md`, `problem_researcher.md`,
`introduction.tex`.

---

## Structure shipped

| Unit | Subsection | Lock content |
|------|------------|--------------|
| 1 | Reward model | $\phi$, $r=w\cdot\phi$, $\|w\|=1$, $b>0$; I1 scale sentence; $f=\sum_t\gamma^t\phi(s_t)$ + Exp single-state honesty |
| 2 | Response model | BT display eq; PL in footnote ($K{=}2$ main) |
| 3 | Population setting | $u_1..u_N$ → prior; new user; $T$ queries then act; Method forward-ref |
| 4 | Objective | $\mathrm{Reg}(\rho)=V^*_w-V^{\pi(\rho)}_w$; $t\le 3$ primary pre-reg; final secondary; light Regan cite |
| 5 | Oracle | True-regret ceiling; every table; Experiments forward-ref |

Order matches the lock (1→5). Section label: `sec:problem`.

---

## Engineer FLAG — segments

Chose researcher option **(b):** formal $f=\sum_t\gamma^t\phi(s_t)$, then one honest
sentence that tabular elicitation experiments use single-state segments
($T{=}1$), so $f=\phi(s)$ there.
Did **not** claim Exp~3 / E3-power queries are multi-step discounted occupancies.

---

## Cite keys

- **Objective:** `regan2009regret` (lineage only; “we do not claim the criterion as new”).
- No other new cites; Intro / Related already carry Model-Free and PbRL neighbors.

---

## Intro / Method sync

- Notation aligns with Intro (BT, linear rewards, $\|w\|=1$, population prior, regret target).
- Method `\subsection{Problem Setup}` still restates a short version; optional later slim-down to “notation as in §Problem Formulation” — not done here.
- Acquisition score / POP-VOI / BALD bake-off left to Method / Experiments.

---

## Hygiene checklist (passed)

- [x] Five lock units in order; no empiric means/CIs/bake-off wins
- [x] I1 justification one sentence; BT with $\|w\|=1$
- [x] PL footnote only; main eq is binary BT
- [x] Primary $t\le 3$ pre-registered; final regret secondary (lock, not RESULTS tertiary detail)
- [x] Oracle = ceiling, every table; not a competitor
- [x] Segment FLAG disclosed
- [x] No novelty claim vs Regan

---

## Optional follow-ups (not done)

- Slim Method §setup once Method writer pass lands.
- If Experiments section label differs from `sec:experiments`, fix the `\cref`.
