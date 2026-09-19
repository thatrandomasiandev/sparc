# Problem Formulation — Tester gate

**Role:** Tester  
**Draft:** `paper/sections/problem.tex` (**not edited**)  
**Authority:** `docs/workflows/problem_user_lock.md`  
**Cross-check:** `problem_engineer.md`, `problem_researcher.md`, `problem_writer_notes.md`,  
`plr/likelihood.py`, `plr/mdp.py` (regret / `oracle_true_regret_score`), `paper/main.tex`,  
`RESULTS.md` § E3-power-120 primary note  
**Date:** 2026-09-17 (retest after $H$ / $T$ fix)

---

## Section decision

**OK_FOR_PAPER**

---

## Must-fixes

*(none)*

---

## Retest note

Prior must-fix (overloaded $T$ for segment horizon and query budget) is resolved:
segment horizon is $H{=}1$; query budget remains $T$. Gates 1–8 remain PASS; no empirics added.

---

## Gate checklist

| # | Gate | Result | Evidence |
|---|------|--------|----------|
| 1 | I1 / BT with $\|w\|=1$ | **PASS** | Scale sentence: only $b\|w\|$ identifiable → pin $\|w\|=1$. Display: $P(A\succ B)=\sigma(b\,w\cdot(f_A-f_B))$ with $\|w\|=1$. Matches `binary_logp` / `unit(w)`. |
| 2 | Segment $f$ honesty if discounted form used | **PASS** | Formal $f=\sum_t\gamma^t\phi(s_t)$; single-state disclosure with segment horizon $H{=}1$; not multi-step discounted occupancy. $T$ reserved for query budget. |
| 3 | Regret definition matches code | **PASS** | $\mathrm{Reg}(\rho)=V^*_w-V^{\pi(\rho)}_w$ ≡ `Gridworld.regret` / `POPVOI.true_regret`. |
| 4 | $t\le 3$ primary / final secondary; pre-reg only; no fake results | **PASS** | Primary pre-registered; final-query secondary; no empiric comparisons in section. |
| 5 | Oracle = ceiling; every table; not baseline to beat | **PASS** | True-regret oracle; ceiling; never baseline to beat; every table; `\cref{sec:experiments}`. |
| 6 | PL footnote OK | **PASS** | PL full rankings; BT = $K{=}2$; main experiments binary. |
| 7 | Lock order 1–5 present | **PASS** | Reward → Response → Population → Objective → Oracle. |
| 8 | `main.tex` wires problem between related and method | **PASS** | related → problem → method. |

---

## Exact lock compliance (units)

| Unit | Lock requirement | Draft | Verdict |
|------|------------------|-------|---------|
| 1 | $\phi$, $r=w\cdot\phi$, $\|w\|=1$, $b>0$; I1 why; $f$ discounted | Present + I1 + $H{=}1$ honesty | **PASS** |
| 2 | BT display; PL footnote if rankings not main | Display eq + PL footnote | **PASS** |
| 3 | Prior users → prior; new user; $T$ queries then act | Present; Method forward-ref | **PASS** |
| 4 | $\mathrm{Reg}$; $t\le 3$ primary pre-reg; final secondary | Present; Regan lineage only | **PASS** |
| 5 | True-regret acquisition; ceiling; every table | Present | **PASS** |

---

## Optional nits (not must-fixes)

- Paper regret is abstract $V^*-V^\pi$; code averages over non-goal/non-wall starts — fine for PF.
- Oracle sentence is lock-level; Method can sharpen expected-drop wording.
- `\textbf{primary}` / `\textbf{secondary}` optional tone-down.
- Method §setup still duplicates short setup — out of PF scope.
