# Problem Formulation — user lock (2026-09-17)

Five units. Section title: **Problem Formulation**.
No empiric bake-off wins. Formal claims must match `plr/` + C-form-* / I1.

## 1. Reward model
- Features $\phi(s)$, reward $r(s)=w\cdot\phi(s)$ with $\|w\|=1$, consistency $b>0$.
- One sentence why: under Bradley–Terry only $b\|w\|$ is identifiable, so normalizing
  puts all scale into $b$.
- Segments summarized by discounted feature counts $f=\sum_t \gamma^t \phi(s_t)$.

## 2. Response model
- $P(A\succ B)=\sigma\big(b\cdot w\cdot(f_A-f_B)\big)$.
- Plackett–Luce in a **footnote** if rankings are not used in the main experiments.

## 3. Population setting
- Previous users $u_1,\ldots,u_N$ each provided a small number of comparisons.
- A new user arrives with unknown $(w,b)$.
- The robot asks $T$ queries, then acts.

## 4. Objective
- Regret of the executed policy under the true reward:
  $\mathrm{Reg}(\rho)=V^*_w - V^{\pi(\rho)}_w$.
- **Low-budget objective** is regret after $t\le 3$ queries, **pre-registered**;
  final regret is **secondary**.

## 5. Oracle
- An acquisition that scores candidates with **true** regret.
- It is a **ceiling**, never a baseline to beat, and appears in **every** table.
