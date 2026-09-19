# Problem Formulation — Engineer notes

**Scope:** Equation ↔ code map for the locked Problem Formulation units only. Do **not** edit `paper/*.tex`.
**Lock (authority):** `docs/workflows/problem_user_lock.md` — five units; no empiric bake-off wins.
**Formal claims:** `docs/CLAIMS.md` C-form-*; invariant **I1** in `plr/likelihood.py`.
**No invented results.** E3-power-120 primary is pre-registered; verdict remains pending.

---

## Global code anchors

| Concern | Source of truth |
|---------|-----------------|
| I1 / BT / PL | `plr/likelihood.py` (`unit`, `rewards`, `binary_logp`, `plackett_luce_logp`) |
| Regret / VI / values | `plr/mdp.py` (`Gridworld.regret`, `policy_value`, `value_iteration`) |
| Oracle ceiling | `plr/mdp.py::oracle_true_regret_score`; wired as `Acquisition.ORACLE` in `plr/algorithm.py` |
| Exp candidate features | `experiment_3.py::_candidate_queries` |
| Pre-registration | `RESULTS.md` § E3-power-120 |

---

## Unit 1 — Reward model

### Equations (lock)

- State features $\phi(s)\in\mathbb{R}^d$.
- Linear reward $r(s)=w\cdot\phi(s)$ with $\|w\|=1$, consistency $b>0$.
- Segment summary $f=\sum_t\gamma^t\phi(s_t)$ (lock / `PROJECT.md` §3.1 intended semantics).

### Why $\|w\|=1$ (I1) — confirmed in code

Under Bradley–Terry / Plackett–Luce, answer probabilities depend on the product $b\|w\|$ only.
Doubling $w$ and halving $b$ leaves every prediction unchanged.
**Only $b\|w\|$ is identifiable → pin $\|w\|=1$ and put all scale into $b$.**

| Claim | Evidence |
|-------|----------|
| C-form-1 | `test_unit_norm_invariant_i1` |
| Docstring I1 | `plr/likelihood.py` module docstring |
| Enforcement | `unit(w)` inside `rewards`, `binary_logp`, `reward_vector`, `policy_value` |

Callers must never compute `w @ phi` raw — always `rewards` / `unit`.

### Equation ↔ code

| Symbol / eq | Code |
|-------------|------|
| $\phi(s)$ | `Gridworld.phi` shape `(S, d)` (`plr/mdp.py`) |
| $r(s)=w\cdot\phi(s)$ with unit $w$ | `likelihood.rewards(phi, w)` → `unit(w)·phi`; MDP: `reward_vector` → `phi @ unit(w)` |
| $\|w\|=1$ | `likelihood.unit(w)` |
| $b>0$ | `_as_positive_b`; raises if $b\le 0$ |

### FLAG — segment features (Writer must be honest)

**Lock / PROJECT.md** define
$f=\sum_t\gamma^t\phi(s_t)$ (discounted feature counts / occupancy-style summary).

**Main elicitation experiments do not build that $f$ today.**

```40:43:experiment_3.py
def _candidate_queries(env, n: int, K: int, g: torch.Generator) -> Tensor:
    S, _d = env.phi.shape
    idx = torch.randint(0, S, (n, K), generator=g)
    return env.phi[idx]
```

- Candidates are **pairs of single-state feature rows** `env.phi[s]` — a length-1 “segment,” not a discounted trajectory sum.
- `RESULTS.md` E3-power-120 protocol states the same: “single-state pairs.”
- Hardware path *can* use discounted sums (`plr/hardware/segments.py::featurize_states(..., feature_fn="discounted_sum")`), but that is not what Exp 3 / E3-power feed into BT.

**Writer options (pick one; do not silently equate them):**

1. Define $f$ as the **segment summary features used in experiments** (currently single-state $\phi(s)$), and note discounted counts as the intended / robotics semantics; or
2. Keep discounted $f=\sum_t\gamma^t\phi(s_t)$ as the formal definition and **explicitly note** that reported gridworld elicitation runs use single-state pairs as a special case ($T=1$).

Do **not** claim Exp 3 queries are multi-step discounted occupancies unless the candidate generator changes.

---

## Unit 2 — Response model

### Equations (lock)

$$
P(A\succ B)=\sigma\big(b\cdot w\cdot(f_A-f_B)\big)
\quad\text{with }\|w\|=1.
$$

Plackett–Luce for full rankings → **footnote** if main experiments stay binary (they do: `K=2` in `_candidate_queries`).

### Code (I1-aware)

```49:61:plr/likelihood.py
def binary_logp(delta: Tensor, w: Tensor, b: Tensor) -> Tensor:
    ...
    w_u = unit(w)
    gap = (delta * w_u).sum(dim=-1)  # (...)
    return logsigmoid(b * gap)
```

With `delta = f_A - f_B` from `binary_logp_from_options`:
$\log P=\mathrm{logsigmoid}(b\cdot\mathrm{unit}(w)\cdot(f_A-f_B))$.

| Paper-safe write | Code truth |
|------------------|------------|
| $\sigma(b\,w\cdot(f_A-f_B))$ with $\|w\|=1$ stated once | Equivalent when $w$ is already unit |
| — | Implementation always applies `unit(w)` so forgotten normalization cannot double-count scale into $b$ |

### PL

| Item | Code / claim |
|------|----------------|
| Full ranking | `plackett_luce_logp(phi, w, b, order)` |
| BT ≡ PL at $K=2$ | C-form-2 / `test_bradley_terry_equals_plackett_luce_at_k2` |
| Online update | `POPVOI.update` uses PL even for binary (`order` length 2) |

**Engineer note for Writer:** Prefer the paper BT formula with $\|w\|=1$; optional parenthetical that code normalizes via `unit(w)`. PL stays in a footnote unless ternary/ranking experiments enter the main tables.

---

## Unit 3 — Population setting

### Lock narrative

Previous users $u_1,\ldots,u_N$ each give a small number of comparisons → population prior.
New user arrives with unknown $(w,b)$.
Robot asks $T$ queries, then acts.

### Equation ↔ code

| Concept | Code |
|---------|------|
| Population particles $\{(w_p,b_p)\}$ | `POPVOI.particles_w` `(P,d)`, `particles_b` `(P,)`; built by `build_population_prior` / `sample_gaussian_mixture_population` |
| Belief $\beta$ | `Belief` / `POPVOI.belief` — categorical weights over **fixed** particles |
| Init | Uniform $1/P$ (`POPVOI.reset`) |
| New-user session | `POPVOI.run` / `step`: select → answer → `update` |
| Act after $T$ | `decide_w` → `posterior_mean_decision`; `decide_policy` → VI for that $w$ |

Acquisition / POP-VOI scoring lives in **Method**, not Problem Formulation — only the setting (prior users → prior; new user → $T$ queries → act) belongs here.

---

## Unit 4 — Objective

### Equations (lock)

$$
\mathrm{Reg}(\rho)=V^*_w - V^{\pi(\rho)}_w.
$$

- $\rho$: belief / posterior used to choose a decision weight (then a policy).
- $\pi(\rho)$: optimal policy for the decision induced by $\rho$ (code: mean of particles under weights, then unit-normalized).
- **Primary (pre-registered, E3-power-120):** low-budget regret after $t\le 3$ queries.
- **Secondary (lock):** final regret. (`RESULTS.md` also lists AUC over queries 1–5 as secondary and final-query regret as tertiary — Problem Formulation should follow the **lock**: $t\le 3$ primary, final regret secondary. Do not invent numbers.)

### Regret matches `env.regret` — confirmed

```139:154:plr/mdp.py
def regret(
    self,
    w_true: Tensor,
    w_decision: Tensor,
    V_star_true: float | Tensor | None = None,
) -> Tensor:
    """Policy regret of acting optimally for ``w_decision`` under true ``w_true``."""
    _, pi_hat = self.value_iteration(w_decision)
    ...
    v_pi = self.policy_value(pi_hat, w_true)
    return v_star - v_pi
```

| Symbol | Code |
|--------|------|
| $V^*_w$ | VI under `w_true`; mean over start mask (non-goal, non-wall) |
| $\pi(\rho)$ | `pi_hat = value_iteration(w_decision)[1]` with `w_decision = decide_w()` = `unit(E_\beta[w])` |
| $V^{\pi}_w$ | `policy_value(pi_hat, w_true)` |
| $\mathrm{Reg}$ | `v_star - v_pi` ≡ `Gridworld.regret` ≡ `POPVOI.true_regret` |

Linearity used elsewhere (C-form-5): $V^\pi(w)=M^\pi\,\mathrm{unit}(w)$ via `policy_value_matrix` — Method / complexity, not required in Problem Formulation beyond enabling exact regret.

### Metric registration (no results)

| Metric | Status |
|--------|--------|
| Mean regret over queries 1–3 (`regret_curve[0:3]`) | **Primary**, pre-specified in `RESULTS.md` E3-power-120 **before launch** |
| Final-query regret | Lock: **secondary**; RESULTS also keeps it for continuity with E3-power |
| Empiric numbers / CIs / bake-off wins | **Do not write** — verdict `_pending launch_` |

---

## Unit 5 — Oracle

### Lock

Acquisition that scores candidates with **true** regret.
**Ceiling**, never a baseline to beat; appears in **every** table.

### Code — `oracle_true_regret_score` (not legacy `oracle_score`)

```388:427:plr/mdp.py
def oracle_true_regret_score(...):
    """Ceiling for final-query regret: expected drop in *true* mean-decision regret.
    ...
    """
    def true_reg(weights: Tensor) -> Tensor:
        decision = posterior_mean_decision(weights, belief.particles_w)
        return belief.env.regret(w_true, decision)
    ...
```

| Item | Binding |
|------|---------|
| Enum | `Acquisition.ORACLE = "oracle"` |
| `POPVOI.score_query` | calls `oracle_true_regret_score(belief, phi, w_true, b_true)` |
| Requires | ground-truth `w_true`, `b_true` (evaluation only) |
| Legacy | `oracle_score` = posterior loss under true answer probs — **not** the metric ceiling (`RESULTS.md` ablation note) |

**Writer:** every elicitation table includes oracle as ceiling. Do not claim methods “beat the oracle.”

---

## C-form-* checklist (Problem Formulation may lean on these; no orphan empirics)

| ID | Claim | Where |
|----|-------|-------|
| C-form-1 | $\|w\|=1$ removes BT scale confound | I1 / Unit 1 |
| C-form-2 | BT ≡ PL at $K=2$ | Unit 2 footnote justification |
| C-form-3 | Trivial query BALD = 0 | Method / diagnostics, not PF body |
| C-form-4 | Difficulty = reward gap ⊥ $b$ | Optional PF aside; I2 |
| C-form-5 | $V^\pi(w)$ linear in $w$ for fixed $\pi$ | Bridges PF regret to Method efficiency |
| C-form-6 | POP-VOI select→update→decide | **Method**, cross-ref only |

---

## Engineer FLAGS (summary for Writer)

1. **FLAG — segment $f$:** Formal discounted sum vs Exp 3 **single-state** `env.phi` pairs. Must disclose; see Unit 1.
2. **BT notation:** Paper $\sigma(b\,w\cdot(f_A-f_B))$ with $\|w\|=1$ is OK; code is $\sigma(b\cdot\mathrm{unit}(w)\cdot\Delta f)$.
3. **Oracle name:** Paper “true-regret oracle” = `oracle_true_regret_score` / `Acquisition.ORACLE`, not `oracle_score`.
4. **No empirics in PF:** Primary $t\le 3$ is pre-registered only; do not paste means/CIs until RESULTS has a verdict.
5. **Novelty:** Regret criterion is Regan-class; PF states the objective, not a new paradigm (Intro already cites Regan).
