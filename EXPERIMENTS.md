# SPARC — Experimental protocol

> **Phase:** 1 (Week 7 deliverable) — specification only.  
> **Version:** v2  
> **Authority:** `PROBLEM.md` > `DESIGN.md` > this file.  
> **Status:** Locked for Phase 3+ runs unless `PROBLEM.md` revision forces a protocol bump (logged below).

---

## Simulation environments

### DeepMind Control (general validation)

| Task ID | Suite | Horizon `H` (segment) | Train steps | Notes |
|---------|-------|-------------------------|-------------|-------|
| `walker-walk` | DMControl | 50 | 500,000 | Primary locomotion baseline (matches B-Pref / PEBBLE literature) |
| `cheetah-run` | DMControl | 50 | 500,000 | Speed–stability tradeoff stress |
| `quadruped-walk` | DMControl | 50 | 500,000 | Higher-dim actions; robustness check |

**Wrapper:** `dmc2gym` equivalent, frame skip 1, action repeat 1, episode length 1000, image observations **off** (state-only, same as B-Pref reproduction).

### MetaWorld (manipulation validation)

| Task ID | Horizon `H` | Train steps |
|---------|-------------|-------------|
| `metaworld_hammer-v2` | 50 | 2,000,000 |
| `metaworld_door-open-v2` | 50 | 2,000,000 |

**Wrapper:** MetaWorld v2, sparse success signal, same protocol as [B-Pref](https://github.com/rll-research/BPref).

### Rover navigation (target deployment proxy)

**Env name:** `sparc-rover-nav-v0` (to be implemented Phase 2 Week 9; spec fixed here).

| Parameter | Value |
|-----------|-------|
| State dim | 8 — \((x, y, \theta, v, \omega, \text{lidar}_{1..3})\) |
| Action dim | 2 — \((v_{\mathrm{cmd}}, \omega_{\mathrm{cmd}})\) |
| Map | 20 m × 20 m obstacle field, 15 static obstacles, 2 goal zones |
| Episode length | 800 steps (Δt = 0.05 s → 40 s sim time per episode) |
| Train steps | 750,000 |
| Success | Reach goal radius 0.5 m within 800 steps |
| Ground-truth reward (sim oracle only) | \(r = -\|v - v_{\mathrm{ref}}\| - 0.5\cdot\mathbb{1}[\text{collision}] + 2\cdot\mathbb{1}[\text{goal}]\) |

*Train steps 750,000:* primary deployment proxy; above DMControl (500k) to accumulate enough comms windows over training, below MetaWorld (2M) because navigation is lower-dimensional than long-horizon manipulation.

**Comms simulation (stress regime):** \(\Delta T = 900\) s of **simulated mission time** (rover in-sim clock only—not wall-clock training compute). A comms window closes every 900 simulated seconds regardless of hardware speed. With 40 s sim time per episode (800 steps × 0.05 s), each window spans **22.5 episodes** (\(900 / 40 = 22.5\)). Batch size \(B = 8\) queries/window.

---

## Synthetic annotator configurations

Three operators \(K=3\). Ground-truth utility for labeling:

\[
U_i(\sigma) = \alpha_i^\top \bar{f}(\sigma), \quad \bar{f}(\sigma) = \big[\text{mean speed}, \text{min obstacle dist}, \text{energy use}\big]
\]

**Per-operator bias vectors** \(\alpha_i\) (fixed unless drift fires):

| Operator | \(\alpha_i\) (speed, clearance, energy) | Role |
|----------|----------------------------------------|------|
| Op-1 | `(1.0, 0.2, 0.1)` | Speed-prioritized (L'SPACE ops lead mental model) |
| Op-2 | `(0.3, 1.0, 0.2)` | Safety-prioritized |
| Op-3 | `(0.4, 0.4, 1.0)` | Energy-prioritized |

**Label noise:** pairwise comparison uses true \(U_i\); with probability `p_mistake = 0.05`, flip label.

**Ties/skips:** with probability `p_skip = 0.02`, return \(y=\emptyset\).

**Operator rotation schedule:** window \(w \mod 3 = i-1\) → labels from Op-\(i\) (deterministic rotation).

**Scripted drift (stress regime only):**

Drift fires at a **fixed fraction** \(f_{\mathrm{drift}} = 0.5\) of each environment's `Train steps` (not a universal step count), so drift occurs at the same relative point in training for every env.

| Event | When (per env) | Effect |
|-------|----------------|--------|
| Drift-1 | Env step \(\lfloor 0.5 \times \mathrm{TrainSteps}\rfloor\) (Op-2 only): walker-walk **250,000**; cheetah-run / quadruped-walk **250,000**; hammer-v2 / door-open-v2 **1,000,000**; sparc-rover-nav-v0 **375,000** | Replace \(\alpha_2\) with `(0.8, 0.5, 0.1)` — safety operator temporarily accepts faster trajectories |

**Easy regime (Phase 4 weeks 20–21):** single synthetic operator Op-2 only, `p_mistake=0`, no drift, on-demand queries (no window constraint) for baseline parity runs.

---

## Baselines and source implementations

All baselines adapted inside SPARC harness (Phase 3) with **identical** env wrappers, seeds, and query budgets.

| Method | Source repository | Entry script / module to adapt |
|--------|-------------------|--------------------------------|
| **PEBBLE** | [rll-research/BPref](https://github.com/rll-research/BPref) | `train_PEBBLE.py`, `scripts/*/run_PEBBLE.sh` |
| **PrefPPO** (vanilla) | [rll-research/BPref](https://github.com/rll-research/BPref) | `train_PrefPPO.py`, `scripts/*/run_PrefPPO.sh` |
| **SURF** | [alinlab/SURF](https://github.com/alinlab/SURF) | `train_PEBBLE_semi_dataaug.py` |
| **RUNE** | [rll-research/rune](https://github.com/rll-research/rune) | `train_PEBBLE_explore.py` (PEBBLE+RUNE) |
| **APReL batch active learner** | [Stanford-ILIAD/APReL](https://github.com/Stanford-ILIAD/APReL) | `QueryOptimizerDiscreteTrajectorySet.optimize(..., optimization_method='medoids')` with `acquisition_func_str='disagreement'` |

**SPARC** (proposed): full stack per `DESIGN.md` v3.

**Query budget (primary success comparison):**

- PEBBLE reference budget: `N_PEBBLE = 1000` preference labels (walker-walk, B-Pref default scale).
- SPARC target budget: `N_SPARC = 200` labels (≤20% of 1000).
- All methods capped at same **environment interaction** steps (500k walker / 2M metaworld / 750k rover-nav).

---

## Metrics (exact definitions)

Let \(R_t\) be episodic return at training checkpoint \(t\), \(N_q(t)\) total preference queries consumed by \(t\).

### 1. Policy return vs. queries used

**Learning curve:** plot \(\mathbb{E}_{\mathrm{seeds}}[R_t \mid N_q(t)]\) with \(R_t\) from 10 consecutive evaluation episodes (deterministic policy, no exploration noise) every 10k env steps.

**Asymptotic return:** \(\bar{R}_\infty = \mathrm{mean}(R_t)\) over last 5 evaluation checkpoints (last 50k steps).

### 2. Reward-model accuracy vs. ground truth

On held-out segment pairs \(\{( \sigma^{(0)}_j, \sigma^{(1)}_j)\}\) labeled by sim oracle (not training labels):

\[
\mathrm{Acc}_{\mathrm{RM}} = \frac{1}{J}\sum_{j=1}^{J} \mathbb{1}\big[\mathrm{sign}(\hat{r}(\sigma^{(0)}_j) - \hat{r}(\sigma^{(1)}_j)) = \mathrm{sign}(U_i(\sigma^{(0)}_j) - U_i(\sigma^{(1)}_j))\big]
\]

Evaluate per operator \(i\) and macro-average. Held-out set size \(J=500\) pairs, refreshed each eval.

### 3. Regret under drift

After Drift-1 at step \(\lfloor 0.5 \times \mathrm{TrainSteps}\rfloor\), over the subsequent \(\lfloor 0.1 \times \mathrm{TrainSteps}\rfloor\) env steps:

\[
\mathrm{Regret}_{\mathrm{drift}} = \frac{1}{W}\sum_{w=1}^{W} \big(R_w^* - R_w\big),
\]

where \(R_w^*\) is return under oracle policy for Op-2 **post-drift** preferences, \(R_w\) is learner return in that window.

### 4. Wall-clock per query

\[
T_{\mathrm{query}} = \frac{1}{N_{\mathrm{windows}}}\sum_w \big(t_w^{\mathrm{labels\_complete}} - t_w^{\mathrm{open}}\big) / |Q_w|
\]

Measured in sim for comms-delay runs; optimizer CPU time for batch selection logged separately (must be \(< 0.1 \cdot \Delta T\) — design gate from ROADMAP Week 11).

### 5. Primary success gate (from `PROBLEM.md`)

**Pass if:** \(\bar{R}_\infty^{\mathrm{SPARC}} \geq \bar{R}_\infty^{\mathrm{PEBBLE}} - \delta\) with \(\delta = 0\) (strict) or CI overlap per statistical test below, **and** \(N_q^{\mathrm{final, SPARC}} \leq 200\), under stress annotator + comms windows.

---

## Seeds and statistical testing

| Parameter | Value |
|-----------|-------|
| Random seeds | `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}` — **10 seeds** per (method, env, regime) |
| Torch / NumPy / env | All seeded from master seed list above |

**Primary comparison:** SPARC vs PEBBLE on \(\bar{R}_\infty\) under stress regime.

**Test:** **Welch's t-test** (two-sided, unequal variance) across 10 seeds at \(\alpha = 0.05\).

**Non-inferiority:** SPARC not worse than PEBBLE if lower bound of 95% CI for \((\bar{R}_\infty^{\mathrm{SPARC}} - \bar{R}_\infty^{\mathrm{PEBBLE}})\) exceeds \(-\Delta_{\mathrm{NI}}\) with \(\Delta_{\mathrm{NI}} = 50\) return points on walker-walk (≈5% of typical PEBBLE asymptote ~1000).

**Query-efficiency:** one-sided test on \(N_q\) at fixed \(\bar{R}_\infty\) threshold (PEBBLE median asymptote): Wilcoxon signed-rank on paired seeds for \(N_q^{\mathrm{SPARC}} < N_q^{\mathrm{PEBBLE}}\).

**Multiple comparisons:** Benjamini–Hochberg FDR \(q=0.10\) across the 6 env×regime cells when reporting full suite.

---

## Protocol revision log

| Version | Date | Notes |
|---------|------|-------|
| v1 | 2026-08-25 | Initial locked protocol — Phase 1 Week 7. |
| v2 | 2026-08-25 | Fixed ΔT self-contradiction (sim mission time only; 22.5 ep/window). |
| v2 | 2026-08-25 | Added rover-nav Train steps 750,000 with justification. |
| v2 | 2026-08-25 | Drift-1 at fixed fraction \(f_{\mathrm{drift}}=0.5\) of Train steps per env (not universal step count). |

---

## Provisional assumptions (superseded by `PROBLEM.md`)

The following were fixed in this document during Phase 1 v2 before `PROBLEM.md` was locked; they now match `PROBLEM.md` (2026-08-25):

- Rover-proxy env spec (not yet tied to URC hardware kinematics)
- \(\Delta T = 900\) s, \(B = 8\)
- Drift event at \(f_{\mathrm{drift}}=0.5\) of Train steps on Op-2 only
- PEBBLE reference budget 1000 labels
- Non-inferiority margin \(\Delta_{\mathrm{NI}} = 50\) on walker-walk
