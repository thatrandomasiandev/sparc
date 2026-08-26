# SPARC — Design specification (pseudocode & math)

> **Version:** v3  
> **Phase:** 1 (Weeks 4–7) — design only, zero implementation.  
> **Authority:** `PROBLEM.md` (scenario & success criterion) > this file > `ROADMAP.md`.

---

## Feedback model

### Trajectory segments and preference queries

Let an environment produce trajectories \(\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)\).

A **segment** \(\sigma\) is a contiguous sub-trajectory of fixed horizon \(H\):

\[
\sigma = (s_t, a_t, \ldots, s_{t+H-1}, a_{t+H-1}), \quad t \in \{0, \ldots, T-H\}.
\]

A **preference query** \(q\) is an ordered pair of segments presented to operator \(i\):

\[
q = (\sigma^{(0)}, \sigma^{(1)}), \quad \sigma^{(0)}, \sigma^{(1)} \in \Sigma,
\]

where \(\Sigma\) is the candidate pool (replay buffer slices, policy rollouts, etc.).

The operator response is \(y \in \{0, 1, \emptyset\}\):

- \(y = 0\): prefer \(\sigma^{(0)}\)
- \(y = 1\): prefer \(\sigma^{(1)}\)
- \(y = \emptyset\): tie / skip (logged, excluded from likelihood loss; synthetic skip rate `p_skip = 0.02` in `EXPERIMENTS.md` stress regime)

Each query is tagged with operator id \(i \in \{1,\ldots,K\}\), window id \(w\), and query time \(t_q\) (when the batch was *issued*, not when the label arrived).

### Per-operator latent preference vector

Each operator \(i\) has a latent vector \(z_i \in \mathbb{R}^d\) that modulates how segment features map to utility.

Shared segment embedding (trajectory encoder):

\[
\phi_\psi(\sigma) = f_\psi\big(\{(s_t, a_t)\}_{t=t_0}^{t_0+H-1}\big) \in \mathbb{R}^m.
\]

Per-operator adapter (linear head for v1 design):

\[
g_i(\sigma) = w_i^\top \phi_\psi(\sigma) + b_i, \quad w_i = A z_i + \mu_w,
\]

where \(A \in \mathbb{R}^{m \times d}\), \(\mu_w \in \mathbb{R}^m\) are global parameters, and \(z_i\) is operator-specific.

**Prior (shared Gaussian across operators):**

\[
z_i \sim \mathcal{N}(0, \sigma_z^2 I_d), \quad i = 1,\ldots,K.
\]

Role in likelihood: \(z_i\) shifts the reward/logit for operator \(i\)'s labels without forcing a single pooled reward — disagreement is explicit.

### Bradley–Terry-with-drift likelihood

Let \(r_\theta(\sigma)\) denote the **pooled** scalar reward from the shared encoder + mean adapter (used for policy training). For operator \(i\)'s label on query \(q = (\sigma^{(0)}, \sigma^{(1)})\):

**Instantaneous utility difference:**

\[
\Delta_i(\sigma^{(0)}, \sigma^{(1)}) = g_i(\sigma^{(0)}) - g_i(\sigma^{(1)}).
\]

**Preference probability (Bradley–Terry):**

\[
P(y=0 \mid \sigma^{(0)}, \sigma^{(1)}, z_i) = \sigma_{\mathrm{BT}}\big(\Delta_i\big) = \frac{1}{1 + \exp(-\Delta_i)}.
\]

**Drift:** preferences are piecewise-stationary in operator–regime pairs \((i, \rho)\). Regime \(\rho\) switches at unknown changepoints \(\{\kappa_j\}\). For label \(n\) from operator \(i\) at global time index \(\tau_n\):

\[
z_i^{(\rho(\tau_n))} = z_i + \delta_{i,\rho(\tau_n)}, \quad \|\delta_{i,\rho}\| \text{ bounded or sparsely non-zero}.
\]

**Likelihood for a single labeled query** \((q_n, y_n, i_n)\):

\[
\mathcal{L}_n(\psi, \{z_i\}, \{\delta_{i,\rho}\}) =
\begin{cases}
-\log \sigma_{\mathrm{BT}}\big(y_n \cdot \Delta_{i_n}(\sigma_n^{(0)}, \sigma_n^{(1)})\big) & y_n \in \{0,1\} \\
0 & y_n = \emptyset \quad \text{(excluded from loss)}
\end{cases}
\]

where \(y_n \cdot \Delta\) means \(\Delta\) if \(y_n=0\) else \(-\Delta\) (logistic loss for preferred segment).

**Dataset loss with drift posterior:** at training step \(t\), use current regime assignment \(\hat{\rho}_t(i)\) from Pillar 4:

\[
\mathcal{L}_{\mathrm{BT+drift}}(\theta) = \sum_{n \in \mathcal{D}_t} w_n \cdot \mathcal{L}_n(\theta),
\]

with optional sample weights \(w_n\) (SURF-style pseudo-labels deferred to Phase 2).

### Communication window (hard batching constraint)

Discrete windows \(w = 0, 1, 2, \ldots\) each of duration \(\Delta T = 900\) s of **simulated mission time** (rover in-sim clock; not wall-clock training compute — locked in `PROBLEM.md`, `EXPERIMENTS.md`).

At window start \(t_w^{\mathrm{open}}\):

1. Optimizer selects a batch \(Q_w = \{q_{w,1}, \ldots, q_{w,B}\}\) **in one shot** from candidate pool \(\mathcal{C}_w\).
2. Batch is transmitted; **no queries are issued** until \(t_{w+1}^{\mathrm{open}}\).
3. Labels \(\{(y_{w,b}, i_{w,b})\}\) may arrive asynchronously before \(t_{w+1}^{\mathrm{close}}\) but are applied to the reward model only when received.

**Formal constraint:**

\[
|Q_w| = B \quad \forall w, \qquad Q_w \cap Q_{w'} = \emptyset \text{ for } w \neq w'.
\]

\[
\text{IssueTime}(q) \in [t_w^{\mathrm{open}}, t_w^{\mathrm{open}} + \epsilon] \text{ only for } q \in Q_w \text{ (single burst)}.
\]

This forbids APReL-style sequential "ask one → update belief → ask next" **within** a window.

---

## Pillar 1 — per-operator latent reward model

### Architecture

```
Input: segment σ, operator id i
  φ = TrajectoryEncoder_ψ(σ)           # shared, dim m=128 (Phase 2 default)
  z_i ~ N(0, σ_z² I)                   # latent, dim d=16
  w_i = A @ z_i + μ_w
  g_i(σ) = w_i · φ + b_i               # operator-specific utility
  r_pool(σ) = mean_i g_i(σ) OR w_pool·φ  # policy-facing pooled reward (config)
```

**Ensemble:** \(M=7\) bootstrap heads \(\{g_i^{(m)}\}_{m=1}^M\) with shared \(\psi\) but independent bootstrap resamples of preference data per head.

### Uncertainty from disagreement

For a segment pair \((\sigma^{(0)}, \sigma^{(1)})\):

\[
\hat{\Delta}^{(m)} = g_i^{(m)}(\sigma^{(0)}) - g_i^{(m)}(\sigma^{(1)}), \quad m=1..M
\]

\[
\mathrm{Disagreement}(\sigma^{(0)}, \sigma^{(1)}) = \mathrm{Std}_m\big[\hat{\Delta}^{(m)}\big]
\]

\[
\mathrm{Uncert}(\sigma) = \mathrm{Std}_m\big[r^{(m)}(\sigma)\big]
\]

### Pseudocode: ensemble training loop (EM-style alternation)

**Design commit:** \(z_i\) is **not** jointly optimized by Adam. During `TrainRewardEnsemble`, each operator's \(z_i\) is always the current **MAP estimate** from `InferOperatorLatent`, held **fixed** for the Adam step on \((\psi, A_m, \mu_{w,m}, b_i)\). After each comms window's labels arrive, `InferOperatorLatent` recomputes \(z_i\) — an **EM-style alternation** (E-step: MAP on \(z_i\); M-step: Adam on shared/adapter params), not single joint SGD.

```
procedure TrainRewardEnsemble(D_prefs, K operators, M=7, epochs=E):
    ψ ← init TrajectoryEncoder
    for m = 1..M:
        D_m ← BootstrapSample(D_prefs, seed=m)
        θ_m ← {A_m, μ_w_m, {b_i,m}}          # no z_i in θ_m
    for epoch = 1..E:
        for i = 1..K:
            z_i ← InferOperatorLatent(i, labels for op i in D_prefs)   # E-step; frozen during M-step
        for m = 1..M:
            batch ← SampleBatch(D_m, operator ids included)
            L ← 0
            for (σ0, σ1, y, i) in batch:
                φ0 ← Encoder_ψ(σ0); φ1 ← Encoder_ψ(σ1)
                w_i ← A_m @ z_i + μ_w_m                                    # z_i fixed
                Δ ← w_i · φ0 + b_{i,m} - (w_i · φ1 + b_{i,m})
                L += BradleyTerryLoss(y, Δ)
            ψ, θ_m ← Adam(∇L)              # M-step; z_i not in autograd graph
    return ψ, {θ_m}, {z_i}
```

**MAP for \(z_i\)** (E-step — after each window's labels arrive, and at start of each ensemble epoch):

```
procedure InferOperatorLatent(i, D_i recent labels):
    z_i ← argmin_z Σ_n -log σ_BT(y_n · (w(z)^T φ_n0 - w(z)^T φ_n1)) + ||z||²/(2σ_z²)
    return z_i
```

---

## Pillar 2 — latency-aware batch query optimizer

### Expected information gain (EIG) scoring

For candidate query \(q = (\sigma^{(0)}, \sigma^{(1)})\), operator \(i\), current ensemble \(\{\theta_m\}\), belief over \(z_i\):

**Predictive preference entropy:**

\[
H[y \mid q, \mathcal{D}] = -\sum_{y \in \{0,1\}} p(y \mid q, \mathcal{D}) \log p(y \mid q, \mathcal{D}),
\]

\[
p(y=0 \mid q, \mathcal{D}) = \mathbb{E}_{z_i \sim p(z_i \mid \mathcal{D})}\big[\sigma_{\mathrm{BT}}(\Delta_i)\big] \approx \frac{1}{M}\sum_m \sigma_{\mathrm{BT}}(\hat{\Delta}^{(m)}).
\]

**EIG (Bernoulli observation):**

\[
\mathrm{EIG}(q, i) = H[y \mid q, \mathcal{D}] - \mathbb{E}_{y \sim p(y \mid q)}\big[H[y \mid q, \mathcal{D} \cup \{(q,y,i)\}]\big].
\]

Monte Carlo with 2 outcomes (exact for binary preferences):

```
function ScoreEIG(q, i, ensemble, D):
    p0 ← mean_m σ_BT(Δ_i^{(m)}(q))
    H_prior ← BernoulliEntropy(p0)
    H_post ← 0
    for y in {0, 1}:
        py ← (1-y)*(1-p0) + y*p0   # P(y)
        D' ← D ∪ {(q, y, i)}
        p0' ← PredictPrefProb(q, i, ensemble, D')
        H_post += py * BernoulliEntropy(p0')
    return H_prior - H_post
```

### Implementation note (EIG posterior update — no retrain per candidate)

`PredictPrefProb(q, i, ensemble, D')` must **not** re-run `TrainRewardEnsemble` or any full ensemble epoch. Literal retraining per candidate × 2 hypothetical outcomes cannot finish inside one comms window (ROADMAP Week 11; \(\Delta T = 900\) s sim mission time in `EXPERIMENTS.md`).

**Approximation (v2):** With \(\psi\) and ensemble heads fixed, treat the on-duty operator's adapter \(w_i = A_m z_i + \mu_w\) as a logistic linear model on cached embeddings \(\phi(\sigma^{(0)}), \phi(\sigma^{(1)})\). For hypothetical label \((q, y, i)\), apply **one Newton/Raphson step** on the MAP objective for \(w_i\) only (equivalently: Laplace update with rank-1 Hessian add from the Bradley–Terry term). Reuse the E-step \(z_i\) from `InferOperatorLatent`; do not backprop into \(\psi\).

**Asymptotic cost per candidate query** (checkable against window budget):

\[
O\big(|\{0,1\}| \cdot m\big) = O(m) \quad \text{per call to } \texttt{PredictPrefProb},
\]

with \(m = 128\). For \(|C_w| = 500\) candidates, `ScoreEIG` costs \(O(|C_w| \cdot m) \approx 6.4 \times 10^4\) floating ops — orders of magnitude below wall-clock optimizer gate (\(< 0.1 \cdot \Delta T\) sim-time budget applies to mission clock; optimizer wall-clock must stay \(\ll\) training step time, typically \(< 1\) s total for batch selection).

### Batch selection under per-window budget

Given candidate set \(\mathcal{C}_w\), batch size \(B\), operator schedule \(\pi_i(w)\) (which operator is on duty for window \(w\)):

```
procedure SelectBatch(C_w, B, operator i, ensemble, D):
    scores ← empty map
    for q in C_w:
        scores[q] ← ScoreEIG(q, i, ensemble, D) + λ_div · DiversityBonus(q, selected)
    selected ← empty list
    while |selected| < B:
        q* ← argmax_{q ∈ C_w \ selected} scores[q]   # greedy
        selected.append(q*)
        # optional: submodular penalty so segments don't overlap
        for q in C_w:
            scores[q] -= η · max_{q' ∈ selected} Similarity(q, q')
    return selected
```

**Hard window coupling:**

```
procedure OnWindowOpen(w):
    assert no outstanding unissued queries
    Q_w ← SelectBatch(C_w, B, OperatorOnDuty(w), ...)
    Transmit(Q_w)   # single burst
    freeze query issuance until OnWindowOpen(w+1)
```

### Difference from APReL (mechanism, not name)

| APReL batch active learner | SPARC Pillar 2 |
|----------------------------|----------------|
| Optimizes acquisition (disagreement, volume removal, mutual information) over a **fixed trajectory set**, often **sequentially** updating belief between queries | Selects **entire batch \(Q_w\)** at window open using EIG; **no** mid-window belief updates or re-optimization |
| Batch methods (medoids, greedy) diversify queries in feature space for **immediate** human response | Batch chosen under **known label latency**; candidates must remain valid until labels return (segment pool from **frozen policy snapshot** at \(t_w^{\mathrm{open}}\)) |
| Single implicit user / GP belief over reward weights | **Operator-indexed** likelihood; EIG computed w.r.t. **on-duty operator** \(i\) and \(z_i\) |
| No comms-window constraint | **\(|Q_w| = B\)** and issue-time burst constraint are hard |

APReL volume removal **shrinks** the feasible weight polytope after each answer; SPARC **does not** assume answers arrive before the next query — it maximizes joint informativeness of \(B\) queries under **pre-window** posterior.

---

## Pillar 3 — confidence-gated policy bounding

### Ensemble disagreement signal

For each policy transition \((s, a)\) or segment \(\sigma\) used in RL update:

\[
D(\sigma) = \mathrm{Std}_m\big[r^{(m)}(\sigma)\big]
\]

Alternatively on TD targets: \(D(s,a) = \mathrm{Std}_m\big[Q^{(m)}(s,a)\big]\) (Phase 2 choice — **see flags**).

### Threshold

Let \(\mathcal{B}_t\) be the replay buffer at update time \(t\). Define the **rolling disagreement percentile** (global scalar, not per \((s,a)\)):

\[
q_{90,t} = \mathrm{P90}\big(\{D(\sigma) : \sigma \in \mathcal{B}_t\}\big),
\]

where \(D(\sigma) = \mathrm{Std}_m[r^{(m)}(\sigma)]\) on segment embeddings drawn from \(\mathcal{B}_t\).

**Update cadence:** recompute \(q_{90,t}\) **once per comms window** when window \(w\)'s labels are merged into the buffer (same time as the E-step `InferOperatorLatent` call). Hold \(q_{90,t}\) fixed between window boundaries.

**Threshold (scalar, shared across all transitions in that window):**

\[
\tau_t = \tau_0 + \kappa \cdot q_{90,t}.
\]

- \(\tau_0\): floor constant (Phase 2 default in `EXPERIMENTS.md`)
- \(\kappa\): unitless scale (default 1.0)
- **Gate active when** \(D(\sigma) > \tau_t\) for segment \(\sigma\) associated with the transition

### Bounding modes (pseudocode)

Applied to policy gradient / SAC update using reward \(\hat{r}\) or advantage \(A\):

```
function ApplyBounding(r_hat, D, tau, mode):
    if D <= tau:                        # tau = τ_t from rolling P90 (scalar)
        return r_hat                    # pass-through
    match mode:
        case "pass-through":
            return r_hat
        case "shrink":
            α ← tau / D                 # ∈ (0,1)
            return α * r_hat            # shrink toward 0 (neutral reward)
        case "freeze":
            return STOP_UPDATE          # skip gradient for this transition
```

**Policy-level freeze:** if fraction of minibatch with \(D > \tau\) exceeds \(f_{\max}\) (default 0.5), skip entire policy update step for that iteration.

---

## Pillar 4 — online preference-drift detector

### Sequential probability ratio test (SPRT)

Null \(H_0\): labels from operator \(i\) still match regime \(\rho\) (current \(z_i^{(\rho)}\)).

Alternative \(H_1\): preference odds shifted by factor \(e^{\eta}\) (equivalent to \(\Delta \to \Delta + \eta\)).

For each incoming label \(n\) from operator \(i\) after window \(w\):

\[
\Lambda_n = \sum_{j=1}^{n} \log \frac{P(y_j \mid H_1, q_j, z_i^{(\rho)})}{P(y_j \mid H_0, q_j, z_i^{(\rho)})}
\]

**Log-likelihood ratio contribution:**

\[
\ell_j = \log \sigma_{\mathrm{BT}}(y_j \tilde{\Delta}_j^{(1)}) - \log \sigma_{\mathrm{BT}}(y_j \tilde{\Delta}_j^{(0)}),
\]

where \(\tilde{\Delta}_j^{(0)}\) uses \(H_0\) params, \(\tilde{\Delta}_j^{(1)}\) uses \(H_1\) (shifted).

**SPRT decision boundaries** ( Wald ):

\[
A = \log\frac{1-\beta}{\alpha}, \quad B = \log\frac{\beta}{1-\alpha}
\]

- \(\alpha = 0.05\) (false alarm), \(\beta = 0.10\) (missed drift) — locked in `EXPERIMENTS.md`

```
procedure UpdateDriftSPRTOperator(i, new labels batch L):
    for each label (q, y) in L:
        Λ ← Λ + logLikRatio(y, q, H1) - logLikRatio(y, q, H0)
        if Λ >= A:
            TriggerDrift(i, regime ρ)
            Λ ← 0
            break
        if Λ <= B:
            Λ ← 0    # accept H0, reset
    return state(i, Λ, ρ)
```

### Targeted re-querying (control flow into Pillar 2)

When `TriggerDrift(i, ρ)` fires:

1. Increment regime: \(\rho \leftarrow \rho + 1\) for operator \(i\); initialize \(\delta_{i,\rho}\) from posterior spike or reset \(z_i\) MAP.
2. Set **re-query flag** `REQUERY[i] = true`.
3. On next `OnWindowOpen(w)`:
   - **Override** candidate pool: \(\mathcal{C}_w' = \{(q, i)\}\) drawn from segments where \(| \mathbb{E}[r \mid H_0] - \mathbb{E}[r \mid H_1] |\) is largest under historical queries (top 20% disagreement queries from last 2 windows).
   - Force **minimum** \(B_{\mathrm{requery}} = \lceil B/2 \rceil\) queries assigned to operator \(i\) before filling remainder with standard EIG selection.
4. Clear `REQUERY[i]` after one re-query window unless SPRT fires again.

```
procedure OnWindowOpen(w):
    if any REQUERY[i]:
        i* ← argmax_i REQUERY[i]
        Q_requery ← TopDisagreementQueries(i*, count=B/2)
        Q_rest ← SelectBatch(C_w \ Q_requery, B - |Q_requery|, ...)
        Q_w ← Q_requery ∪ Q_rest
    else:
        Q_w ← SelectBatch(C_w, B, ...)
    Transmit(Q_w)
```

---

## Red-team notes

> **Scenario source:** `PROBLEM.md` (locked 2026-08-25): delayed/multi-operator URC rover ops, 3 rotating operators, batched comms feedback (\(\Delta T = 900\) s sim), safety envelope on high-uncertainty actions. Failure modes below assume that scenario.

### Pillar 1 — per-operator latent reward model

**Failure mode:** With ≤20% of PEBBLE's query budget, each of 3 operators may contribute <30 labels; \(z_i \in \mathbb{R}^{16}\) is **unidentifiable**, and adapters collapse to the shared encoder — SPARC silently reduces to single-reward PEBBLE but with **worse** data efficiency.

**Falsifiable check:** After training, \(\mathrm{Var}_i(\|w_i - \bar w\|) < \epsilon_{z}\) with \(\epsilon_{z}=0.01\) while operators have **ground-truth** distinct bias vectors in sim → Pillar 1 failed.

### Pillar 2 — latency-aware batch query optimizer

**Failure mode:** Rover comms windows force policy snapshot at \(t_w^{\mathrm{open}}\); by the time labels return, policy has drifted 15+ minutes of sim time — labeled segments are **off-manifold**, EIG was computed for the wrong behavior distribution, and batch labels **hurt** the reward model.

**Falsifiable check:** Compare reward-model NLL on labels received vs. labels evaluated on **current** policy rollouts; if NLL\_received − NLL\_current > 0.5 nats sustained over 5 windows → batch optimizer failed latency assumption.

### Pillar 3 — confidence-gated policy bounding

**Failure mode:** On narrow URC navigation corridors, \(D(\sigma) > \tau\) for >50% of replay batch always (ensemble disagree everywhere near obstacles) → **freeze** mode blocks all learning after ~100k steps; rover never reaches PEBBLE asymptotic return.

**Falsifiable check:** Policy update skip rate > 80% for 10 consecutive training iterations while task success rate plateaus below 30% of PEBBLE baseline → gating too conservative.

### Pillar 4 — online preference-drift detector

**Failure mode:** Operator rotation (speed-prioritized vs safety-prioritized crew) looks like **preference drift** to SPRT when regime is actually **multi-operator stationarity** → false positive every shift change, triggering re-query storms that burn the 20% query budget in 3 windows.

**Falsifiable check:** Under **no** scripted drift (only operator rotation with fixed per-operator biases), SPRT false-alarm rate > 1 trigger per 5 windows at \(\alpha=0.05\) → drift detector confuses rotation with drift.

---

## Design revision log

| Version | Date | Notes |
|---------|------|-------|
| v1 | 2026-08-25 | Initial draft — Phase 1 Weeks 4–7. Pending external review (Week 6). |
| v2 | 2026-08-25 | Fixed Pillar 3 τ to rolling replay-buffer P90, updated once per comms window. |
| v2 | 2026-08-25 | Resolved z_i optimization ambiguity as EM-style alternation (MAP E-step / Adam M-step). |
| v2 | 2026-08-25 | Added Pillar 2 EIG implementation note: one Newton step on w_i, O(m) per candidate (no retrain). |
| v3 | 2026-08-25 | Aligned comms window to sim mission time only; locked tie/skip handling to EXPERIMENTS.md. |
