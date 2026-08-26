# SPARC — Problem statement

> **Phase:** 0 (Week 3 scope lock) — authoritative over `DESIGN.md` and `EXPERIMENTS.md`.  
> **Locked:** 2026-08-25  
> **Downstream specs:** `DESIGN.md` v3, `EXPERIMENTS.md` v2

---

## One-sentence problem

Physically deployed robots that learn from human preferences receive **sparse, delayed, and inconsistent** feedback — but mainstream preference-based RL (PbRL) assumes a **single attentive labeler** who answers queries **on demand** with **stationary** preferences.

---

## Primary deployment scenario

**Delayed, multi-operator preference feedback for a semi-autonomous URC rover** (NASA L'SPACE ops-team mental model).

| Element | Locked specification |
|---------|---------------------|
| **Robot / task** | Mobile rover navigation and behavior shaping; sim proxy `sparc-rover-nav-v0` (Phase 2); URC hardware in Phase 5; tabletop robot-arm fallback if rover access is limited |
| **Operators** | \(K = 3\) rotating human labelers with distinct, persistent biases (speed-prioritized, safety-prioritized, energy-prioritized) |
| **Feedback form** | Pairwise preferences over **trajectory segments** (not scalar step rewards) |
| **Comms constraint** | **Batched windows only** — no on-demand queries mid-window. Window duration \(\Delta T = 900\) s of **simulated mission time** (rover in-sim clock); batch size \(B = 8\) queries per window. See `EXPERIMENTS.md` for episode scaling (22.5 episodes/window at 40 s sim/episode) |
| **Operator schedule** | Deterministic rotation: window \(w \mod 3\) assigns labels to Op-\((w \mod 3) + 1\) |
| **Safety envelope** | Policy updates must be **bounded or frozen** when the reward-model ensemble disagrees beyond a threshold — uncertainty is a hard safety gate, not only an exploration bonus |
| **Stress beyond rotation** | Scripted mid-training preference drift on one operator (Op-2) at 50% of each env's training horizon — tests Pillar 4, not normal ops |

**Why lab PbRL breaks here:** PEBBLE, SURF, and RUNE assume one consistent annotator and synchronous queries. APReL's active learners update belief between queries — incompatible with a hard comms blackout. Multi-annotator LLM methods (VPL, crowd-PrefRL) target cheap parallel text labor, not one bandwidth-limited robot with a physical safety envelope.

**Gap this project closes:** No open engine combines (1) per-operator preference modeling, (2) latency-aware batch query optimization, (3) confidence-gated policy bounding, and (4) online preference-drift detection — validated under the stress regime above and, in Phase 5, on physical hardware.

---

## Falsifiable success criterion

### Primary gate (simulation, Phase 4 — stress regime)

**Pass if all of the following hold** on the primary rover-proxy env and on DMControl `walker-walk` (generalization check), with 10 seeds and tests defined in `EXPERIMENTS.md`:

1. **Sample efficiency:** SPARC uses **≤20% of PEBBLE's preference-query budget** — i.e. \(N_q^{\mathrm{SPARC}} \leq 200\) when PEBBLE reference budget \(N_{\mathrm{PEBBLE}} = 1000\).
2. **Task quality:** SPARC asymptotic return \(\bar{R}_\infty^{\mathrm{SPARC}}\) is **not worse than PEBBLE** — non-inferiority via Welch's t-test / 95% CI with margin \(\Delta_{\mathrm{NI}} = 50\) return points on walker-walk (protocol in `EXPERIMENTS.md`).
3. **Stress conditions active:** simulated comms windows (\(\Delta T = 900\) s sim), \(K = 3\) disagreeing synthetic annotators (`p_{\mathrm{mistake}} = 0.05`), and scripted Drift-1 on Op-2 at \(f_{\mathrm{drift}} = 0.5\) of training steps.

### Secondary gates (must not regress)

- **Easy regime** (single annotator, zero delay, no drift): SPARC must not underperform PEBBLE at equal query budget.
- **Reward-model accuracy:** macro-averaged held-out pairwise accuracy ≥ PEBBLE at SPARC's query budget.
- **Safety gating (Phase 2):** in toy MDPs with known safe bounds, zero policy gradient steps executed when ensemble disagreement exceeds \(\tau_t\) in freeze mode.

### Tertiary (hardware, Phase 5 — planned, not required for sim pass)

3–5 real annotators under a constrained query schedule on URC or fallback arm; document synthetic↔real gap in `RESULTS.md`.

---

## What this project is explicitly NOT trying to solve

- **Not** reimplementing or improving [STAR](https://github.com/alexdobin/STAR) (RNA-seq aligner) — STAR is the **engineering discipline reference** only.
- **Not** LLM / chatbot alignment SOTA (Chatbot Arena, DPO at scale).
- **Not** a general-purpose PbRL framework for every domain — one deployment class first (delayed multi-operator field robot).
- **Not** replacing hand-engineered safety controllers or formal verification — bounding is a learned-reward uncertainty gate, not a certificate.
- **Not** unlimited query budgets, real-time teleoperation preference labeling, or crowd-sourced async labeling economics.
- **Not** pillar implementation before `DESIGN.md` is locked (Phase 1 complete); **not** training runs before `EXPERIMENTS.md` is locked.

---

## Proposed approach (scope pointer)

**SPARC** — Sparse-feedback Preference Active Reward Core — integrates four pillars (per-operator latent reward model, latency-aware batch query optimizer, confidence-gated policy bounding, online preference-drift detector). Algorithmic detail, pseudocode, and red-team notes: **`DESIGN.md` v3**. Metrics, baselines, seeds, and statistical tests: **`EXPERIMENTS.md` v2**.

Prior-art comparison matrix: `docs/prior-art-matrix.md`. Field deployment rationale: `docs/field-deployment-memo.md`. Sequencing: `ROADMAP.md`.
