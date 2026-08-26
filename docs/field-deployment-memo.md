# Field deployment memo (Phase 0, Week 2)

> **Purpose:** Document where mainstream preference-based RL (PbRL) assumes abundant, synchronous, single-source feedback — and why those assumptions fail for a physically deployed robot.  
> **Authority:** Informs `PROBLEM.md` (locked scenario). Does not override `DESIGN.md` or `EXPERIMENTS.md`.  
> **Date:** 2026-08-25

---

## Thesis

Lab PbRL treats human feedback as a **cheap, always-on oracle**: one attentive teacher, queries answered within seconds, preferences that do not shift mid-training. Field robots — especially comms-limited rover ops with rotating crews — violate all three. Methods that work in simulation at 1,000 on-demand labels do not automatically transfer when labels arrive in **batched windows**, come from **disagreeing operators**, and may **drift** after mission priority changes.

SPARC is designed against these failures explicitly, not as an incremental tweak to PEBBLE.

---

## Five lab assumptions that break in the field

### 1. Synchronous, on-demand queries

**Lab:** [PEBBLE](https://proceedings.mlr.press/v139/lee21i.html) and [Christiano et al.](https://arxiv.org/abs/1706.03741) issue a preference query when the learner is ready; the labeler responds before the next query. [APReL](https://github.com/Stanford-ILIAD/APReL) active learners often **update belief between queries** (disagreement, volume removal, mutual information).

**Field:** Rover downlink/uplink is **windowed**. Operators review telemetry in batches; the robot cannot "ask one more question" until the next pass. A query issued at window open may label behavior from a **policy snapshot** that is stale when the label arrives.

**SPARC response (Pillar 2):** Select the **entire batch** \(Q_w\) at window open under hard budget \(B=8\), \(\Delta T = 900\) s sim mission time — no mid-window re-optimization. Candidates drawn from a frozen policy snapshot.

### 2. Single consistent annotator

**Lab:** PEBBLE, [SURF](https://arxiv.org/abs/2203.10050), and [RUNE](https://openreview.net/pdf?id=OWZVD-l-ZrC) evaluate with one synthetic or human teacher. Pooling labels is implicit.

**Field:** Rotating ops crews bring **different priorities** (speed vs clearance vs energy). Averaging their preferences into one reward destroys the signal each operator provides and misattributes labels during rotation.

**SPARC response (Pillar 1):** Shared trajectory encoder + **per-operator adapter** under shared prior; operator id \(i\) in the likelihood. With ≤20% of PEBBLE's query budget, identifiability is a real risk — see red-team checks in `DESIGN.md`.

### 3. Stationary preferences

**Lab:** [B-Pref](https://arxiv.org/abs/2111.03026) models label noise and irrationality but not **mid-mission priority shifts** (fatigue, weather, changed goal).

**Field:** What "good navigation" means can change when mission context changes. Silent blending of old and new preferences produces a reward model that satisfies no operator.

**SPARC response (Pillar 4):** Sequential probability ratio test (SPRT) over incoming labels; on trigger, **targeted re-query** window for the affected operator (see `EXPERIMENTS.md` Drift-1 stress test at \(f_{\mathrm{drift}} = 0.5\) of training).

### 4. Uncertainty → explore, not bound

**Lab:** RUNE uses ensemble disagreement as an **exploration bonus** — visit states where the reward model is uncertain.

**Field:** High reward uncertainty near obstacles is a **safety signal**, not an invitation to explore. Deployed systems need clip/freeze behavior when disagreement exceeds threshold.

**SPARC response (Pillar 3):** Confidence-gated policy bounding — shrink or freeze policy updates when \(D(\sigma) > \tau_t\), with \(\tau_t\) from rolling replay-buffer P90 (updated once per comms window).

### 5. Multi-user methods target the wrong economics

**Lab / NLP:** Variational per-annotator learning (VPL), Crowd-PrefRL, and distributional preference models assume **many cheap, parallel, asynchronous** labelers (crowd workers, LLM raters).

**Field:** One robot, **three bandwidth-limited operators**, sparse labels each — closer to mission ops than to Chatbot Arena. Per-user models under extreme sparsity can **destabilize** learning if not structured carefully.

**SPARC response:** Shared representation with lightweight per-operator heads (not independent reward models per user); EM-style MAP on \(z_i\) alternated with ensemble training — design commit in `DESIGN.md` v3.

---

## Scenario shortlist (Week 2 → locked Week 3)

| Rank | Scenario | Evidence generatable this year | SPARC pillar fit |
|------|----------|--------------------------------|------------------|
| **1 — locked** | **URC rover, delayed multi-operator comms** | Sim `sparc-rover-nav-v0` + URC hardware (Phase 5); L'SPACE ops mental model | All four pillars naturally motivated |
| 2 | Single-caregiver assistive arm | Tabletop fallback; real HITL feasible | Strong for Pillars 1 & 4; weaker comms-delay story |
| 3 | Generic rotating field crew (sim-only) | DMControl/MetaWorld stress tests | Good for sim; Phase 5 hardware harder |

**Locked choice:** #1 with robot-arm tabletop fallback if URC access is limited (`PROBLEM.md`).

---

## Implication for evaluation

B-Pref-style benchmarks (single annotator, zero delay) remain necessary as an **easy regime** non-regression check (`EXPERIMENTS.md`). The **primary success gate** must run under stress: comms windows, \(K=3\) disagreeing annotators, scripted drift — otherwise SPARC's four pillars are untested window dressing.

---

## Related documents

| Document | Role |
|----------|------|
| `docs/prior-art-matrix.md` | Method-by-method comparison |
| `PROBLEM.md` | Locked scenario and falsifiable success criteria |
| `DESIGN.md` v3 | Four pillars, pseudocode, red-team notes |
| `EXPERIMENTS.md` v2 | Locked protocol, metrics, baselines |
