# Related Work — Engineer notes

**Scope:** Related Work cite-key map + wording guardrails only. Do **not** edit `paper/*.tex`.
**Lock (authority on order/content):** `docs/workflows/related_user_lock.md` — four paragraphs; each ends with difference; no conceptual novelty vs Regan / Model-Free / Biyik equivalence.
**Also align:** `paper/sections/introduction.tex`, `AGENTS.md` novelty boundaries, `PROJECT.md` §1.3.
**Bib:** `paper/references.bib` now has every lock mention’s key. Related prose may only `\cite` keys that exist.

---

## Venue / year naming traps

| Lock mention | Bib key | Cite carefully as |
|--------------|---------|-------------------|
| Hejna & Sadigh **2022** (ensembles) | `hejna2023fewshot` | **CoRL 2022** paper; proceedings / bib `year={2023}`. Prose: “Hejna & Sadigh (CoRL 2022)” or “2022/2023 CoRL proceedings.” Do **not** invent a second paper. |
| SPARQ **2026** (lock) | `muraleedharan2025sparq` | **2025 arXiv** (`arXiv:2509.20541`). Prefer “SPARQ (2025 arXiv)” over a fake venue year. |
| Query synthesis **2026** | `zhou2026preferenceekf` | **2026 arXiv** (`arXiv:2609.04066`; PreferenceEKF). No conference claim unless bib gains one. |
| PREC **2026** | `kim2026prec` | **2026 arXiv** (`arXiv:2607.12466`). Same rule. |
| Biyik et al. **2024** (closest PbRL positioning) | `ellis2024equivalence` | First author Ellis; Bıyık last author. Lock label “Biyik et al. 2024” is fine in prose; cite key is `ellis2024equivalence` (arXiv:2403.06003). |

SPARQ / Zhou / PREC are **2025–2026 arXiv** until a venue field is filled — do not invent CoRL/ICML/etc.

---

## Cite-key map (every locked mention → bib key)

### ¶1 Decision-relevant elicitation (FIRST)

| Lock mention | Bib key | Role |
|--------------|---------|------|
| Regan & Boutilier 2009 | `regan2009regret` | Minimax-regret reward elicitation in MDPs; **must lead Related** |
| Alizadeh et al. | `alizadeh2015approximate` | Approximate regret-based elicitation in MDPs |
| Model-Free Preference Elicitation (IJCAI 2024) | `modelfree2024` | EVOI for recommendations / equivalence; no MDP policy |

**Locked difference (end ¶1):** comparison queries rather than bound queries; learned population prior rather than all of reward space; robotics setting with trajectories.

Optional background only (not in lock): `christiano2017deep`.

---

### ¶2 Active preference-based reward learning

| Lock mention | Bib key | Role |
|--------------|---------|------|
| Sadigh 2017 | `sadigh2017active` | Volume removal; active robot PbRL |
| Biyik & Sadigh 2018 | `biyik2018batch` | Batch volume removal |
| Biyik 2019 (IG → easy questions) | `biyik2019asking` | Info gain; human-answerable queries; still parameter target |
| Hejna & Sadigh 2022 (ensembles) | `hejna2023fewshot` | Ensemble disagreement; see venue trap above |
| 2026 query synthesis | `zhou2026preferenceekf` | Continuous query synthesis (PreferenceEKF) |
| SPARQ 2026 | `muraleedharan2025sparq` | When/how to ask under effort; arXiv year trap |
| **Biyik et al. 2024** (own positioning sentence) | `ellis2024equivalence` | Closest PbRL neighbor: reward **equivalence classes**, **single user** |
| EPIG (AISTATS 2023) | `epig2023` | General AL: parameter information is the wrong target (**predictive** MI) |

Optional (intro/acquisition vocabulary; not a lock bullet): `houlsby2011bald` (BALD); `lee2021pebble` (feedback-efficiency, not acquisition cousin).

**Locked difference for ¶2 cluster:** keep segment comparisons, but score by expected **policy regret** under a **population prior**—not parameter IG / volume / equivalence for one user.  
**Biyik 2024 sentence (mandatory):** their target = reward equivalence classes for a **single user**; ours = regret of the **executed policy** under a **population prior**.  
**EPIG:** cite as general AL critique of parameter targets; **not** Regan-style regret / policy VOI.

---

### ¶3 Population and pluralistic preference models

| Lock mention | Bib key | Role |
|--------------|---------|------|
| MaxMin-RLHF (Thm 1: average user is nobody) | `chakraborty2024maxmin` | **Their** Theorem 1 — cite, do not own |
| DPL | `siththaranjan2024dpl` | Distributional prefs / hidden context |
| PAL | `chen2025pal` | Pluralistic / prototype mixture personalization |
| VPL (closest architecture) | `poddar2024vpl` | Variational user latent; **fixed survey queries**; names active selection as future work |
| PREC 2026 | `kim2026prec` | Preference clustering / representative rewards; arXiv |

**Locked difference (end ¶3):** we use the population to *choose queries*, not only to model users.

---

### ¶4 Active personalization elsewhere

| Lock mention | Bib key | Role |
|--------------|---------|------|
| AMPLe (ACL 2025) | `oh2025ample` | Comparison-based active multi-dimensional personalization (NLP) |
| Active utility-based pairwise sampling (2025) | `boroomand2025utility` | Recsys-style active pairwise utility sampling |

**Locked difference (end ¶4):** neither has a downstream **policy** or a **regret** target.

---

## Safe vs forbidden wording

### Safe (say this)

- **Instantiation, not paradigm:** we instantiate Regan’s decision-theoretic goal for preference-based robotics (match intro).
- **¶1 triad difference:** (1) human **trajectory-segment comparisons** / BT, not bound queries; (2) **learned population prior** (particles over previous users), not unrestricted reward space; (3) robotics / trajectory setting.
- **¶2:** acquisition lineage (volume → batch → IG/easy → ensembles → synthesis / SPARQ) still targets parameters or single-user objectives; EPIG = predictive MI critique.
- **¶2 Biyik 2024:** equivalence-class acquisition for one user ≠ policy-regret under population prior.
- **¶3:** pluralism models users / clusters / latents; VPL = closest encoder architecture + fixed queries; we use population for **query selection**.
- **¶4:** active personalization without MDP policy / regret.
- MaxMin: “as shown by MaxMin-RLHF (Theorem 1)…” — attribute clearly.
- EPIG: “parameter information can be the wrong target” (predictive MI); do not call it decision-relevant elicitation.

### Forbidden (do not say this)

| Forbidden | Why |
|-----------|-----|
| Conceptual novelty vs Regan / Model-Free / Biyik equivalence | Explicit lock + `AGENTS.md` / `PROJECT.md` §1.3 |
| “New elicitation paradigm” | Intro: instantiate only |
| Owning **MaxMin Theorem 1** (or Regan minimax proofs) | Cite as theirs; never “we prove” / “our Thm 1” |
| Calling EPIG regret-based, decision-relevant, or policy VOI | EPIG = expected **predictive** information gain |
| Claiming VPL does active query selection | Lock: fixed survey queries; active selection = their future work |
| Empiric numbers, CIs, seeds, bake-offs in Related | Section hygiene / I8 |
| “BALD worse than random” as our established result | Retracted (`PROJECT.md` §5.4); EPIG may motivate critique only |
| Fake venues for SPARQ / Zhou / PREC | Cite as 2025–2026 arXiv |
| Inventing bib keys | Only keys in `references.bib` |

### Differentiation one-liners (match lock endings)

- **¶1:** Unlike bound-query / full-space regret elicitation and recsys EVOI, we use **comparison queries**, a **learned population prior**, and a **robotics trajectory** setting.
- **¶2:** Unlike parameter IG / volume / single-user equivalence acquisition (and unlike EPIG’s predictive MI), we target **executed-policy regret** under a **population prior**.
- **¶3:** Unlike pluralism models that mainly **represent** users, we use the population to **choose queries**.
- **¶4:** Unlike AMPLe / utility pairwise active sampling, we have a downstream **policy** and a **regret** target.

---

## Theorem / novelty ownership checklist

1. Decision-relevant / regret elicitation concept → **Regan & Boutilier** (`regan2009regret`), not us.
2. Recommendation-quality / model-free EVOI → **`modelfree2024`**, not us.
3. Reward equivalence-class acquisition (PbRL) → **`ellis2024equivalence`**, not us; differentiate single-user vs population policy regret.
4. MaxMin-RLHF **Theorem 1** → **`chakraborty2024maxmin`** only.
5. Our opening (safe claim): population prior + human trajectory comparisons + (policy) regret scoring as **instantiation**.

---

## Writer handoff

1. Follow lock paragraph order exactly (¶1 → ¶4); do not revive the old Theme 1–4 order from prior drafts.
2. End each paragraph with the locked difference.
3. One dedicated Biyik-2024 / `ellis2024equivalence` positioning sentence in ¶2.
4. EPIG only in ¶2; predictive-MI wording only.
5. Hejna = CoRL 2022 / bib year 2023; SPARQ/PREC/Zhou = arXiv 2025–2026.
6. No numbers; no theorem ownership; no `paper/*.tex` edits from Engineer.

See also: `docs/workflows/related_researcher.md`.
