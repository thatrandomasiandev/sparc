# Related Work — Researcher outline

**Scope:** Paragraph-by-paragraph Related Work outline only. No `paper/*.tex` edits.
**Lock (exact structure):** `docs/workflows/related_user_lock.md` — four paragraphs; order fixed; each ends with how this work differs.
**Consistency:** `paper/sections/introduction.tex`; novelty in `AGENTS.md` / `PROJECT.md` §1.3.
**Keys:** `docs/workflows/related_engineer.md` + `paper/references.bib`.
**Hygiene:** No conceptual novelty vs Regan / Model-Free / Biyik equivalence. **No empiric numbers.** Instantiation only.

---

## Global constraints

- **Paragraph order = lock order.** Do not reorder to “Active PbRL first.”
- **Every paragraph ends with a difference sentence** (lock wording or faithful paraphrase).
- **No bake-off numbers, CIs, seeds, or “VOI beats BALD” claims** in Related.
- **Do not own theorems:** MaxMin Theorem 1 is theirs (`chakraborty2024maxmin`); Regan minimax theory is theirs.
- Differentiation vocabulary (use consistently): comparison / segment queries; population (particle) prior; policy regret; robotics trajectories — **not** “new elicitation paradigm.”

---

## ¶1 — Decision-relevant elicitation (FIRST)

**Cluster:** Elicitation aimed at near-optimal **decisions / policies**, not fine reward identification. Lead Related with this cluster (matches intro + lock).

**Core papers / ideas:**
- Regan & Boutilier 2009 (`regan2009regret`) — minimax-regret reward elicitation for MDPs; bound-style queries; search over reward uncertainty relevant to near-optimal policies.
- Alizadeh et al. (`alizadeh2015approximate`) — approximate regret-based elicitation continuing the MDP line.
- Model-Free Preference Elicitation, IJCAI 2024 (`modelfree2024`) — EVOI toward recommendation quality / reward equivalence; recommenders; **no MDP policy**.

**Shared assumption:** Only distinctions that change decisions (or recommendations) are worth eliciting.

**What is *not* novel for us:** The decision-relevant / regret-based **goal**; Model-Free’s move off raw parameters toward decision-quality VOI.

**End-of-¶ difference (lock):** We use **comparison queries** rather than bound queries; a **learned population prior** rather than all of reward space; a **robotics** setting with **trajectories**.

**Writer beat (suggested):** Open with Regan goal → Alizadeh as approximate neighbor → Model-Free as modern non-MDP EVOI → close with the three-way instantiation difference. No numbers.

---

## ¶2 — Active preference-based reward learning

**Cluster:** Robotics / PbRL acquisition that chooses which comparison to ask next; plus the general AL critique that parameter information is the wrong target.

**Core papers / ideas (lineage order):**
- Sadigh et al. 2017 (`sadigh2017active`) — volume removal over reward weights.
- Bıyık & Sadigh 2018 (`biyik2018batch`) — batch volume removal for live interaction.
- Bıyık et al. 2019 (`biyik2019asking`) — information gain → “easy” / human-answerable questions; still parameter uncertainty.
- Hejna & Sadigh 2022 (`hejna2023fewshot`; CoRL 2022 / proceedings year 2023) — ensembles / disagreement queries (and few-shot prior across tasks).
- Query synthesis 2026 (`zhou2026preferenceekf`; 2026 arXiv) — synthesize queries in continuous space rather than only scoring a pool.
- SPARQ (`muraleedharan2025sparq`; 2025 arXiv) — when / how to ask under human effort; still not population policy-regret VOI.

**Mandatory positioning sentence — Biyik et al. 2024 (`ellis2024equivalence`):**
Give this paper **its own sentence** (closest PbRL neighbor). Content of the sentence:
- **Their target:** reward **equivalence classes** for a **single user**.
- **Ours:** **regret of the executed policy** under a **population prior**.
Do not fold them into a generic “IG lineage” cite-only.

**EPIG (`epig2023`, AISTATS 2023) — in ¶2 only:**
- Role: general active-learning version of “parameter information is the wrong target.”
- Correct framing: **predictive** mutual information (query label ↔ future predictions).
- Incorrect framing: decision-relevant elicitation, policy regret, or Regan-style VOI.
- Use as intellectual neighbor to doubting BALD / parameter MI — not as our criterion.

**Shared assumption of the PbRL cluster:** Better parameter estimates (volume / entropy / IG / disagreement) → better behavior; usually **single-user**; pool or synthesized queries.

**End-of-¶ difference:** Same segment-comparison interface as active PbRL, but the objective is **expected reduction in policy regret** under a **population prior**—not parameter IG/volume, not single-user equivalence acquisition, and not EPIG’s predictive MI.

**Writer beat (suggested):** Short lineage (Sadigh → batch → easy IG → ensembles → synthesis/SPARQ) → **dedicated Biyik-2024 sentence** → EPIG as general AL parallel → difference close. No empirics; no “BALD worse than random” as our result.

---

## ¶3 — Population and pluralistic preference models

**Cluster:** Modeling preference diversity / multiple users as structure, not noise.

**Core papers / ideas:**
- MaxMin-RLHF (`chakraborty2024maxmin`) — **their** Theorem 1: a single average reward / “average user” fails under diversity (“average user is nobody”). Cite; **do not claim as ours**.
- DPL (`siththaranjan2024dpl`) — distributional utilities / hidden context.
- PAL (`chen2025pal`) — pluralistic / prototype-based personalized reward modeling.
- VPL (`poddar2024vpl`) — **closest architecture**: variational encoder from annotations to a user latent + latent-conditioned reward; **fixed survey queries**; authors name **active selection as future work**.
- PREC (`kim2026prec`; 2026 arXiv) — clustering / representative rewards from diverse preferences.

**Shared assumption:** Population structure helps **represent** or **align** to diverse users (latents, mixtures, clusters, worst-case gaps).

**End-of-¶ difference (lock):** We use the population to *choose queries*, not only to model users.

**Writer beat (suggested):** Pluralism turn → MaxMin Thm 1 attributed → DPL/PAL briefly → VPL as architectural cousin with fixed queries → PREC as clustering neighbor → difference: population for **acquisition**, not representation alone.

---

## ¶4 — Active personalization elsewhere

**Cluster:** Active preference / utility sampling outside robot policy learning.

**Core papers / ideas:**
- AMPLe, ACL 2025 (`oh2025ample`) — comparison-based active preference learning for multi-dimensional personalization (language / personalization setting).
- Active utility-based pairwise sampling, 2025 (`boroomand2025utility`) — personalized recommendations via active pairwise utility sampling.

**Shared assumption:** Actively choosing pairwise comparisons improves a personal utility / preference model in NLP or recsys settings.

**End-of-¶ difference (lock):** Neither has a downstream **policy** or a **regret** target.

**Writer beat (suggested):** Brief external neighbors → difference close. Keep short; do not overclaim robotics novelty beyond the locked contrast.

---

## Positioning (optional one-liner for Writer, not a fifth paragraph)

If a closing Positioning sentence is needed inside ¶4 or as the final Related sentence, keep it instantiation-only:

> Across decision-relevant elicitation, active PbRL, pluralism, and non-robot personalization, the unoccupied cell we claim is **policy-regret scoring of human trajectory comparisons under a population prior**—an instantiation of known goals, not a new paradigm.

Do **not** expand this into a fifth theme block that reorders the lock.

---

## Novelty hygiene

| Claim type | Status |
|------------|--------|
| Decision-relevant / regret-based elicitation as a concept | **Not novel** — Regan & Boutilier 2009 |
| Model-Free / recommendation-quality EVOI | **Not novel** — `modelfree2024` |
| Reward equivalence-class acquisition (PbRL) | **Not novel** — `ellis2024equivalence`; differentiate single-user vs population policy regret |
| EPIG / predictive MI | **Not ours**; do not rebrand as regret |
| MaxMin Theorem 1 | **Theirs** — cite only |
| Our opening | **Instantiation:** comparison queries + population prior + robotics trajectories / policy regret |
| Empiric bake-off numbers | **Out of Related** entirely |
| Retracted framing | Do not use “BALD worse than random” as settled lore for this paper |

---

## Citation map by paragraph

| ¶ | Primary keys | End-of-¶ differ |
|---|--------------|-----------------|
| 1 Decision-relevant (FIRST) | `regan2009regret`, `alizadeh2015approximate`, `modelfree2024` | Comparisons + population prior + robotics trajectories |
| 2 Active PbRL (+ EPIG + Biyik 2024 sentence) | `sadigh2017active`, `biyik2018batch`, `biyik2019asking`, `hejna2023fewshot`, `zhou2026preferenceekf`, `muraleedharan2025sparq`, **`ellis2024equivalence`**, `epig2023` | Policy regret under population prior ≠ parameter/equivalence/predictive targets |
| 3 Population / pluralism | `chakraborty2024maxmin`, `siththaranjan2024dpl`, `chen2025pal`, `poddar2024vpl`, `kim2026prec` | Population chooses queries, not only models users |
| 4 Active personalization elsewhere | `oh2025ample`, `boroomand2025utility` | No downstream policy; no regret target |

**Intro alignment:** Intro already places Regan + Model-Free as prior art on decision-quality targets and Sadigh/Bıyık as dominant parameter practice. Related deepens in lock order; ¶1 keeps Regan/Model-Free early (not buried).

---

## Writer handoff order

1. ¶1 — Decision-relevant elicitation (Regan → Alizadeh → Model-Free → difference)  
2. ¶2 — Active PbRL lineage → **Biyik 2024 own sentence** → EPIG → difference  
3. ¶3 — MaxMin (their Thm 1) / DPL / PAL / VPL / PREC → difference  
4. ¶4 — AMPLe / utility pairwise → difference  

See also: `docs/workflows/related_engineer.md` (keys, venue traps, safe/forbidden).
