# Related Work — Tester gate

**Role:** Tester  
**Draft:** `paper/sections/related.tex` (not edited)  
**Authority:** `docs/workflows/related_user_lock.md`  
**Cross-check:** `related_engineer.md`, `related_researcher.md`, `related_writer_notes.md`, `introduction.tex`, `references.bib`  
**Date:** 2026-09-17

---

## Section decision

**OK_FOR_PAPER**

---

## Must-fixes

*(none)*

---

## Exact lock compliance

| # | Check | Result | Evidence |
|---|--------|--------|----------|
| 1 | Four themes in lock order; ¶1 decision-relevant **FIRST** | **PASS** | Subsections: (1) Decision-relevant elicitation → (2) Active preference-based reward learning → (3) Population and pluralistic preference models → (4) Active personalization elsewhere. Opens with Regan, not Sadigh. |
| 2 | Each theme ends with locked difference | **PASS** | See unit closes below. |
| 3 | Biyik/Ellis 2024 dedicated positioning sentence | **PASS** | Own sentence on `ellis2024equivalence`: equivalence classes / **single user** vs regret of **executed policy** / **population prior**. |
| 4 | EPIG in ¶2 as predictive / wrong-target for params | **PASS** | In ¶2 only; “parameter information can be the wrong target”; “expected *predictive* mutual information”; explicit “not policy regret or Regan-style VOI.” |
| 5 | MaxMin Thm 1 attributed, not claimed | **PASS** | “As shown by MaxMin-RLHF (Theorem~1)” + `chakraborty2024maxmin`; no “we prove.” |
| 6 | All cite keys in `references.bib` | **PASS** | 18 keys; missing = none (script check). |
| 7 | No empiric numbers; novelty hygiene | **PASS** | No CIs/seeds/bake-offs; “goal itself is not novel”; ¶4 close = instantiation, not new paradigm; VPL = fixed surveys + active selection as future work. |

---

## Unit closes (locked differences)

| ¶ | Required close | Draft close | Verdict |
|---|----------------|-------------|---------|
| 1 | Comparison queries ≠ bound; learned population prior ≠ full reward space; robotics / trajectories | Ends with BT segment comparisons vs bound queries; learned population prior vs unrestricted reward space; robotics trajectories | **PASS** |
| 2 | Policy regret under population prior ≠ parameter volume/IG / single-user equivalence / EPIG predictive MI | Ends with segment interface kept; score by expected policy-regret reduction under population prior vs volume, IG, single-user equivalence, predictive MI | **PASS** |
| 3 | Population to *choose queries*, not only model users | “We use the population to *choose queries*, not only to model users.” | **PASS** |
| 4 | Neither has downstream policy or regret target | “Neither setting has a downstream policy or a regret target.” (+ optional instantiation Positioning allowed by researcher) | **PASS** |

---

## Cite-key audit (Related only)

`regan2009regret`, `alizadeh2015approximate`, `modelfree2024`, `sadigh2017active`, `biyik2018batch`, `biyik2019asking`, `hejna2023fewshot`, `zhou2026preferenceekf`, `muraleedharan2025sparq`, `ellis2024equivalence`, `epig2023`, `chakraborty2024maxmin`, `siththaranjan2024dpl`, `chen2025pal`, `kim2026prec`, `poddar2024vpl`, `oh2025ample`, `boroomand2025utility`

All present in `paper/references.bib`. No invented keys. Venue traps avoided (Hejna / SPARQ / Zhou / PREC: cite-only, no fake conferences).

---

## Intro / novelty sync

- Intro already leads decision-quality prior art (Model-Free + Regan) and parameter practice (Sadigh/Bıyık); Related ¶1 deepens that cluster first — aligned.
- Instantiation triad matches intro: comparisons + population prior + policy-regret scoring.
- No conceptual novelty vs Regan / Model-Free / Biyik equivalence; MaxMin and EPIG not owned.

---

## Optional nits (not must-fixes)

- Themes use `\subsection` titles rather than bare paragraphs; lock content/order still exact.
- Within ¶3, PREC appears before the VPL “closest architecture” beat; all required cites and the locked close are present.
