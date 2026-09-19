# Related Work — Writer notes (2026-09-17)

**Deliverable:** Overwrote `paper/sections/related.tex` to match
`docs/workflows/related_user_lock.md` (four paragraphs / short `\subsection`
titles). Sources: lock, `related_engineer.md`, `related_researcher.md`,
`introduction.tex`, `references.bib`.

---

## Structure shipped

| ¶ | Subsection title | Locked difference (close) |
|---|------------------|---------------------------|
| 1 | Decision-relevant elicitation | Comparison queries; learned population prior; robotics trajectories |
| 2 | Active preference-based reward learning | Policy regret under population prior ≠ parameter / single-user equivalence / EPIG predictive MI |
| 3 | Population and pluralistic preference models | Population chooses queries, not only models users |
| 4 | Active personalization elsewhere | No downstream policy; no regret target (+ brief instantiation close) |

Order matches the lock (decision-relevant **first**), not the old Active-PbRL-first draft.

---

## Cite keys used (bib only)

- **¶1:** `regan2009regret`, `alizadeh2015approximate`, `modelfree2024`
- **¶2:** `sadigh2017active`, `biyik2018batch`, `biyik2019asking`, `hejna2023fewshot`, `zhou2026preferenceekf`, `muraleedharan2025sparq`, `ellis2024equivalence`, `epig2023`
- **¶3:** `chakraborty2024maxmin`, `siththaranjan2024dpl`, `chen2025pal`, `kim2026prec`, `poddar2024vpl`
- **¶4:** `oh2025ample`, `boroomand2025utility`

**Not cited in Related (intentional):** `christiano2017deep`, `houlsby2011bald`, `lee2021pebble` — intro / optional; lock does not require them in Related.

---

## Intro sync

- Intro already places Regan + Model-Free as prior art on decision-quality targets and Sadigh/Bıyık as dominant parameter practice; Related ¶1 leads with Regan → Alizadeh → Model-Free and ends with the same three-way instantiation (comparisons, population prior, robotics trajectories).
- Intro “we do not claim a new elicitation paradigm, only this instantiation” echoed in ¶1 (“goal itself is not novel”) and ¶4 close.
- No empiric numbers; no bake-off; BALD appears only in Intro, not Related (EPIG only as predictive-MI critique in ¶2).

---

## Hygiene checklist (passed)

- [x] No conceptual novelty vs Regan / Model-Free / Biyik equivalence
- [x] Dedicated Biyik-2024 / `ellis2024equivalence` sentence (single-user equivalence vs population policy regret)
- [x] MaxMin Theorem 1 attributed to them (`chakraborty2024maxmin`)
- [x] EPIG = predictive MI only; not Regan/policy VOI
- [x] VPL = fixed survey queries; active selection = their future work
- [x] Hejna not given a fake 2022-only venue year in prose (cite only)
- [x] SPARQ / Zhou / PREC not given fake conference venues
- [x] No orphan empiric numbers

---

## Optional follow-ups (not done)

- If Reviewer asks for BALD vocabulary in Related, one sentence pointing to Intro + EPIG is enough; do not revive the old BALD/EPIG subsection.
- Prefer leaving `christiano2017deep` out of Related unless a reviewer wants PbRL grounding beyond the intro.
