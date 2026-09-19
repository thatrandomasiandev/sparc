# Related Work — user lock (2026-09-17)

Four paragraphs only. Each ends with how this work differs.
No conceptual novelty vs Regan / Model-Free / Biyik equivalence.

## ¶1 Decision-relevant elicitation (FIRST)
- Regan & Boutilier 2009 (minimax-regret reward elicitation in MDPs)
- Alizadeh et al. (approximate regret-based elicitation)
- Model-Free Preference Elicitation (IJCAI 2024, EVOI for recommendations)
- **Difference:** comparison queries rather than bound queries; learned population prior
  rather than all of reward space; robotics setting with trajectories.

## ¶2 Active preference-based reward learning
- Sadigh 2017; Biyik & Sadigh 2018; Biyik 2019 (IG → easy questions)
- Hejna & Sadigh 2022 (ensembles); 2026 query synthesis; SPARQ 2026
- **Biyik et al. 2024** gets its own positioning sentence (closest): their target is
  reward equivalence classes for a single user; ours is regret of the executed policy
  under a population prior.
- EPIG (AISTATS 2023): general AL version of “parameter information is the wrong target.”

## ¶3 Population and pluralistic preference models
- MaxMin-RLHF (Thm 1: average user is nobody); DPL; PAL; VPL (closest architecture;
  fixed survey queries; names active selection as future work); PREC 2026
- **Difference:** we use the population to *choose queries*, not only to model users.

## ¶4 Active personalization elsewhere
- AMPLe (ACL 2025); active utility-based pairwise sampling (2025)
- **Difference:** neither has a downstream policy or a regret target.
