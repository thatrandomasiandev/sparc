| Method / Paper | Year | Setting | Core idea | Assumptions | Domains | Gains | Weaknesses | Notes |
|----------------|------|---------|-----------|-------------|---------|-------|------------|-------|
| Regan & Boutilier | 2009 | Tabular MDP | Minimax regret query selection | Bound queries on reward | Small MDPs | Decision-relevant | No human comparisons; no population prior | Cite ¶1 |
| Sadigh et al. | 2017 | Robot PbRL | Volume removal | Single user | Physical | First active robot PbRL | Trivial query optimum | |
| Bıyık et al. | 2019 | Robot PbRL | Info gain / easy questions | Single user | Physical | Human-answerable | Parameter entropy | |
| Houlsby BALD | 2011 | Pref. learning | Mutual info w/ params | | | | Decouples from decisions | EPIG critique |
| EPIG | 2023 | Active learning | Predictive MI | | | BALD can lose to random | | Cite as foundation |
| PEBBLE | 2021 | Deep PbRL | Relabel + unsupervised pretrain | Single attentive labeler | DMControl | Sample efficiency | Sync queries | Baseline lineage |
| VPL | 2024 | Personalized prefs | Latent z + fixed survey queries | Fixed queries | | Few-shot adapt | Authors: relax fixed queries | Closest arch. |
| Model-Free Pref. Elicitation | 2024 | Recommenders | VOI for recommendation quality | No policy/MDP | Recsys | Decision quality | Not robotics | Cite ¶1 |
| MaxMin-RLHF | 2024 | LLM | Alignment gap vs diversity | | | Theorem 1 | | Cite intro pluralism |
