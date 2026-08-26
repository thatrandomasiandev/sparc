# Prior-art matrix (Phase 0, Week 1)

Comparison of preference-based RL and related methods against SPARC's target deployment class (delayed, multi-operator, safety-bounded field robot). See `PROBLEM.md` for the locked scenario and `ROADMAP.md` Week 1 reading list.

| Method | Query strategy | Reward model | Annotator model | Tested domains | Reported sample efficiency | Open weaknesses (for SPARC scenario) |
|--------|----------------|--------------|-----------------|----------------|----------------------------|--------------------------------------|
| **Wirth et al. 2017** (survey) | Taxonomy: static vs active queries; trajectory vs state-action pairs | Bradley–Terry / logistic on hand-crafted or learned features | Single implicit human | Gridworlds, inverted pendulum, Atari (early PbRL) | Active querying reduces queries vs random | Assumes synchronous, attentive labeler; no comms delay, multi-operator, or safety gating |
| **Christiano et al. 2017** | Active selection of trajectory pairs by ensemble disagreement | Bradley–Terry on trajectory segments; CNN encoder | Single human rater | Atari, MuJoCo locomotion | ~700–5.5k comparisons for Atari/MuJoCo tasks | On-demand queries; single annotator; no drift detection or policy bounding under uncertainty |
| **PEBBLE** (Lee et al. 2021) | Unsupervised exploration (RUNE-style bonus optional) + relabeling; queries when buffer fills | Ensemble of reward MLPs/CNNs; Bradley–Terry loss | Single synthetic or human labeler | DMControl, MetaWorld, Atari | Matches oracle with ~1k preferences on walker-walk (B-Pref scale) | Single annotator; synchronous query loop; uncertainty used for exploration not hard safety freeze |
| **B-Pref** (Lee et al. 2022) | Benchmark harness; implements PEBBLE, PrefPPO, etc. | Same as constituent methods | Single labeler + scripted irrationality modes | DMControl, MetaWorld | Reproduces PEBBLE ~1000-query budget on walker-walk | Evaluation assumes lab feedback; irrationality ≠ multi-operator rotation or comms batching |
| **SURF** (Park et al. 2022) | Same query schedule as PEBBLE + pseudo-labeling on unlabeled segments | Semi-supervised reward model with data augmentation | Single labeler | DMControl, MetaWorld | Fewer queries than PEBBLE on some tasks via augmentation | Still single-annotator, on-demand; pseudo-labels risk amplifying bias under operator disagreement |
| **RUNE** (Liang et al. 2022) | Exploration driven by reward-model disagreement | PEBBLE-style ensemble | Single labeler | DMControl | Improved exploration efficiency vs PEBBLE alone | Disagreement → explore more, opposite of deploy-time **freeze/shrink**; no batch latency model |
| **APReL** (Bıyık et al. 2020–22) | Active: disagreement, volume removal, mutual information; batch (medoids/greedy) over trajectory sets | Linear / GP reward on trajectory features | Single implicit user in core API | Driving sims, synthetic prefs | Strong sample efficiency in **sequential** active setting | Sequential belief updates assume answers before next query; not designed for hard comms blackout or per-operator latents |
| **VPL** (variational per-annotator) | Batch or pool labeling (NLP/LLM settings) | Shared backbone + per-annotator heads; variational inference | Multiple annotators with latent prefs | Text/LLM alignment, crowd settings | Handles annotator heterogeneity in text | Async cheap crowd labor, not one robot + bandwidth-limited ops team; no policy safety bounding or drift SPRT |
| **Crowd-PrefRL** | Crowd aggregation over many labelers | Distributional / aggregated preference model | Many parallel workers | LLM RLHF-style pipelines | Scales label throughput | Economics and latency model wrong for 3 rotating field operators on one rover |
| **Distributional pref. reward models** | Standard PbRL query loops | Capture label noise as distribution | Often single pooled labeler | Sim locomotion, alignment | Robust to some label noise | Pooling or noise modeling ≠ explicit per-operator adapters + comms-window batching + gating |

## SPARC gap (summary row)

| **SPARC** (proposed) | **Batch EIG** per comms window (\(B=8\), \(\Delta T=900\) s sim); no mid-window re-query | Shared encoder + per-operator adapter + bootstrap ensemble (EM-style \(z_i\)) | \(K\) operators with latent \(z_i\); drift via SPRT | Rover-nav proxy + DMControl + MetaWorld | Target: PEBBLE asymptote at **≤20%** queries under stress | Combines all four: multi-operator latents, latency-aware batching, confidence gating, drift re-query — **not present as one engine in prior art** |

## Sources

- Wirth, Christian, et al. "A survey of preference-based reinforcement learning." *JMLR* 2017.
- Christiano, Paul, et al. "Deep reinforcement learning from human preferences." *NeurIPS* 2017.
- Lee, Kimin, et al. "PEBBLE: Feedback-efficient interactive reinforcement learning via relabeling and unsupervised pre-training." *ICML* 2021.
- Lee, Kimin, et al. "B-Pref: Benchmarking preference-based reinforcement learning." *NeurIPS* 2022.
- Park, Seohong, et al. "SURF: Semi-supervised reward learning with data augmentation for preference-based RL." *ICLR* 2022.
- Liang, Chang, et al. "RUNE: Reward learning with unified exploration." *NeurIPS* 2022.
- Bıyık, Erdem, et al. APReL / active preference-based learning (Stanford-ILIAD).
- Variational and crowd multi-annotator preference work (VPL, Crowd-PrefRL) — see `ROADMAP.md` Week 2 reading list.
