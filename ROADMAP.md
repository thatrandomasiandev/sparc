# SPARC: from problem to release

A 38-week, week-by-week plan to design, build, and validate **SPARC** — a proprietary
preference-learning engine for robots that get sparse, delayed, and inconsistent human
feedback — engineered to the standard set by [`alexdobin/STAR`](https://github.com/alexdobin/STAR),
used here only as the bar for depth and rigor, never as a relevant subject.

> **For Cursor / any AI assistant working in this repo:** this file is the source of truth
> for scope, sequencing, and the four proprietary pillars (SPARC). Do not implement
> anything outside the current phase's tasks without flagging it. Treat `PROBLEM.md`,
> `DESIGN.md`, `EXPERIMENTS.md`, and `RESULTS.md` (created in Phases 0–4 below) as the
> authoritative specs once they exist — this roadmap sequences the work, those files
> define it precisely.

---

## What "STAR-grade" means

Eleven traits extracted from the STAR repository's engineering and governance discipline
— not its RNA-seq subject matter, which is irrelevant here. Every phase below checks
itself against this list.

1. **Algorithm before code.** The two-pass, seed-and-extend splice algorithm was published
   (Dobin et al., 2013) before a decade of engineering followed. Design on paper first.
2. **One file, one concern.** The core read-alignment class alone is split across 20+
   single-purpose files (`ReadAlign_*.cpp`). No god-objects.
3. **Tuned at the metal.** Hand-written AVX2 SIMD paths, OpenMP threading, custom packed
   arrays and suffix-array structures — performance is designed in, not bolted on.
4. **Dependencies vendored, pinned.** `htslib` is bundled in-tree, not left to float on
   whatever version happens to be installed.
5. **One engine, many products.** STARsolo, STARconsensus, STARdiploid — one core aligner,
   several specialized front-ends built on it over time.
6. **A parameter file, not a flag maze.** `parametersDefault` documents every option with
   a sane default — the interface is designed, not accreted.
7. **A real manual.** A standalone typeset manual, not just a README — every flag and
   output file documented with examples.
8. **Governance from day one.** `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, and a disciplined
   bug-report template exist alongside the source, not after the fact.
9. **CI on every change.** Continuous integration has gated changes since the project's
   early history.
10. **Versioned honesty.** `CHANGES.md` logs every behavior change, bug fix, and issue
    number back to 2009 — nothing hidden between releases.
11. **Still the default, 15+ years on.** Funded, cited, and still the field's reference
    tool — the payoff of the ten traits above, compounded.

---

## The real problem

Preference-based RL (PbRL) lets a robot learn a reward function from comparisons
("this behavior is better than that one") instead of a hand-written reward. Every
mainstream method shares three assumptions that hold in a lab and break in the field:

- **One attentive annotator.** PEBBLE, SURF, and RUNE are tuned and evaluated with a
  single, consistent labeler answering queries continuously.
- **Low-latency, on-demand queries.** The active-query optimizers in APReL and B-Pref
  assume a query can be answered within seconds, not batched behind a communications
  delay.
- **Stationary preferences.** None of the above detect or react when what "good" means
  changes mid-deployment — operator fatigue, a rotating crew, a shifted mission priority.
- **Where multi-annotator disagreement *is* addressed** — variational per-annotator
  preference learning (VPL), crowd-sourced preference RL, distributional preference
  reward models — the setting is text/LLM alignment with cheap, parallel, asynchronous
  labor, not a single physically deployed robot with a hard safety envelope and a
  bandwidth-limited human on the other end.

**No existing open tool combines latency-aware batch querying, per-operator preference
modeling, safety-bounded policy updates, and online drift detection in one engine,
validated on physical hardware. That combination is the gap this roadmap closes.**

---

## The proprietary core — SPARC

**SPARC** — Sparse-feedback Preference Active Reward Core. Four pillars, each a
defensible technical contribution on its own, designed specifically for feedback that
is scarce, delayed, and inconsistent.

| Pillar | What it does |
|---|---|
| **1. Per-operator latent reward model** | A shared trajectory encoder feeds per-operator adapter heads under a shared Bayesian prior, so disagreement between labelers is modeled explicitly instead of averaged away. |
| **2. Latency-aware batch query optimizer** | Selects a full batch of queries per communications window by expected information gain under a hard per-window budget — not one-at-a-time selection assuming an always-available labeler. |
| **3. Confidence-gated policy bounding** | Policy updates are clipped or frozen wherever the reward-model ensemble disagrees above a threshold — reward uncertainty becomes a hard safety bound, not just an exploration bonus. |
| **4. Online preference-drift detector** | A sequential test over incoming comparisons flags when new feedback statistically contradicts the current reward posterior, triggering targeted re-querying instead of silent blending. |

---

## The 38-week build

### Phase 0 — Foundation & immersion (Weeks 1–3)
*Objective: understand every existing method's failure mode before writing an algorithm.*
*STAR-bar checkpoint: Dobin designed STAR against the specific failures of every prior aligner. Same discipline, week one.*

**Week 1 — literature base & prior-art map**
- [ ] Read and annotate the foundational PbRL survey (Wirth et al., JMLR 2017), Christiano et al. 2017, PEBBLE, and B-Pref
- [ ] Build a prior-art matrix: method → query strategy → reward model → annotator model → tested domains → reported sample efficiency → open weaknesses
- [ ] Install APReL (Stanford-ILIAD) and run its bundled example end-to-end; log every friction point
- [ ] Cursor role: summarize each paper into the matrix row format — not writing code yet

**Week 2 — field deployment survey**
- [ ] Read SURF, RUNE, VPL, Crowd-PrefRL, and distributional preference reward modeling
- [ ] Write a two-page memo: where every existing method assumes abundant, synchronous, single-source feedback — and why that fails for a physically deployed robot
- [ ] Shortlist 2–3 candidate real-world scenarios (delayed-comms rover ops, single-caregiver assistive robot, rotating-operator field robot) and rank by evidence you can generate this year

**Week 3 — scope lock & success criteria**
- [ ] Pick one deployment scenario as the primary use case (recommended: delayed/multi-operator feedback for a semi-autonomous mobile robot, reusing your URC rover platform and NASA L'SPACE ops-team mental model)
- [ ] Write a one-page problem statement with a falsifiable success criterion (e.g., match PEBBLE's asymptotic quality with ≤20% of its queries under simulated comms delay and 3 disagreeing synthetic annotators)
- [ ] Stand up the repo skeleton: license, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates — governance from day one
- [ ] **Deliverable:** `PROBLEM.md`

---

### Phase 1 — Problem definition & proprietary design (Weeks 4–7)
*Objective: design the four SPARC pillars on paper, get them red-teamed, and lock the experimental protocol before any implementation.*
*STAR-bar checkpoint: the seed-and-extend splice algorithm existed as a design before it existed as code.*

**Week 4 — formalize the feedback model**
- [ ] Define the preference query over trajectory-segment pairs, the per-operator latent preference vector, and a Bradley-Terry-with-drift likelihood
- [ ] Formalize the "communication window" as a hard batching constraint on the query optimizer
- [ ] Draft the reward-model architecture: shared trajectory encoder + per-operator adapter heads

**Week 5 — design the four SPARC pillars**
- [ ] Pillar 1 — per-operator latent embeddings under a shared-prior Bayesian ensemble reward model
- [ ] Pillar 2 — latency-aware batch active query optimizer under a per-window budget
- [ ] Pillar 3 — confidence-gated policy bounding from ensemble disagreement
- [ ] Pillar 4 — online sequential drift detector triggering targeted re-querying
- [ ] **Deliverable:** `DESIGN.md` (pseudocode only, no code)

**Week 6 — adversarial self-review**
- [ ] Red-team each pillar: drift-detector false-positive rate under normal noise; over-conservative gating killing learning progress
- [ ] Send `DESIGN.md` to Prof. Erdem Bıyık's LIRA lab and/or Prof. Daniel Seita's SLURM lab for a sanity-check conversation
- [ ] Revise and version `DESIGN.md` — start the changelog discipline now, not at release

**Week 7 — experimental protocol**
- [ ] Choose simulation environments: MetaWorld/DMControl for general validation, a rover-navigation env for the target scenario
- [ ] Define synthetic annotator models: configurable noise, per-operator bias, and scripted mid-training drift
- [ ] Lock the exact baselines to reproduce: PEBBLE, SURF, RUNE, vanilla PrefPPO, APReL's batch active learner
- [ ] Decide every metric, seed count, and statistical test before a single training run
- [ ] **Deliverable:** `EXPERIMENTS.md`

---

### Phase 2 — Core engineering buildout (Weeks 8–15)
*Objective: build the STAR-grade skeleton — one file per concern, dependencies pinned, tests written the moment code is, one engine behind one entry point.*
*STAR-bar checkpoint: STAR splits one read-alignment concern across 20+ files and vendors its dependencies in-tree.*

**Week 8 — repo architecture**
- [ ] Lay out `env/ reward_model/ query_optimizer/ policy/ annotator_sim/ drift_detector/ cli/` as separate, single-purpose packages
- [ ] Pin exact dependency versions (torch, gymnasium, stable-baselines3) — the Python equivalent of vendoring htslib
- [ ] Configure pre-commit (black, ruff, mypy); Cursor drafts the config, you review every hook

**Week 9 — environment & data layer**
- [ ] Implement the synthetic annotator: configurable Bradley-Terry noise, bias, and scripted drift
- [ ] Implement the trajectory-segment buffer and pairwise query dataset class
- [ ] Write unit tests the moment each class lands — not batched at the end

**Week 10 — reward model (Pillar 1)**
- [ ] Implement the shared trajectory encoder and per-operator adapter heads with a shared Gaussian prior
- [ ] Implement the bootstrap ensemble (5–7 heads) for uncertainty estimates
- [ ] Add a golden-output regression test: fixed seed, fixed data, exact reward output checked into the suite

**Week 11 — query optimizer (Pillar 2)**
- [ ] Implement expected-information-gain scoring over candidate segment pairs
- [ ] Implement greedy submodular batch selection under a hard per-window budget; compare against APReL's disagreement/volume-removal heuristics
- [ ] Benchmark optimizer wall-clock time — it must select a batch inside one simulated comms window

**Week 12 — confidence-gated bounding (Pillar 3)**
- [ ] Wrap SAC/PPO policy updates with an ensemble-disagreement gate on the reward signal
- [ ] Implement freeze / shrink / pass-through bounding modes as a configured parameter, in the spirit of STAR's `parametersDefault`
- [ ] Test on toy MDPs where the analytically correct bounding behavior is known

**Week 13 — drift detector (Pillar 4)**
- [ ] Implement a sequential probability ratio test over incoming preferences vs. the current posterior
- [ ] Wire drift detection to trigger a targeted re-query request to the optimizer
- [ ] Stress-test against the scripted-drift annotator from Week 9

**Week 14 — integration & CLI**
- [ ] Wire all four pillars behind one entry point, `sparc train --config ...` — one engine, invoked one way
- [ ] Add structured JSON-lines logging of every query, reward update, and gate/drift event — your `Log.out` equivalent
- [ ] Write the first end-to-end smoke test: full pipeline on a toy env in under two minutes

**Week 15 — hardening pass**
- [ ] Delete Cursor-generated dead code and resolve every open TODO and exception path
- [ ] Add type hints and docstrings across the core — resist feature-first temptation; stability comes first
- [ ] Tag `v0.1.0-alpha` and start `CHANGES.md` today

---

### Phase 3 — Baseline reproduction & benchmark harness (Weeks 16–19)
*Objective: earn the right to compare by reproducing every baseline under one shared, honest evaluation harness first.*
*STAR-bar checkpoint: STAR earned trust by being benchmarked against every existing aligner on the same data. No credit for beating a strawman.*

**Week 16 — reproduce PEBBLE & PrefPPO**
- [ ] Adapt open-source PEBBLE/PrefPPO baselines inside your harness with identical env wrappers and seeds
- [ ] Confirm published sample-efficiency numbers reproduce within reasonable variance before trusting later comparisons

**Week 17 — reproduce SURF, RUNE, APReL**
- [ ] Apply the same reproduction discipline to SURF, RUNE, and APReL's active learner
- [ ] Build one shared evaluation harness: identical seeds, protocol, and plotting code for every method

**Week 18 — metrics & statistical rigor**
- [ ] Implement policy return vs. queries used, reward-model accuracy vs. ground truth, regret under drift, wall-clock per query
- [ ] Implement multi-seed statistical testing with confidence intervals — no single-run cherry-picking

**Week 19 — dry run**
- [ ] Run the full harness end-to-end on one environment with every method including SPARC at small scale
- [ ] Treat this as a rehearsal for finding harness bugs — not a result to report

---

### Phase 4 — Proprietary validation in simulation (Weeks 20–26)
*Objective: prove each pillar earns its keep, in the easy regime, the stress regime, and in isolation.*
*STAR-bar checkpoint: STAR's original paper reported results across the full ENCODE benchmark suite, not one dataset.*

**Weeks 20–21 — core benchmark suite**
- [ ] Run SPARC vs. all baselines on 3–4 MetaWorld/DMControl tasks under the standard, non-adversarial synthetic annotator
- [ ] Confirm no performance was traded away in the "easy" regime to win the "hard" regime

**Weeks 22–23 — stress regime: sparse + delayed feedback**
- [ ] Re-run everything under the comms-delay, small-query-budget regime
- [ ] Confirm Pillar 2 shows a clear query-efficiency advantage here specifically

**Week 24 — stress regime: multi-operator + drift**
- [ ] Re-run with multiple disagreeing synthetic annotators and scripted mid-training drift
- [ ] Confirm Pillars 1, 3, and 4 show a clear robustness advantage over VPL/crowd-preference-style baselines

**Week 25 — ablations**
- [ ] Ablate each pillar independently (SPARC minus Pillar 1, minus Pillar 2, etc.)
- [ ] Confirm every pillar is pulling measurable weight — this is what reviewers check first

**Week 26 — result write-up**
- [ ] Produce tables and plots — clarity over flashiness, every figure regenerable from a script
- [ ] Write the interpretation honestly, including where SPARC loses
- [ ] **Deliverable:** `RESULTS.md`, backed by regenerable scripts

---

### Phase 5 — Physical validation (Weeks 27–31)
*Objective: real annotators, real hardware, real inconsistency — the part synthetic annotators can't simulate.*
*STAR-bar checkpoint: STAR wasn't validated on synthetic reads alone — a decade of real biological data and bug reports shaped it.*

**Week 27 — testbed selection & instrumentation**
- [ ] Port the SPARC reward model and query optimizer onto the URC rover platform (tabletop robot-arm fallback if rover access is constrained) for one concrete behavior-shaping task
- [ ] Instrument the platform to log trajectory segments in SPARC's expected format

**Week 28 — human-in-the-loop pilot**
- [ ] Recruit 3–5 real annotators (labmates, teammates) under an artificially constrained query schedule mirroring rover-ops delay
- [ ] Watch for the first real (not synthetic) inconsistency to hit the system — expect Pillar 4 to fire

**Week 29 — iterate on real feedback**
- [ ] Fix what breaks: skipped queries, ties, fatigue-driven inconsistency the synthetic model didn't anticipate
- [ ] Run this as a real bug-report-to-patch loop, the same discipline STAR's issue tracker enforces

**Week 30 — safety & bounding validation**
- [ ] Verify Pillar 3: the robot never executes a high-uncertainty action outside the frozen envelope
- [ ] Log and document every gating event with before/after trajectories

**Week 31 — physical validation write-up**
- [ ] Write a short report on what changed between synthetic and real deployment, and why
- [ ] **Deliverable:** hardware validation report

---

### Phase 6 — Hardening, documentation, governance, release (Weeks 32–35)
*Objective: the "boring" layer that makes a project look maintained instead of abandoned.*
*STAR-bar checkpoint: a 60+ page manual, a real license, a decade of versioned release notes — this layer is why STAR is still the default in 2026.*

**Week 32 — the manual**
- [ ] Write a full manual: every CLI flag, config option, and output format, documented with defaults and examples

**Week 33 — test coverage & CI**
- [ ] Push unit and integration coverage over the core package to a defensible level
- [ ] Wire CI to run tests and lint on every pull request

**Week 34 — governance layer**
- [ ] Finalize `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and PR templates
- [ ] Write `CHANGES.md` entries for every version tag so far

**Week 35 — v1.0 release**
- [ ] Tag `v1.0.0`, write release notes, cut a short demo video of SPARC on the rover
- [ ] Publish the repo publicly and submit to relevant preference-RL trackers/lists

---

### Phase 7 — Dissemination (Weeks 36–38)
*Objective: turn the finished engine into a credible research artifact and a portfolio piece.*
*STAR-bar checkpoint: STAR is a paper as much as it is code — the tool and the paper made each other credible.*

**Week 36 — write it up**
- [ ] Draft a workshop-length paper from `RESULTS.md` and the physical-validation report

**Week 37 — external validation**
- [ ] Send the draft and repo to Prof. Erdem Bıyık and/or Prof. Daniel Seita for feedback
- [ ] Submit to a workshop track if timing allows

**Week 38 — portfolio close-out**
- [ ] Record a three-minute demo video
- [ ] Write the README the way STAR's reads — funding, support, manual, and limitations sections, no overclaiming
- [ ] Fold the finished project into your resume and NASA L'SPACE / SpAIder narratives where genuinely relevant

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Scope creep into a general framework instead of one solved problem | High | Scope locked in `PROBLEM.md`, Week 3; anything outside `EXPERIMENTS.md` waits until after v1.0 |
| Cursor-generated code passes review by volume, not correctness | High | Every PR gets a human diff read before merge; unit tests are never optional |
| Physical testbed access falls through | Medium | Tabletop robot-arm fallback identified at Week 27; the simulation result set stands alone if needed |
| Reproduced baselines don't match published numbers | Medium | Weeks 16–17 are budgeted specifically for reproduction before any comparison is trusted |
| Too few real annotators available for Phase 5 | Medium | 3-person minimum; document the limitation honestly rather than skip validation |
| Overstating what SPARC solves | Medium | `RESULTS.md` documents losses as rigorously as wins, matching STAR's own limitations section |

---

## Sources consulted

1. [alexdobin/STAR — repository](https://github.com/alexdobin/STAR)
2. Wirth et al., ["A Survey of Preference-Based Reinforcement Learning Methods,"](https://jmlr.org/papers/v18/16-634.html) JMLR 2017
3. Lee et al., ["B-Pref: Benchmarking Preference-Based Reinforcement Learning,"](https://ar5iv.labs.arxiv.org/html/2111.03026) 2021
4. Liang et al., ["Reward Uncertainty for Exploration"](https://arxiv.org/pdf/2205.12401) (RUNE), ICLR 2022
5. Bıyık et al., ["APReL: A Library for Active Preference-based Reward Learning Algorithms,"](https://arxiv.org/abs/2108.07259) 2021
6. [Stanford-ILIAD/APReL — repository](https://github.com/Stanford-ILIAD/APReL)
7. ["Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning"](https://weirdlabuw.github.io/vpl/) (VPL)
8. ["Crowd-PrefRL: Preference-Based Reward Learning from Crowds,"](https://arxiv.org/html/2401.10941) 2024
9. ["Aligning Crowd Feedback via Distributional Preference Reward Modelling,"](https://arxiv.org/html/2402.09764v3) 2024
10. [USC LIRA Lab (Prof. Erdem Bıyık)](https://liralab.usc.edu/)
