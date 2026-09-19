# Agent instructions

Short operational contract for coding agents. Full scientific context and results live in
[`PROJECT.md`](PROJECT.md). **If the two disagree: this file wins on invariants; PROJECT.md
wins on science and reported numbers.**

**Optimize for shipping** an honest modest RSS 2027 paper — not an unfinished ambitious one.

---

## Repo map

```
PROJECT.md          Scientific context, results, queue, novelty boundaries
AGENTS.md           This file (invariants incl. I8 claim↔experiment)
docs/CLAIMS.md      Every paper claim → experiment ID (authoritative matrix)
EXPERIMENTS.md      Short status board into CLAIMS.md
plr/
  algorithm.py    POP-VOI — paper algorithm (select / update / decide)
  likelihood.py   Bradley-Terry / Plackett-Luce; invariant I1 (unit-norm w)
  users.py          Simulated populations with ground-truth (w, b)
  encoder.py        Factored variational encoder q(w, b | answers)
  acquisition.py    BALD, EPIG, VOI, effort normalization, QueryCosts
  mdp.py            Gridworld; exact VI; policy_value_matrix regret trick
tests/test_core.py  Property tests for PAPER CLAIMS (not smoke)
experiment_1.py     Factored latent vs VPL-style baseline (NEGATIVE so far)
experiment_2.py     Consistency recovery vs query difficulty
experiment_3.py     random vs BALD vs VOI vs oracle (PRELIMINARY POSITIVE)
aggregate_exp3.py   Paired bootstrap aggregation for exp 3
plr/hardware/       Segment library + human session logger (optional; see docs/hardware.md)
docs/hardware.md    Robot / human-study plan
paper/              LaTeX manuscript
experiments/runs/   Logged run artifacts
```

Deps: **PyTorch, NumPy, pytest only** for core. Hardware drivers stay optional and out of
core install.

---

## Invariants (breaking silently invalidates results)

| ID | Rule |
|----|------|
| **I1** | Always normalize `w` before rewards. Use `likelihood.unit` / `likelihood.rewards`. Never `w @ phi`. |
| **I2** | Difficulty is `reward_gap` over options — **no dependence on `b`**. |
| **I3** | `QueryCosts` raises `KeyError` on missing modality. Never default a cost. |
| **I4** | Encoder per-annotation features **must** include option spread (+ option count). |
| **I5** | Context set ⊥ held-out target set. Always. |
| **I6** | Every paper number regenerates from one integer `--seed`. |
| **I7** | No claim from a single seed. ≥3 seeds exploratory; paired bootstrap for paper claims. |
| **I8** | **Claim↔experiment.** Every paper number/empirical sentence has a Claim ID in `docs/CLAIMS.md`, a reproducing experiment, a run under `experiments/runs/`, and a `RESULTS.md` entry. No orphan numbers. Be creative and thorough; prefer an extra diagnostic over an unsupported claim. |

**Standing order (2026-09-17):** formulate rich experiments; back the whole paper. See `docs/CLAIMS.md`.

---

## Statistics (non-negotiable)

- Strategies share users and candidates → **paired** comparisons.
- Paired bootstrap CI with **10,000** resamples (not unpaired SEs).
- Report **tie rate** and **censoring rate** (queries-to-threshold uses `n_queries+1`).
- **Pre-specify primary metric before the next run.** Current recommendation in PROJECT.md §5.3:
  prefer **final-query regret** as primary; queries-to-threshold secondary (fragile CI).
- Marginal → say "preliminary" and add seeds. Do not headline a +0.02 CI lower bound.

---

## Novelty boundaries — do not overclaim

- **Not novel:** decision-relevant / regret-based elicitation (Regan & Boutilier 2009).
- **Our opening:** population prior + human trajectory comparisons + (eventually) continuous
  control — cite Regan & Boutilier and Model-Free Pref. Elicitation (IJCAI 2024) in **¶1**.
- Retracted: "BALD worse than random"; "factored latent beats VPL". See PROJECT.md §5.4.

---

## Code standards

- One module, one job; docstring explains **why** (failure mode prevented).
- Shape comments on every tensor op: `# (B, N, K, d)`.
- Validate at boundaries (non-positive `b`, shape mismatches) — raise, don't silently coerce.
- Tests assert **claims** (`test_trivial_query_has_zero_information`, BT≡PL at K=2, …).
- Long runs: background + log file (~25 min for powered d=8).

---

## Experiment queue (kill conditions in PROJECT.md §8 + docs/CLAIMS.md)

**Process:** design → Claim ID → pre-specify primary metric in RESULTS → run → paste
numbers → only then edit `paper/`.

1. ~~Improve VOI estimator~~ — done, partial kill (E3-ablate)
2. **E-eq / E-eq-d** — measure policy equivalence (grounds intro) — cheap next
3. **E3-power** — settle VOI vs BALD/random with true-regret oracle or rewrite abstract
4. **E-decouple** — structural irrelevant features (flagship mechanistic)
5. **E-prior + E-act** — prior necessity + decision rule vs oracle gap
6. **E-calib / E-cover / E-neg** — reviewer-proofing
7. E1-strat, E-ternary, E-mismatch — secondary

**Hardware:** active — robot arm available; sim still owns RSS numbers (`docs/hardware.md`).

---

## Things not to do

- No deep reward network yet (linear keeps exact regret + identifiability).
- Do not tune away a negative result before understanding it.
- Always include the exp-2 caveat: random-query corr(b)=0.753 alongside hard=0.867.
- Keep the **oracle** in every elicitation experiment.
- Do not cite Regan & Boutilier / MaxMin-RLHF theorems as ours.
