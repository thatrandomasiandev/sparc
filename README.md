# Decision-relevant elicitation in preference-based RL

Select preference queries by **expected reduction in policy regret** under a
**population prior** — not by information gain about reward parameters.

| Doc | Role |
|-----|------|
| [`AGENTS.md`](AGENTS.md) | Invariants, repo map, standards (**wins on invariants**; includes **I8**) |
| [`docs/CLAIMS.md`](docs/CLAIMS.md) | Every paper claim → experiment (thorough catalog) |
| [`PROJECT.md`](PROJECT.md) | Full scientific context + results (**wins on science**) |
| [`RESULTS.md`](RESULTS.md) | Compact results log |
| [`paper/`](paper/) | LaTeX manuscript (RSS 2027 target) |

**Novelty boundary:** regret-based elicitation is Regan & Boutilier (2009). Our opening is
the instantiation — human trajectory comparisons + population prior (+ later continuous
control). Cite Regan & Boutilier and Model-Free Preference Elicitation (IJCAI 2024) in ¶1.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test
```

Deps: **PyTorch, NumPy, pytest** only.

---

## Layout

```
plr/                 likelihood, users, encoder, acquisition, mdp
tests/test_core.py   claim tests (BT≡PL, trivial BALD=0, I1–I4, …)
experiment_1.py      factored vs VPL-style latent (NEGATIVE)
experiment_2.py      consistency recovery vs difficulty
experiment_3.py      random / BALD / VOI / oracle
aggregate_exp3.py    paired bootstrap CIs
```

---

## Reproduce

```bash
# Claim tests
pytest -q

# Exp 3 smoke (seconds–minutes)
python experiment_3.py --seed 0 --d 4 --n-users 4 --n-queries 5 --n-particles 16 --n-candidates 8 \
  --out experiments/runs/exp3_smoke.jsonl
python aggregate_exp3.py experiments/runs/exp3_smoke.jsonl

# Powered d=8 (~25 min): background + log
python experiment_3.py --seeds 0 1 2 3 4 5 6 7 8 9 --d 8 --n-users 15 \
  --out experiments/runs/exp3_d8.jsonl > experiments/runs/exp3_d8.log 2>&1 &
```

Primary metric for the next paper table: **final-query regret** (see PROJECT.md §5.3).
Queries-to-threshold is secondary.

---

## Experiment queue

See `AGENTS.md` / PROJECT.md §8. Next highest value: **improve the VOI estimator** (close
the oracle gap).

**Hardware:** parked — author is not operating lab robots for this submission
(`docs/hardware.md`). Sim-only paper is the plan.

