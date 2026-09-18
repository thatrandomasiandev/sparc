# CARC (USC Discovery) — experiment cluster helpers

Topic-agnostic wrappers for syncing this repo to Discovery and submitting Slurm jobs.
Defaults assume lab account `biyik_1165` and netid `jjt_373` — override via `carc/env.sh`.

## Paths (defaults)

| Role | Path |
|------|------|
| SSH host | `discovery` (`~/.ssh/config`) |
| Account | `biyik_1165` |
| Repo on CARC | `/project2/biyik_1165/jjt_373/research` |
| Scratch runs | `/scratch1/jjt_373/research/experiments/runs` |
| Conda | `/project2/biyik_1165/jjt_373/miniconda3` env `research` |

## Local → cluster

```bash
./carc/scripts/sync_to_carc.sh
./carc/scripts/submit.sh carc/jobs/bootstrap_env.job   # first time
./carc/scripts/submit.sh carc/jobs/smoke.job
./carc/scripts/status.sh
./carc/scripts/sync_from_carc.sh experiments/runs/carc_smoke
```

Update job scripts once `EXPERIMENTS.md` locks the real suite command.
