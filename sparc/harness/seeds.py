"""Master seed list locked in EXPERIMENTS.md."""

from __future__ import annotations

MASTER_SEEDS: list[int] = list(range(10))


def parse_seeds(spec: str | None, default: list[int] | None = None) -> list[int]:
    """
    Parse seed CLI argument.

    - ``None`` or ``""`` → default (single seed from config)
    - ``all`` → MASTER_SEEDS (10 seeds)
    - ``0,1,2`` → explicit list
    """
    if spec is None or spec.strip() == "":
        return default or [0]
    if spec.strip().lower() == "all":
        return MASTER_SEEDS.copy()
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"no seeds parsed from: {spec!r}")
    return out
