"""One human participant session: show pairs, record answers + wall-clock costs.

Costs go into ``QueryCosts`` — never invent them (I3).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

from plr.acquisition import QueryCosts
from plr.hardware.segments import SegmentLibrary

AnswerFn = Callable[[Tensor, list[str]], list[int]]
# (phi_Kd, option_labels) -> ranking order (best-first indices)


@dataclass
class SessionConfig:
    session_id: str
    platform: str
    modality: str = "binary"
    strategy: str = "voi"  # label only — schedule is supplied by caller
    seed: int = 0
    git_commit: str | None = None
    irb_protocol: str = "TODO"
    notes: str = ""


@dataclass
class PreferenceSession:
    """Runs a fixed list of queries; writes the hardware logging contract."""

    config: SessionConfig
    library: SegmentLibrary
    out_dir: Path
    answer_fn: AnswerFn
    query_pairs: list[tuple[int, int]] = field(default_factory=list)
    _latencies_s: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta = asdict(self.config)
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def run(self) -> QueryCosts:
        if not self.query_pairs:
            raise ValueError("query_pairs is empty — build a VOI/BALD schedule first")
        q_path = self.out_dir / "queries.jsonl"
        a_path = self.out_dir / "answers.jsonl"
        with q_path.open("w") as qf, a_path.open("w") as af:
            for t, (i, j) in enumerate(self.query_pairs):
                phi = self.library.pair_features(i, j)  # (2, d)
                labels = [
                    self.library.segments[i].segment_id,
                    self.library.segments[j].segment_id,
                ]
                t0 = time.perf_counter()
                order = self.answer_fn(phi, labels)
                dt = time.perf_counter() - t0
                self._latencies_s.append(dt)
                qf.write(
                    json.dumps(
                        {
                            "t": t,
                            "i": i,
                            "j": j,
                            "labels": labels,
                            "modality": self.config.modality,
                            "strategy": self.config.strategy,
                            "seconds": dt,
                        }
                    )
                    + "\n"
                )
                af.write(json.dumps({"t": t, "order": order}) + "\n")

        costs = self.measured_costs()
        cost_payload = {self.config.modality: costs.cost(self.config.modality)}
        (self.out_dir / "costs.jsonl").write_text(
            json.dumps({"median_seconds": cost_payload, "n": len(self._latencies_s)}) + "\n"
        )
        return costs

    def measured_costs(self) -> QueryCosts:
        if not self._latencies_s:
            raise RuntimeError("No trials recorded — run() first")
        med = float(torch.tensor(self._latencies_s).median().item())
        return QueryCosts(seconds={self.config.modality: med})


def cli_answer_fn(phi: Tensor, labels: list[str]) -> list[int]:
    """Minimal terminal UI for lab dry-runs. Replace with video GUI in the lab."""
    print("\nWhich behavior do you prefer?")
    for k, lab in enumerate(labels):
        print(f"  [{k}] {lab}")
    while True:
        raw = input("Enter index of the better option: ").strip()
        try:
            choice = int(raw)
            if choice in (0, 1):
                other = 1 - choice
                return [choice, other]
        except ValueError:
            pass
        print("Please enter 0 or 1.")
