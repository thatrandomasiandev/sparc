"""Sequential probability ratio test per operator (DESIGN.md Pillar 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from sparc.drift_detector.config import DriftDetectorConfig
from sparc.drift_detector.likelihood import log_likelihood_ratio, mean_preference_delta
from sparc.env.dataset import PreferenceRecord
from sparc.query_optimizer.eig import QueryCandidate
from sparc.reward_model.ensemble import RewardEnsemble


@dataclass
class OperatorDriftState:
    log_lambda: float = 0.0
    regime: int = 0
    requery: bool = False


@dataclass(frozen=True)
class DriftTriggerEvent:
    operator_id: int
    regime: int
    log_lambda: float


@dataclass
class HistoricalQuery:
    """Query issued in a past window with disagreement score for re-query pool."""

    candidate: QueryCandidate
    disagreement: float
    window_id: int
    operator_id: int


@dataclass
class DriftDetector:
    """Online SPRT over incoming preference labels; sets re-query flags on trigger."""

    ensemble: RewardEnsemble
    config: DriftDetectorConfig
    operator_ids: list[int] = field(default_factory=lambda: [1, 2, 3])
    states: dict[int, OperatorDriftState] = field(default_factory=dict)
    query_history: list[HistoricalQuery] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.states:
            self.states = {op: OperatorDriftState() for op in self.operator_ids}

    def update_with_labels(
        self,
        records: list[PreferenceRecord],
    ) -> list[DriftTriggerEvent]:
        """Process a batch of newly arrived labels; may trigger drift for operators."""
        triggered: list[DriftTriggerEvent] = []
        cfg = self.config
        a_bound = cfg.boundary_upper
        b_bound = cfg.boundary_lower

        for record in records:
            if not record.is_labeled:
                continue
            op_id = record.query.operator_id
            state = self.states[op_id]
            delta_h0 = mean_preference_delta(self.ensemble, record)
            assert record.label is not None
            ell = log_likelihood_ratio(record.label, delta_h0, cfg.eta)
            state.log_lambda += ell

            if state.log_lambda >= a_bound:
                state.regime += 1
                state.requery = True
                triggered.append(
                    DriftTriggerEvent(
                        operator_id=op_id,
                        regime=state.regime,
                        log_lambda=state.log_lambda,
                    )
                )
                state.log_lambda = 0.0
            elif state.log_lambda <= b_bound:
                state.log_lambda = 0.0

        return triggered

    def record_window_queries(
        self,
        selected: list[QueryCandidate],
        window_id: int,
        operator_id: int,
    ) -> None:
        """Track issued queries for top-disagreement re-query selection."""
        for cand in selected:
            d = self.ensemble.disagreement(
                cand.segment_0,
                cand.segment_1,
                operator_id,
            )
            self.query_history.append(
                HistoricalQuery(
                    candidate=cand,
                    disagreement=d,
                    window_id=window_id,
                    operator_id=operator_id,
                )
            )
        self._trim_history()

    def _trim_history(self) -> None:
        if not self.query_history:
            return
        max_window = max(h.window_id for h in self.query_history)
        min_window = max_window - self.config.history_windows + 1
        self.query_history = [h for h in self.query_history if h.window_id >= min_window]

    def requery_operator(self) -> int | None:
        """Return operator id with active re-query flag, if any."""
        flagged = [op for op, st in self.states.items() if st.requery]
        if not flagged:
            return None
        return max(flagged, key=lambda op: self.states[op].regime)

    def clear_requery(self, operator_id: int) -> None:
        self.states[operator_id].requery = False

    def top_disagreement_candidates(
        self,
        operator_id: int,
        count: int,
    ) -> list[QueryCandidate]:
        """Top historical queries by disagreement for operator (DESIGN.md re-query pool)."""
        pool = [h for h in self.query_history if h.operator_id == operator_id]
        if not pool:
            return []
        pool.sort(key=lambda h: h.disagreement, reverse=True)
        top_n = max(1, int(len(pool) * self.config.top_fraction))
        top = pool[:top_n]
        seen: set[int] = set()
        out: list[QueryCandidate] = []
        for h in top:
            key = id(h.candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(h.candidate)
            if len(out) >= count:
                break
        return out
