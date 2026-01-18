# logp_credit/contrib/common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


ContribMethod = Literal["prefix", "loo"]
ScoreTarget = Literal["gt"]


@dataclass(frozen=True)
class ContribResult:
    """
    Contribution estimation result for one rollout.
    For method="prefix":
      - scores: length = 1 + n_segments (scores[0] is empty-think baseline)
      - deltas: length = n_segments, deltas[i] = scores[i+1] - scores[i]
    """
    method: ContribMethod
    score_target: ScoreTarget

    segs: List[str]
    truncated: bool
    split_method: str

    ans_text: str

    # Prefix method outputs
    scores: List[float]
    deltas: List[float]

    # Optional diagnostics
    notes: Optional[str] = None

    def n_segments(self) -> int:
        return len(self.segs)
