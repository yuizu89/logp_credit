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
      - seg_token_lens: length = n_segments (token length per seg, optional)

    NOTE:
      seg_token_lens is used for token-window selection (top-k per window).
      Keep it Optional to preserve backward compatibility (e.g., old runs / other methods).
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

    # NEW: token lengths per segment (aligned with segs)
    seg_token_lens: Optional[List[int]] = None

    # Optional diagnostics
    notes: Optional[str] = None

    def n_segments(self) -> int:
        return len(self.segs)

    def has_token_lens(self) -> bool:
        return self.seg_token_lens is not None
