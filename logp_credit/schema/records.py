# logp_credit/schema/records.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, List, Literal


# -------------------------
# Common literals
# -------------------------
ContribMethod = Literal["prefix", "loo"]
SplitMethod = Literal["period"]
ScoreTarget = Literal["gt"]
PredExtraction = Literal["hash", "last_number", "none"]


# -------------------------
# Rollout-level meta (optional container)
# -------------------------
@dataclass(frozen=True)
class RolloutMeta:
    """
    Rollout-level metadata and (optionally) text blobs.
    This is convenient to keep around in memory, but for storage we typically
    flatten needed fields into SegmentRecord (long format).
    """
    # Run identity
    run_id: str
    dataset_name: str
    split: str

    # Example identity
    question_idx: int
    question_id: Optional[str]

    # Sampling identity
    rollout_id: int
    seed: int

    # Text / labels (some may be omitted depending on save_level)
    question: Optional[str]
    true_answer_text: Optional[str]
    true_norm: str

    prompt_text: Optional[str]
    decoded_text: Optional[str]
    think_text: Optional[str]
    rest_text: Optional[str]

    pred_norm: Optional[str]
    pred_extraction: PredExtraction
    correct: bool


# -------------------------
# Long-format storage record: one row per segment
# -------------------------
@dataclass(frozen=True)
class SegmentRecord:
    """
    The canonical long-format schema: 1 row per (question, rollout, segment).
    This schema is designed to be directly converted into a pandas DataFrame.

    Notes:
      - score_target is fixed to "gt" by design.
      - For prefix method:
          score_prev/score_curr/delta are populated.
          full_score/loo are None.
      - For loo method:
          full_score/loo are populated.
          score_prev/score_curr/delta are None (or optionally could be filled).

    Token-window selection:
      - seg_token_len / seg_token_start are used to build token-windows.
      - window_id is computed as seg_token_start // window_tokens (per rollout).
      - is_active indicates whether the segment participates in softmax/reward.
    """

    # -------- identity / keys --------
    run_id: str
    dataset_name: str
    split: str

    question_idx: int
    question_id: Optional[str]

    rollout_id: int
    seed: int

    seg_id: int              # 1..n_segments recommended (no "empty think" row)
    n_segments: int
    truncated: bool

    # -------- method metadata --------
    contrib_method: ContribMethod
    split_method: SplitMethod
    score_target: ScoreTarget  # always "gt"

    # -------- segment content --------
    seg_text: str

    # -------- NEW: token-window bookkeeping --------
    # Token length of this segment (in tokenizer space). Optional for backward compatibility.
    seg_token_len: Optional[int] = None
    # Token start offset of this segment within the rollout (cumulative sum of seg_token_len). Optional.
    seg_token_start: Optional[int] = None
    # Token-window id within rollout (computed from seg_token_start // window_tokens). Optional.
    window_id: Optional[int] = None
    # Whether this segment is selected (top-k) and participates in softmax/reward.
    is_active: bool = True

    # -------- target answer representation --------
    ans_text: str = ""        # e.g. "\n#### 13" (logging)
    true_norm: str = ""

    # -------- rollout outcome (duplicated per row for convenience) --------
    correct: bool = False
    pred_norm: Optional[str] = None
    pred_extraction: PredExtraction = "none"

    # -------- contribution scores (GT logP) --------
    # Prefix method:
    score_prev: Optional[float] = None
    score_curr: Optional[float] = None
    delta: Optional[float] = None

    # LOO method:
    full_score: Optional[float] = None
    loo: Optional[float] = None

    # -------- softmax input / weights --------
    # x_raw: before normalization (typically delta or loo, with any pre-processing).
    x_raw: float = 0.0
    # x_norm: normalized (and optional clipped) value used for softmax.
    x_norm: float = 0.0
    # x_signed: AFTER flip (incorrect -> -x_norm), this is the actual softmax input.
    x_signed: float = 0.0
    tau: float = 1.0
    weight: float = 0.0

    # -------- reward assignment --------
    R: float = 0.0
    assigned_reward: float = 0.0

    # -------- quality / diagnostics --------
    invalid: bool = False
    notes: Optional[str] = None


# -------------------------
# Helpers
# -------------------------
def record_to_dict(rec: Any) -> Dict[str, Any]:
    """
    Convert dataclass record (RolloutMeta / SegmentRecord) to a plain dict.
    """
    return asdict(rec)


def segment_record_columns() -> List[str]:
    """
    Return the canonical column order for SegmentRecord.
    Useful for DataFrame column ordering and stable CSV output.
    """
    return [f.name for f in fields(SegmentRecord)]


def rollout_meta_columns() -> List[str]:
    return [f.name for f in fields(RolloutMeta)]


def make_run_id(prefix: str, timestamp: str, suffix: Optional[str] = None) -> str:
    """
    Optional helper to build a run_id outside of pipeline code.
    """
    if suffix:
        return f"{prefix}_{timestamp}_{suffix}"
    return f"{prefix}_{timestamp}"
