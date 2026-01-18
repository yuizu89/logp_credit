# logp_credit/contrib/prefix_marginal.py
from __future__ import annotations

from typing import List, Optional

import torch

from logp_credit.config import PromptConfig, SegmentationConfig, ContributionConfig
from logp_credit.contrib.common import ContribResult
from logp_credit.text.segment import split_segments_by_period
from logp_credit.model.kv import forward_append, score_answer_logprob


def _truncate_segments(
    segs: List[str],
    max_segments: Optional[int],
    policy: str,
) -> tuple[List[str], bool]:
    if max_segments is None or len(segs) <= max_segments:
        return segs, False

    if policy == "head":
        return segs[:max_segments], True
    if policy == "tail":
        return segs[-max_segments:], True
    raise ValueError(f"Unknown truncate_policy: {policy!r} (expected 'head' or 'tail')")


@torch.inference_mode()
def compute_prefix_marginal(
    model,
    tok,
    prompt_ids: torch.Tensor,
    think_text: str,
    true_answer_norm: str,
    *,
    prompt_cfg: PromptConfig,
    seg_cfg: SegmentationConfig,
    contrib_cfg: ContributionConfig,
) -> ContribResult:
    """
    Method 2 (prefix marginal), GT-targeted:

      scores[k] = log P( ans_text | prompt + think_open + prefix_k + think_close )
      deltas[i] = scores[i+1] - scores[i]

    Where:
      - scores[0] uses empty prefix (no segments).
      - ans_text is built from GT: "\\n{hash_prefix}{true_answer_norm}"

    Returns ContribResult with:
      - scores length = 1 + n_segments
      - deltas length = n_segments
    """
    if contrib_cfg.score_target != "gt":
        raise ValueError("compute_prefix_marginal is GT-only by design (score_target must be 'gt').")
    if true_answer_norm is None or str(true_answer_norm).strip() == "":
        raise ValueError("true_answer_norm must be a non-empty string.")

    device = getattr(model, "device", None) or prompt_ids.device
    if prompt_ids.device != device:
        prompt_ids = prompt_ids.to(device)

    # Segment
    segs = split_segments_by_period(think_text)
    segs, truncated = _truncate_segments(segs, seg_cfg.max_segments, seg_cfg.truncate_policy)

    # Prepare tokens (once)
    open_ids = tok(prompt_cfg.think_open, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    close_ids = tok(prompt_cfg.think_close, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # GT answer text (VeRL/GSM8K-style marker)
    # NOTE: prompt_cfg.hash_prefix includes trailing space by design (e.g., "#### ")
    ans_text = f"\n{prompt_cfg.hash_prefix}{true_answer_norm}"
    ans_ids = tok(ans_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # 1) Build base context: prompt + think_open (KV cached)
    past, _ = forward_append(model, prompt_ids, past=None)
    past, _ = forward_append(model, open_ids, past=past)

    # 2) scores[0]: empty prefix (no segments)
    # score = log P(ans | prompt + think_open + think_close)
    past_closed, last_closed = forward_append(model, close_ids, past=past)
    scores: List[float] = [score_answer_logprob(model, past_closed, last_closed, ans_ids)]

    # 3) prefix_i: append segments incrementally to past_inside (do NOT include close in the cached past)
    past_inside = past
    for seg in segs:
        seg_ids = tok(" " + seg, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        past_inside, _ = forward_append(model, seg_ids, past=past_inside)

        # close and score from this prefix
        past_closed, last_closed = forward_append(model, close_ids, past=past_inside)
        scores.append(score_answer_logprob(model, past_closed, last_closed, ans_ids))

    # 4) deltas
    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]

    # If caller wants to *not* keep empty prefix, we can drop it here,
    # but keeping it is strongly recommended for analysis and sanity checks.
    if not contrib_cfg.include_empty_prefix:
        # Drop scores[0]; deltas still correspond to segs but now relative to dropped baseline is ambiguous.
        # We keep behavior explicit: don't drop unless you truly know what you're doing.
        raise ValueError(
            "include_empty_prefix=False is not supported for prefix_marginal, because deltas "
            "are defined relative to the empty-prefix baseline."
        )

    return ContribResult(
        method="prefix",
        score_target="gt",
        segs=segs,
        truncated=truncated,
        split_method=seg_cfg.method,
        ans_text=ans_text,
        scores=scores,
        deltas=deltas,
        notes=None,
    )
