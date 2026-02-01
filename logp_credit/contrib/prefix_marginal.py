# logp_credit/contrib/prefix_marginal.py
from __future__ import annotations

from typing import List, Optional

import torch

from logp_credit.config import PromptConfig, SegmentationConfig, ContributionConfig
from logp_credit.contrib.common import ContribResult
from logp_credit.text.segment import split_segments_by_period
from logp_credit.model.kv import forward_append
from logp_credit.model.scoring import score_value_after_hash


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
    Method 2 (prefix marginal), GT-targeted, "1B" scoring:

      scores[k] = log P( value | prompt + think_open + prefix_k + think_close + "\\n#### " )
      deltas[i] = scores[i+1] - scores[i]

    Where:
      - scores[0] uses empty prefix (no segments).
      - hash_prefix is conditioned (NOT scored), and only the value tokens are scored.

    Returns ContribResult with:
      - scores length = 1 + n_segments
      - deltas length = n_segments
      - seg_token_lens length = n_segments
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
    segs = [s for s in segs if s and s.strip()]
    segs, truncated = _truncate_segments(segs, seg_cfg.max_segments, seg_cfg.truncate_policy)

    # Prepare tokens once
    open_ids = tok(prompt_cfg.think_open, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    close_ids = tok(prompt_cfg.think_close, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # For logging only (NOT used for scoring in 1B)
    ans_text = f"\n{prompt_cfg.hash_prefix}{true_answer_norm}"

    # 1) Base context: prompt + think_open (KV cached)
    past, _ = forward_append(model, prompt_ids, past=None)
    past, _ = forward_append(model, open_ids, past=past)

    # 2) scores[0]: empty prefix baseline
    # context = prompt + think_open + think_close
    past_closed, _ = forward_append(model, close_ids, past=past)
    scores: List[float] = [
        score_value_after_hash(
            model,
            tok,
            ctx_past=past_closed,
            true_answer_norm=true_answer_norm,
            hash_prefix=prompt_cfg.hash_prefix,
            device=device,
        )
    ]

    # NEW: per-segment token lengths (aligned with segs)
    seg_token_lens: List[int] = []

    # 3) prefix_i: append segments incrementally, then close and score
    past_inside = past
    for seg in segs:
        seg_ids = tok(" " + seg, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        seg_token_lens.append(int(seg_ids.size(1)))  # <-- NEW

        past_inside, _ = forward_append(model, seg_ids, past=past_inside)

        past_closed, _ = forward_append(model, close_ids, past=past_inside)
        scores.append(
            score_value_after_hash(
                model,
                tok,
                ctx_past=past_closed,
                true_answer_norm=true_answer_norm,
                hash_prefix=prompt_cfg.hash_prefix,
                device=device,
            )
        )

    # 4) deltas
    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]

    if not contrib_cfg.include_empty_prefix:
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
        seg_token_lens=seg_token_lens,           # <-- NEW
        notes="score_value_after_hash (1B)",
    )
