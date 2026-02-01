# logp_credit/model/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from logp_credit.model.kv import forward_append


@torch.inference_mode()
def score_answer_logprob(
    model,
    ctx_past,
    ctx_last_logits: torch.Tensor,
    ans_ids: torch.Tensor,
) -> float:
    """
    Compute log P(ans_ids | context) using:
      - ctx_last_logits for the first token probability
      - ctx_past + ans_ids[:-1] to get logits for remaining tokens

    Args:
      ctx_past: past_key_values for the context
      ctx_last_logits: (1, vocab) logits of next token after the context
      ans_ids: (1, T_ans) token ids to score

    Returns:
      total log probability (float)
    """
    assert ans_ids.ndim == 2 and ans_ids.size(0) == 1, f"ans_ids must be (1,T), got {tuple(ans_ids.shape)}"
    assert ctx_last_logits.ndim == 2 and ctx_last_logits.size(0) == 1, (
        f"ctx_last_logits must be (1,V), got {tuple(ctx_last_logits.shape)}"
    )

    # Ensure device consistency
    if ans_ids.device != ctx_last_logits.device:
        ans_ids = ans_ids.to(ctx_last_logits.device)

    T = ans_ids.size(1)

    # 1) First token prob from context last logits
    logp1 = F.log_softmax(ctx_last_logits, dim=-1).gather(-1, ans_ids[:, :1]).squeeze(-1)  # (1,)
    if T == 1:
        return float(logp1.item())

    # 2) Remaining tokens: feed ans[:-1], predict ans[1:]
    inp = ans_ids[:, :-1]
    out = model(input_ids=inp, past_key_values=ctx_past, use_cache=False)
    logits = out.logits  # (1, T-1, vocab)

    logp = F.log_softmax(logits, dim=-1).gather(-1, ans_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    total = logp1 + logp.sum(dim=-1)  # (1,)
    return float(total.item())


@dataclass(frozen=True)
class ValueAfterHashTokens:
    marker_text: str
    value_text: str
    marker_ids: torch.Tensor  # (1, Tm)
    value_ids: torch.Tensor   # (1, Tv)


def build_value_after_hash_tokens(
    tok,
    *,
    hash_prefix: str,
    true_answer_norm: str,
    device: torch.device,
) -> ValueAfterHashTokens:
    """
    Build tokens for:
      marker_text = "\\n" + hash_prefix   (e.g., "\\n#### ")
      value_text  = true_answer_norm (or with leading space if hash_prefix has no trailing space)

    Rules:
      - If hash_prefix endswith(" "), don't add extra space before value.
      - Otherwise, value_text gets a leading single space to mimic "#### <answer>".
    """
    hp = hash_prefix
    v = str(true_answer_norm).strip()

    marker_text = "\n" + hp
    value_text = v if hp.endswith(" ") else (" " + v)

    marker_ids = tok(marker_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    value_ids = tok(value_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    return ValueAfterHashTokens(
        marker_text=marker_text,
        value_text=value_text,
        marker_ids=marker_ids,
        value_ids=value_ids,
    )


@torch.inference_mode()
def score_value_after_hash(
    model,
    tok,
    *,
    ctx_past,
    true_answer_norm: str,
    hash_prefix: str = "#### ",
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute log P(value | context + "\\n#### ") where:
      - context is represented by ctx_past (past_key_values).
      - We DO NOT score the marker itself; we condition on it by appending marker_ids to ctx_past.
      - Then we score only the value token sequence.

    This is the 1B design:
      - "#### " is conditioned (format is given)
      - only the value tokens contribute to the score

    Returns:
      scalar float (log probability)
    """
    # Choose a device for tokenization tensors.
    # For single-GPU runs, model.device is fine. You can override via `device=...`.
    dev = device or getattr(model, "device", None)
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    toks = build_value_after_hash_tokens(
        tok,
        hash_prefix=hash_prefix,
        true_answer_norm=true_answer_norm,
        device=dev,
    )

    # 1) Append marker to context (conditioning)
    past_m, last_logits_m = forward_append(model, toks.marker_ids, past=ctx_past)

    # 2) Score only the value tokens after marker
    return score_answer_logprob(model, past_m, last_logits_m, toks.value_ids)
