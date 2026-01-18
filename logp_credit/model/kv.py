# logp_credit/model/kv.py
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def forward_append(model, input_ids: torch.Tensor, past=None):
    """
    Append input_ids to the model context, optionally using cached past_key_values.

    Args:
      input_ids: (1, T_new)
      past: past_key_values or None

    Returns:
      new_past: updated past_key_values
      last_logits: (1, vocab) logits for next token after the last token of this chunk
    """
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    new_past = out.past_key_values
    last_logits = out.logits[:, -1, :]
    return new_past, last_logits


@torch.inference_mode()
def score_answer_logprob(model, ctx_past, ctx_last_logits, ans_ids: torch.Tensor) -> float:
    """
    Compute log P(ans_ids | context) using:
      - ctx_last_logits for the first token probability
      - ctx_past + ans_ids[:-1] to get logits for remaining tokens

    Args:
      ctx_past: past_key_values for the context
      ctx_last_logits: (1, vocab) logits of next token after the context
      ans_ids: (1, T_ans)

    Returns:
      total log probability (float)
    """
    assert ans_ids.ndim == 2 and ans_ids.size(0) == 1
    T = ans_ids.size(1)

    logp1 = F.log_softmax(ctx_last_logits, dim=-1).gather(-1, ans_ids[:, :1]).squeeze(-1)  # (1,)
    if T == 1:
        return float(logp1.item())

    # feed ans[:-1], predict ans[1:]
    inp = ans_ids[:, :-1]
    out = model(input_ids=inp, past_key_values=ctx_past, use_cache=False)
    logits = out.logits  # (1, T-1, vocab)

    logp = F.log_softmax(logits, dim=-1).gather(-1, ans_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    total = logp1 + logp.sum(dim=-1)  # (1,)
    return float(total.item())
