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