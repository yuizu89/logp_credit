# logp_credit/weight/softmax.py
from __future__ import annotations

from typing import List, Optional
import torch


def softmax(
    x: List[float],
    *,
    tau: float = 1.0,
    stable: bool = True,
    mask: Optional[List[bool]] = None,
) -> List[float]:
    """
    Compute softmax weights over a list of scalars.

    Args:
      x: list of floats
      tau: temperature (>0). weights = softmax(x / tau)
      stable: if True, use max-shift for numerical stability
      mask: optional list[bool] same length as x. True means "keep", False means "masked out".
            Masked entries get weight 0, and remaining are renormalized.

    Returns:
      list of weights (len == len(x)), sum(weights over unmasked) == 1 (if any unmasked).
      If all masked or x empty: returns all zeros or empty.
    """
    if len(x) == 0:
        return []

    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    t = torch.tensor(x, dtype=torch.float32)

    if mask is not None:
        if len(mask) != len(x):
            raise ValueError("mask length must match x length")
        m = torch.tensor(mask, dtype=torch.bool)
        if not torch.any(m):
            # all masked
            return [0.0] * len(x)
        # set masked to -inf so softmax gives 0
        t = t.masked_fill(~m, float("-inf"))

    t = t / float(tau)

    if stable:
        # max-shift over finite entries only
        finite = torch.isfinite(t)
        if torch.any(finite):
            t = t - torch.max(t[finite])

    w = torch.softmax(t, dim=0)

    # If some entries were -inf, softmax returns 0 there, but due to float errors,
    # renormalize over unmasked finite entries.
    if mask is not None:
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        s = w.sum()
        if s > 0:
            w = w / s

    return [float(v) for v in w.tolist()]
