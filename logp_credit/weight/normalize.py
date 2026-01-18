# logp_credit/weight/normalize.py
from __future__ import annotations

from typing import List, Optional, Literal
import math


NormMethod = Literal["zscore", "robust", "none"]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: List[float], eps: float) -> float:
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) + eps


def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def _mad(xs: List[float], eps: float) -> float:
    med = _median(xs)
    dev = [abs(x - med) for x in xs]
    mad = _median(dev)
    # MAD -> robust std estimate: 1.4826 * MAD (for normal)
    return 1.4826 * mad + eps


def _clip(xs: List[float], clip: Optional[float]) -> List[float]:
    if clip is None:
        return xs
    c = float(clip)
    return [max(-c, min(c, x)) for x in xs]


def normalize_values(
    x_raw: List[float],
    *,
    method: NormMethod,
    eps: float,
    clip: Optional[float],
) -> List[float]:
    """
    Normalize x_raw -> x_norm.

    - method="zscore": (x - mean)/std
    - method="robust": (x - median)/(1.4826*MAD)
    - method="none":   x

    IMPORTANT: clip is applied AFTER normalization: x_norm = clip(x_norm, [-clip, +clip])
    """
    if len(x_raw) == 0:
        return []

    if method == "none":
        x_norm = list(x_raw)
        return _clip(x_norm, clip)

    if method == "zscore":
        mu = _mean(x_raw)
        sd = _std(x_raw, eps)
        x_norm = [(x - mu) / sd for x in x_raw]
        return _clip(x_norm, clip)

    if method == "robust":
        med = _median(x_raw)
        scale = _mad(x_raw, eps)
        x_norm = [(x - med) / scale for x in x_raw]
        return _clip(x_norm, clip)

    raise ValueError(f"Unknown normalization method: {method!r}")