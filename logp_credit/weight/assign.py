# logp_credit/weight/assign.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from logp_credit.config import (
    NormalizationConfig,
    SoftmaxConfig,
    RewardConfig,
    WindowSelectConfig,
)
from logp_credit.contrib.common import ContribResult
from logp_credit.schema.records import RolloutMeta, SegmentRecord
from logp_credit.weight.normalize import normalize_values
from logp_credit.weight.softmax import softmax


# -------------------------
# Helpers
# -------------------------
def _rollout_reward_scalar(reward_cfg: RewardConfig, correct: bool) -> float:
    return float(reward_cfg.reward_correct if correct else reward_cfg.reward_incorrect)


def _base_contrib_values(contrib: ContribResult) -> List[float]:
    """
    Return per-segment base contribution values BEFORE any flip.
    - prefix: deltas
    - loo: (future) per-seg loo values
    """
    if contrib.method == "prefix":
        return list(contrib.deltas)

    if contrib.method == "loo":
        raise NotImplementedError(
            "LOO support requires ContribResult to carry per-segment loo values."
        )

    raise ValueError(f"Unknown contrib.method: {contrib.method!r}")


def _is_finite(x: float) -> bool:
    return (x == x) and (x != float("inf")) and (x != float("-inf"))


def _metric_for_selection(r: SegmentRecord) -> float:
    """
    Metric used for window selection (pre-flip).
    Current design: use per-segment delta (prefix) / loo (future).
    We fallback to x_raw if delta is None.
    """
    if r.delta is not None:
        return float(r.delta)
    if r.loo is not None:
        return float(r.loo)
    return float(r.x_raw)


def _compute_window_id(seg_token_start: Optional[int], window_tokens: int) -> Optional[int]:
    if seg_token_start is None:
        return None
    return int(seg_token_start // window_tokens)


# -------------------------
# Pass 1: build base records (x_raw only + token bookkeeping)
# -------------------------
def build_segment_records_base_from_rollout(
    meta: RolloutMeta,
    contrib: ContribResult,
) -> List[SegmentRecord]:
    """
    Build SegmentRecord list with:
      - x_raw filled as base contribution (delta/loo), BEFORE flip
      - x_norm/x_signed/weight/assigned_reward are set to 0 for now
      - scores/delta fields are filled (prefix), so later plotting works
      - NEW: seg_token_len / seg_token_start are filled when available

    This is "pass 1" of a two-pass pipeline.
    """
    segs = contrib.segs
    n = len(segs)
    if n == 0:
        return []

    base = _base_contrib_values(contrib)
    if len(base) != n:
        raise ValueError(f"Contribution length mismatch: len(base)={len(base)} vs n_segments={n}")

    # NEW: token length & start offset per segment (aligned with segs)
    seg_token_lens: Optional[List[int]] = contrib.seg_token_lens
    if seg_token_lens is not None and len(seg_token_lens) != n:
        raise ValueError(
            f"seg_token_lens length mismatch: len(seg_token_lens)={len(seg_token_lens)} vs n_segments={n}"
        )

    seg_token_starts: Optional[List[int]] = None
    if seg_token_lens is not None:
        starts: List[int] = []
        cur = 0
        for L in seg_token_lens:
            starts.append(int(cur))
            cur += int(L)
        seg_token_starts = starts

    rows: List[SegmentRecord] = []
    for i, seg_text in enumerate(segs, start=1):
        score_prev = score_curr = delta = None
        full_score = loo = None

        if contrib.method == "prefix":
            # scores: length 1+n, deltas: length n
            score_prev = float(contrib.scores[i - 1])
            score_curr = float(contrib.scores[i])
            delta = float(contrib.deltas[i - 1])

        rec = SegmentRecord(
            # identity
            run_id=meta.run_id,
            dataset_name=meta.dataset_name,
            split=meta.split,
            question_idx=meta.question_idx,
            question_id=meta.question_id,
            rollout_id=meta.rollout_id,
            seed=meta.seed,
            seg_id=i,
            n_segments=n,
            truncated=contrib.truncated,

            # method
            contrib_method=contrib.method,
            split_method=contrib.split_method,
            score_target=contrib.score_target,

            # content
            seg_text=seg_text,

            # NEW: token bookkeeping (window_id/is_active computed in pass2)
            seg_token_len=(int(seg_token_lens[i - 1]) if seg_token_lens is not None else None),
            seg_token_start=(int(seg_token_starts[i - 1]) if seg_token_starts is not None else None),
            window_id=None,
            is_active=True,

            # answer/labels
            ans_text=contrib.ans_text,
            true_norm=meta.true_norm,
            correct=meta.correct,
            pred_norm=meta.pred_norm,
            pred_extraction=meta.pred_extraction,

            # scores
            score_prev=score_prev,
            score_curr=score_curr,
            delta=delta,
            full_score=full_score,
            loo=loo,

            # pass1: base only
            x_raw=float(base[i - 1]),  # pre-flip
            x_norm=0.0,
            x_signed=0.0,
            tau=0.0,
            weight=0.0,
            R=0.0,
            assigned_reward=0.0,

            invalid=(not _is_finite(float(base[i - 1]))),
            notes=None,
        )
        rows.append(rec)

    return rows


# -------------------------
# Pass 2: (optional) window selection -> normalize -> per-rollout softmax -> reward
# -------------------------
def finalize_records_two_pass(
    records: List[SegmentRecord],
    *,
    norm_cfg: NormalizationConfig,
    softmax_cfg: SoftmaxConfig,
    reward_cfg: RewardConfig,
    window_cfg: Optional[WindowSelectConfig] = None,
    separate_correctness: bool = False,  # kept for compatibility; prefer norm_cfg.separate_correctness
) -> List[SegmentRecord]:
    """
    Two-pass finalize:
      0) (Optional) Token-window selection:
           - Compute window_id = seg_token_start // window_tokens
           - For each rollout & window:
               correct   -> pick top-k by metric (max delta)
               incorrect -> pick top-k by metric (min delta)
           - Only selected segments are active (is_active=True); others get weight=0 and assigned_reward=0.
      1) Compute x_norm by normalizing x_raw over the chosen scope group.
         (clip is applied AFTER normalization; per requirement)
         Normalization stats are computed using ACTIVE & finite segments only.
      2) Apply flip AFTER normalization:
           x_signed = x_norm (correct)
           x_signed = -x_norm (incorrect)  if reward_cfg.flip_on_incorrect
         Inactive segments get x_signed=0.
      3) Compute softmax weights WITHIN EACH ROLLOUT over x_signed / tau
         using mask = (is_active and finite and not invalid), if mask_invalid=True.
      4) assigned_reward = R * weight, where R depends on correctness.

    Scope rules:
      - scope="rollout": normalize within each (question_idx, rollout_id, method)
      - scope="question": normalize across all rollouts+segments for each question (and method)
      - scope="batch": normalize across entire batch (and method)

    NOTE:
      Softmax is always per-rollout across segments, regardless of normalization scope.
    """
    if len(records) == 0:
        return []

    # -------------------------
    # 0) Token-window selection (optional)
    # -------------------------
    do_window = (window_cfg is not None) and bool(window_cfg.enabled)

    window_id_out: List[Optional[int]] = [r.window_id for r in records]
    is_active_out: List[bool] = [bool(r.is_active) for r in records]

    # Rollout key (same as softmax granularity)
    def rollout_key(r: SegmentRecord) -> Tuple:
        return (r.run_id, r.question_idx, r.rollout_id, r.contrib_method, r.split_method, r.score_target)

    rollouts: Dict[Tuple, List[int]] = {}
    for idx, r in enumerate(records):
        rollouts.setdefault(rollout_key(r), []).append(idx)

    if do_window:
        wtoks = int(window_cfg.window_tokens)
        k_per = int(window_cfg.top_k_per_window)

        for rkey, idxs in rollouts.items():
            idxs_sorted = sorted(idxs, key=lambda i: records[i].seg_id)
            correct = records[idxs_sorted[0]].correct

            # Compute window_id for this rollout
            for i in idxs_sorted:
                window_id_out[i] = _compute_window_id(records[i].seg_token_start, wtoks)

            # If any window_id is None (missing token_start), skip selection for this rollout
            if any(window_id_out[i] is None for i in idxs_sorted):
                # keep all active (do not restrict)
                for i in idxs_sorted:
                    is_active_out[i] = True
                continue

            # Group by window_id
            by_win: Dict[int, List[int]] = {}
            for i in idxs_sorted:
                wid = int(window_id_out[i])  # not None here
                by_win.setdefault(wid, []).append(i)

            # Select top-k within each window
            for wid, win_idxs in by_win.items():
                # Filter finite candidates by selection metric
                cand = []
                for i in win_idxs:
                    m = _metric_for_selection(records[i])
                    if _is_finite(m) and (not records[i].invalid):
                        cand.append((m, i))

                # Default: none active if no valid candidates
                for i in win_idxs:
                    is_active_out[i] = False

                if len(cand) == 0:
                    continue

                # correct -> max; incorrect -> min
                cand.sort(key=lambda t: t[0], reverse=bool(correct))
                k = min(k_per, len(cand))
                chosen = [i for _, i in cand[:k]]
                for i in chosen:
                    is_active_out[i] = True

    # -------------------------
    # 1) Build normalization groups (for x_norm)
    # -------------------------
    def norm_key(r: SegmentRecord) -> Tuple:
        base = (r.contrib_method, r.split_method, r.score_target)
        if norm_cfg.scope == "rollout":
            k = base + (r.question_idx, r.rollout_id)
        elif norm_cfg.scope == "question":
            k = base + (r.question_idx,)
        elif norm_cfg.scope == "batch":
            k = base
        else:
            raise ValueError(f"Unknown norm scope: {norm_cfg.scope!r}")

        sep = norm_cfg.separate_correctness or separate_correctness
        if sep:
            k = k + (r.correct,)
        return k

    groups: Dict[Tuple, List[int]] = {}
    for idx, r in enumerate(records):
        groups.setdefault(norm_key(r), []).append(idx)

    x_norm_out: List[float] = [0.0] * len(records)
    invalid_out: List[bool] = [False] * len(records)

    for gkey, idxs in groups.items():
        # Use ACTIVE finite values only for stats
        active_idxs = [i for i in idxs if is_active_out[i]]
        if len(active_idxs) == 0:
            for i in idxs:
                invalid_out[i] = True
                x_norm_out[i] = 0.0
            continue

        vals = [records[i].x_raw for i in active_idxs]
        finite_mask = [_is_finite(v) and (not records[i].invalid) for v, i in zip(vals, active_idxs)]

        if not any(finite_mask):
            for i in idxs:
                invalid_out[i] = True
                x_norm_out[i] = 0.0
            continue

        finite_vals = [v for v, m in zip(vals, finite_mask) if m]
        finite_norm = normalize_values(
            finite_vals,
            method=norm_cfg.method,
            eps=norm_cfg.eps,
            clip=norm_cfg.clip,  # clip AFTER normalization
        )

        it = iter(finite_norm)
        # Fill active positions
        for local_j, rec_i in enumerate(active_idxs):
            if finite_mask[local_j]:
                x_norm_out[rec_i] = float(next(it))
                invalid_out[rec_i] = bool(records[rec_i].invalid)
            else:
                x_norm_out[rec_i] = 0.0
                invalid_out[rec_i] = True

        # Inactive positions: keep 0 and mark invalid so they won't participate
        for rec_i in idxs:
            if not is_active_out[rec_i]:
                x_norm_out[rec_i] = 0.0
                invalid_out[rec_i] = True

    # -------------------------
    # 2) Flip AFTER normalization to get x_signed (inactive => 0)
    # -------------------------
    x_signed_out: List[float] = [0.0] * len(records)
    for i, r in enumerate(records):
        if not is_active_out[i]:
            x_signed_out[i] = 0.0
            continue

        xn = x_norm_out[i]
        if (not r.correct) and reward_cfg.flip_on_incorrect:
            x_signed_out[i] = -xn
        else:
            x_signed_out[i] = xn

    # -------------------------
    # 3) Softmax weights per rollout (mask inactive/invalid if requested)
    # -------------------------
    weight_out: List[float] = [0.0] * len(records)
    R_out: List[float] = [0.0] * len(records)
    assigned_out: List[float] = [0.0] * len(records)
    tau_out: List[float] = [float(softmax_cfg.tau)] * len(records)

    for rkey, idxs in rollouts.items():
        idxs_sorted = sorted(idxs, key=lambda i: records[i].seg_id)

        xs = [x_signed_out[i] for i in idxs_sorted]
        mask = None
        if softmax_cfg.mask_invalid:
            mask = []
            for j, rec_i in enumerate(idxs_sorted):
                ok = (
                    is_active_out[rec_i]
                    and (not invalid_out[rec_i])
                    and _is_finite(xs[j])
                )
                mask.append(bool(ok))

        w = softmax(
            xs,
            tau=softmax_cfg.tau,
            stable=softmax_cfg.stable,
            mask=mask,
        )

        correct = records[idxs_sorted[0]].correct
        R = _rollout_reward_scalar(reward_cfg, correct)

        for local_j, rec_i in enumerate(idxs_sorted):
            # If inactive, keep 0 regardless of softmax impl details
            if not is_active_out[rec_i]:
                weight_out[rec_i] = 0.0
                R_out[rec_i] = float(R)
                assigned_out[rec_i] = 0.0
                continue

            weight_out[rec_i] = float(w[local_j])
            R_out[rec_i] = float(R)
            assigned_out[rec_i] = float(R * w[local_j])

    # -------------------------
    # 4) Emit updated records
    # -------------------------
    out: List[SegmentRecord] = []
    for i, r in enumerate(records):
        out.append(
            replace(
                r,
                window_id=window_id_out[i],
                is_active=bool(is_active_out[i]),
                x_norm=float(x_norm_out[i]),
                x_signed=float(x_signed_out[i]),
                tau=float(tau_out[i]),
                weight=float(weight_out[i]),
                R=float(R_out[i]),
                assigned_reward=float(assigned_out[i]),
                invalid=bool(invalid_out[i]),
            )
        )
    return out
