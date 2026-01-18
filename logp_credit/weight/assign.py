# logp_credit/weight/assign.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

from logp_credit.config import NormalizationConfig, SoftmaxConfig, RewardConfig
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


# -------------------------
# Pass 1: build base records (x_raw only)
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

    This is "pass 1" of a two-pass pipeline.
    """
    segs = contrib.segs
    n = len(segs)
    if n == 0:
        return []

    base = _base_contrib_values(contrib)
    if len(base) != n:
        raise ValueError(f"Contribution length mismatch: len(base)={len(base)} vs n_segments={n}")

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
            x_raw=float(base[i - 1]),   # pre-flip
            x_norm=0.0,                # to be filled
            x_signed=0.0,              # to be filled
            tau=0.0,                   # to be filled
            weight=0.0,                # to be filled

            R=0.0,                     # to be filled
            assigned_reward=0.0,       # to be filled

            invalid=(not _is_finite(float(base[i - 1]))),
            notes=None,
        )
        rows.append(rec)

    return rows


# -------------------------
# Pass 2: group-normalize (question/batch/rollout) then per-rollout softmax
# -------------------------
def finalize_records_two_pass(
    records: List[SegmentRecord],
    *,
    norm_cfg: NormalizationConfig,
    softmax_cfg: SoftmaxConfig,
    reward_cfg: RewardConfig,
    separate_correctness: bool = False,
) -> List[SegmentRecord]:
    """
    Two-pass finalize:
      1) Compute x_norm by normalizing x_raw over the chosen scope group.
         (clip is applied AFTER normalization; per your requirement)
      2) Apply flip AFTER normalization:
           x_signed = x_norm (correct)
           x_signed = -x_norm (incorrect)  if reward_cfg.flip_on_incorrect
      3) Compute softmax weights WITHIN EACH ROLLOUT over x_signed / tau
      4) assigned_reward = R * weight, where R depends on correctness

    Scope rules:
      - scope="rollout": normalize within each (question_idx, rollout_id, contrib_method)
      - scope="question": normalize across all rollouts+segments for each question (and method)
      - scope="batch": normalize across entire batch (and method)

    separate_correctness:
      - If True, the normalization group additionally splits by correct/incorrect.
        (Useful for analysis; optional.)

    NOTE:
      Softmax is always per-rollout across segments, regardless of normalization scope.
    """
    if len(records) == 0:
        return []

    # ---- 1) Build normalization groups (for x_norm) ----
    # Always keep methods separate to avoid mixing prefix-delta with future loo, etc.
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

        if norm_cfg.separate_correctness:
            k = k + (r.correct,)
        return k

    groups: Dict[Tuple, List[int]] = {}
    for idx, r in enumerate(records):
        groups.setdefault(norm_key(r), []).append(idx)

    # Compute x_norm per group (keeping index alignment)
    x_norm_out: List[float] = [0.0] * len(records)
    invalid_out: List[bool] = [False] * len(records)

    for gkey, idxs in groups.items():
        vals = [records[i].x_raw for i in idxs]

        # If any invalid in group, still normalize finite entries; mark non-finite as invalid.
        finite_mask = [_is_finite(v) for v in vals]
        if not any(finite_mask):
            # all invalid -> x_norm stays 0, mark invalid
            for i in idxs:
                invalid_out[i] = True
            continue

        finite_vals = [v for v, m in zip(vals, finite_mask) if m]
        finite_norm = normalize_values(
            finite_vals,
            method=norm_cfg.method,
            eps=norm_cfg.eps,
            clip=norm_cfg.clip,   # clip AFTER normalization (your requirement)
        )

        # Put back
        it = iter(finite_norm)
        for local_j, rec_i in enumerate(idxs):
            if finite_mask[local_j]:
                x_norm_out[rec_i] = float(next(it))
                invalid_out[rec_i] = records[rec_i].invalid
            else:
                x_norm_out[rec_i] = 0.0
                invalid_out[rec_i] = True

    # ---- 2) Apply flip AFTER normalization to get x_signed ----
    x_signed_out: List[float] = [0.0] * len(records)
    for i, r in enumerate(records):
        xn = x_norm_out[i]
        if (not r.correct) and reward_cfg.flip_on_incorrect:
            x_signed_out[i] = -xn
        else:
            x_signed_out[i] = xn

    # ---- 3) Softmax weights per rollout (within-rollout across segments) ----
    # Rollout key must not include seg_id. Keep method separate.
    def rollout_key(r: SegmentRecord) -> Tuple:
        return (r.run_id, r.question_idx, r.rollout_id, r.contrib_method, r.split_method, r.score_target)

    rollouts: Dict[Tuple, List[int]] = {}
    for idx, r in enumerate(records):
        rollouts.setdefault(rollout_key(r), []).append(idx)

    weight_out: List[float] = [0.0] * len(records)
    R_out: List[float] = [0.0] * len(records)
    assigned_out: List[float] = [0.0] * len(records)
    tau_out: List[float] = [float(softmax_cfg.tau)] * len(records)

    for rkey, idxs in rollouts.items():
        # Keep segment order stable (softmax input order doesn't matter mathematically,
        # but stable order makes debugging easier)
        idxs_sorted = sorted(idxs, key=lambda i: records[i].seg_id)

        xs = [x_signed_out[i] for i in idxs_sorted]
        masks = None
        if softmax_cfg.mask_invalid:
            masks = [not invalid_out[i] and _is_finite(xs_j) for i, xs_j in zip(idxs_sorted, xs)]

        w = softmax(
            xs,
            tau=softmax_cfg.tau,
            stable=softmax_cfg.stable,
            mask=masks,
        )

        # Rollout scalar reward depends on correctness (same for all segments in rollout)
        # (rollout is defined by question_idx+rollout_id, so correctness is consistent)
        correct = records[idxs_sorted[0]].correct
        R = _rollout_reward_scalar(reward_cfg, correct)

        for local_j, rec_i in enumerate(idxs_sorted):
            weight_out[rec_i] = float(w[local_j])
            R_out[rec_i] = float(R)
            assigned_out[rec_i] = float(R * w[local_j])

    # ---- 4) Emit updated records ----
    out: List[SegmentRecord] = []
    for i, r in enumerate(records):
        out.append(
            replace(
                r,
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
