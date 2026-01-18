# logp_credit/pipeline/runner.py
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logp_credit.config import ExperimentConfig
from logp_credit.schema.records import RolloutMeta, SegmentRecord
from logp_credit.contrib.prefix_marginal import compute_prefix_marginal
from logp_credit.weight.assign import (
    build_segment_records_base_from_rollout,
    finalize_records_two_pass,
)
from logp_credit.io.save import ensure_dir, save_json, save_records_csv

# ---- you already have these in your snippet; put them in appropriate modules if you want ----
from logp_credit.data.gsm8k import load_gsm8k
from logp_credit.prompt.build import build_prompt_text, strip_prompt_from_decoded
from logp_credit.text.parse import parse_think_and_rest
from logp_credit.text.extract import extract_final_after_hashes, extract_last_number


def _now_run_id(prefix: str = "run") -> str:
    # stable-ish run id
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_and_tokenizer(cfg: ExperimentConfig):
    model_cfg = cfg.model
    device = model_cfg.resolved_device()
    torch_dtype = model_cfg.resolved_torch_dtype()

    tok = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=model_cfg.trust_remote_code)

    # common: Qwen系は pad 未設定のことがあるので eos を使う
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=model_cfg.device_map,
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if model_cfg.device_map is None:
        model = model.to(device)

    model.eval()
    return model, tok


def _generate_one(model, tok, prompt_ids: torch.Tensor, cfg: ExperimentConfig) -> str:
    gen = cfg.gen
    pad_id = tok.eos_token_id if gen.pad_token_id_strategy == "eos" else None

    gen_ids = model.generate(
        prompt_ids,
        do_sample=gen.do_sample,
        temperature=gen.temperature,
        top_p=gen.top_p,
        top_k=gen.top_k,
        max_new_tokens=gen.max_new_tokens,
        use_cache=gen.use_cache,
        pad_token_id=pad_id,
        repetition_penalty=gen.repetition_penalty,
    )
    new_ids = gen_ids[:, prompt_ids.size(1):]          # ← 生成部分だけ
    assistant = tok.decode(new_ids[0], skip_special_tokens=True).strip()
    return assistant


def _build_rollout_meta(
    *,
    run_id: str,
    dataset_name: str,
    split: str,
    question_idx: int,
    question_id: Optional[str],
    rollout_id: int,
    seed: int,
    question: Optional[str],
    true_answer_text: Optional[str],
    true_norm: str,
    prompt_text: Optional[str],
    decoded_text: Optional[str],
    think_text: Optional[str],
    rest_text: Optional[str],
    pred_norm: Optional[str],
    pred_extraction: str,
    correct: bool,
) -> RolloutMeta:
    return RolloutMeta(
        run_id=run_id,
        dataset_name=dataset_name,
        split=split,
        question_idx=question_idx,
        question_id=question_id,
        rollout_id=rollout_id,
        seed=seed,
        question=question,
        true_answer_text=true_answer_text,
        true_norm=true_norm,
        prompt_text=prompt_text,
        decoded_text=decoded_text,
        think_text=think_text,
        rest_text=rest_text,
        pred_norm=pred_norm,
        pred_extraction=pred_extraction,  # "hash" / "last_number" / "none"
        correct=correct,
    )


def run_experiment(cfg: ExperimentConfig) -> List[SegmentRecord]:
    # -------- setup --------
    run_id = cfg.run.run_id or _now_run_id("logp_credit")
    out_dir = os.path.join(cfg.run.out_dir, run_id)
    ensure_dir(out_dir)

    # save config for reproducibility
    save_json(cfg.to_dict(), os.path.join(out_dir, "config.json"))

    model, tok = _load_model_and_tokenizer(cfg)

    ds = load_gsm8k(cfg.data.split)

    # decide which indices to run
    if cfg.data.idxs is not None:
        idxs = list(cfg.data.idxs)
    else:
        max_q = cfg.data.max_questions or 1
        idxs = list(range(max_q))

    all_base_records: List[SegmentRecord] = []

    # -------- main loop --------
    for qi, idx in enumerate(idxs):
        ex = ds[idx]
        q = ex[cfg.data.question_field]
        gt_solution = ex[cfg.data.answer_field]

        true_norm = extract_final_after_hashes(gt_solution)
        if true_norm is None:
            # GSM8Kは通常ここは取れるが、念のため
            true_norm = extract_last_number(gt_solution)
        if true_norm is None:
            raise RuntimeError(f"Failed to extract GT final answer for idx={idx}")

        prompt_text = build_prompt_text(tok, q, cfg)  # uses cfg.prompt
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

        # per question rollouts
        for r in range(cfg.run.n_rollouts_per_question):
            seed = cfg.run.seed + (idx * 1000) + r
            _set_seed(seed)

            assistant = _generate_one(model, tok, prompt_ids, cfg)

            # Try to remove prompt text (best-effort) then parse think/rest
            #assistant = strip_prompt_from_decoded(decoded_full, prompt_text)
            think, rest = parse_think_and_rest(assistant)

            # Prediction extraction
            pred_norm = extract_final_after_hashes(rest)
            pred_extraction = "hash"
            if pred_norm is None:
                pred_norm = extract_last_number(rest)
                pred_extraction = "last_number" if pred_norm is not None else "none"

            correct = (pred_norm == true_norm) if (pred_norm is not None) else False

            # Save-level controls (keep it simple here; you can refine later)
            save_level = cfg.run.save_level
            keep_text = (save_level == "full")
            keep_decoded = cfg.run.save_decoded and (save_level in ("full", "light"))
            keep_prompt = cfg.run.save_prompt and (save_level == "full")

            meta = _build_rollout_meta(
                run_id=run_id,
                dataset_name=cfg.data.dataset_name,
                split=cfg.data.split,
                question_idx=idx,
                question_id=None,
                rollout_id=r,
                seed=seed,
                question=q if keep_text else None,
                true_answer_text=gt_solution if keep_text else None,
                true_norm=true_norm,
                prompt_text=prompt_text if keep_prompt else None,
                decoded_text=decoded_full if keep_decoded else None,
                think_text=think if cfg.run.save_think_rest else None,
                rest_text=rest if cfg.run.save_think_rest else None,
                pred_norm=pred_norm,
                pred_extraction=pred_extraction,
                correct=correct,
            )

            # ---- contribution (prefix marginal) ----
            contrib = compute_prefix_marginal(
                model,
                tok,
                prompt_ids=prompt_ids,
                think_text=think,
                true_answer_norm=true_norm,
                prompt_cfg=cfg.prompt,
                seg_cfg=cfg.seg,
                contrib_cfg=cfg.contrib,
            )

            # ---- pass1 records ----
            base_records = build_segment_records_base_from_rollout(meta, contrib)
            all_base_records.extend(base_records)

    # -------- pass2 finalize (scope-aware normalization) --------
    final_records = finalize_records_two_pass(
        all_base_records,
        norm_cfg=cfg.norm,
        softmax_cfg=cfg.softmax,
        reward_cfg=cfg.reward,
    )

    # -------- save --------
    save_records_csv(final_records, os.path.join(out_dir, "segments.csv"))
    # (optional) parquet is nicer for large runs
    # save_records_parquet(final_records, os.path.join(out_dir, "segments.parquet"))

    return final_records
