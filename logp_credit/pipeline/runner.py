# logp_credit/pipeline/runner.py
from __future__ import annotations

import os
import time
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logp_credit.config import ExperimentConfig
from logp_credit.schema.records import RolloutMeta, SegmentRecord
from logp_credit.contrib.prefix_marginal import compute_prefix_marginal
from logp_credit.weight.assign import (
    build_segment_records_base_from_rollout,
    finalize_records_two_pass,
)
from logp_credit.io.save import ensure_dir, save_json, save_jsonl, save_records_csv

from logp_credit.data.gsm8k import load_gsm8k
from logp_credit.prompt.build import build_prompt_text
from logp_credit.text.parse import parse_think_and_rest
from logp_credit.text.extract import extract_final_after_hashes, extract_last_number


def _now_run_id(prefix: str = "run") -> str:
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

    tok = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # Qwen系は pad 未設定のことがあるので eos を使う
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

    # 生成部分だけ取り出す
    new_ids = gen_ids[:, prompt_ids.size(1):]
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
        pred_extraction=pred_extraction,
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
    questions_map = {}
    rollout_logs = []

    for idx in idxs:
        ex = ds[idx]
        q = ex[cfg.data.question_field]
        gt_solution = ex[cfg.data.answer_field]

        true_norm = extract_final_after_hashes(gt_solution)
        if true_norm is None:
            true_norm = extract_last_number(gt_solution)
        if true_norm is None:
            raise RuntimeError(f"Failed to extract GT final answer for idx={idx}")

        questions_map[str(idx)] = {
            "question": q,
            "true_norm": true_norm,
            "gt_solution": gt_solution,
        }

        prompt_text = build_prompt_text(tok, q, cfg)  # uses cfg.prompt
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)

        # Save-level controls
        save_level = cfg.run.save_level
        keep_text = (save_level == "full")
        keep_decoded = cfg.run.save_decoded and (save_level in ("full", "light"))
        keep_prompt = cfg.run.save_prompt and (save_level == "full")
        keep_think_rest = bool(cfg.run.save_think_rest and (save_level in ("full", "light")))

        # per question rollouts
        for r in range(cfg.run.n_rollouts_per_question):
            seed = cfg.run.seed + (idx * 1000) + r
            _set_seed(seed)

            assistant = _generate_one(model, tok, prompt_ids, cfg)

            # Parse think/rest (thinking 有効前提; ない場合も parse が吸収)
            think, rest = parse_think_and_rest(assistant)

            # Prediction extraction（正誤判定にも last_number 救済を使う方針）
            pred_norm = extract_final_after_hashes(rest)
            pred_extraction = "hash"
            if pred_norm is None:
                pred_norm = extract_last_number(rest)
                pred_extraction = "last_number" if pred_norm is not None else "none"

            correct = (pred_norm == true_norm) if (pred_norm is not None) else False

            # (optional) lightweight rollout log
            rollout_logs.append({
                "run_id": run_id,
                "question_idx": int(idx),
                "rollout_id": int(r),
                "seed": int(seed),
                "correct": bool(correct),
                "true_norm": true_norm,
                "pred_norm": pred_norm,
                "pred_extraction": pred_extraction,
                "assistant": assistant if keep_decoded else None,
                "think": think if keep_think_rest else None,
                "rest": rest if keep_think_rest else None,
            })

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
                decoded_text=assistant if keep_decoded else None,
                think_text=think if keep_think_rest else None,
                rest_text=rest if keep_think_rest else None,
                pred_norm=pred_norm,
                pred_extraction=pred_extraction,
                correct=correct,
            )

            # ---- contribution (prefix marginal) ----
            contrib = compute_prefix_marginal(
                model,
                tok,
                prompt_ids=prompt_ids,
                think_text=think,                 # thinking を segment 対象にする
                true_answer_norm=true_norm,
                prompt_cfg=cfg.prompt,
                seg_cfg=cfg.seg,
                contrib_cfg=cfg.contrib,
            )

            # ---- pass1 records ----
            base_records = build_segment_records_base_from_rollout(meta, contrib)
            all_base_records.extend(base_records)

    # -------- pass2 finalize (normalization + window selection + softmax + reward) --------
    final_records = finalize_records_two_pass(
        all_base_records,
        norm_cfg=cfg.norm,
        softmax_cfg=cfg.softmax,
        reward_cfg=cfg.reward,
        window_cfg=cfg.window,   # ★ 追加：window 選択を有効化
    )

    # -------- save --------
    save_records_csv(final_records, os.path.join(out_dir, "segments.csv"))
    save_json(questions_map, os.path.join(out_dir, "questions.json"))
    save_jsonl(rollout_logs, os.path.join(out_dir, "rollouts.jsonl"))

    return final_records
