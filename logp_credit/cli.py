# logp_credit/cli.py
from __future__ import annotations

import argparse
from typing import List, Optional

from logp_credit.config import (
    ExperimentConfig,
    ModelConfig,
    GenConfig,
    DataConfig,
    RunConfig,
    PromptConfig,
    WindowSelectConfig,  # <-- add
    ContributionConfig,
    NormalizationConfig,
    SoftmaxConfig,
    RewardConfig,
)
from logp_credit.pipeline.runner import run_experiment


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _unescape_cli_string(s: Optional[str]) -> Optional[str]:
    """
    Make CLI strings convenient:
      - converts '\\n' -> '\n', '\\t' -> '\t', '\\r' -> '\r'
      - leaves other characters as-is

    Example:
      --hash_prefix "Final answer:\\n#### "
        -> "Final answer:\n#### "
    """
    if s is None:
        return None
    return (
        s.replace("\\n", "\n")
         .replace("\\t", "\t")
         .replace("\\r", "\r")
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--split", type=str, default="test")

    # problems
    p.add_argument("--idxs", type=str, default=None, help="e.g. 0,1,2 (comma-separated)")
    p.add_argument("--max_questions", type=int, default=None, help="use 0..max_questions-1 if --idxs is not set")

    # rollouts
    p.add_argument("--rollouts", type=int, default=2, help="n_rollouts_per_question")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)

    # prompt / scoring marker
    p.add_argument(
        "--hash_prefix",
        type=str,
        default=None,
        help=(
            "Override PromptConfig.hash_prefix. "
            "Use \\n for newline. Example: --hash_prefix \"Final answer:\\n#### \"\n"
            "NOTE: keep trailing space if you want (e.g., '#### '). "
            "Extraction still relies on '####' in outputs."
        ),
    )

    # window selection (new method)
    p.add_argument(
        "--window_tokens",
        type=int,
        default=None,
        help="Token window size for selection (e.g., 64, 128). If omitted, use config default.",
    )
    p.add_argument(
        "--top_k_per_window",
        type=int,
        default=None,
        help="Select top-k segments per token-window. If omitted, use config default.",
    )
    p.add_argument(
        "--window_enabled",
        action="store_true",
        help="Force enable token-window selection (overrides config).",
    )
    p.add_argument(
        "--window_disabled",
        action="store_true",
        help="Force disable token-window selection (overrides config).",
    )

    # seed / output
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="runs/logp_credit")
    p.add_argument("--run_id", type=str, default=None)

    # saving
    p.add_argument("--save_level", type=str, default="light", choices=["minimal", "light", "full"])
    p.add_argument("--save_decoded", action="store_true")
    p.add_argument("--save_think_rest", action="store_true")
    p.add_argument("--save_prompt", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    idxs = _parse_int_list(args.idxs)

    # -------------------------
    # Prompt override
    # -------------------------
    if args.hash_prefix is None or args.hash_prefix.strip() == "":
        prompt_cfg = PromptConfig()
    else:
        hp = _unescape_cli_string(args.hash_prefix)
        prompt_cfg = PromptConfig(hash_prefix=hp)

    # -------------------------
    # Window selection override
    # -------------------------
    base_window = WindowSelectConfig()
    if args.window_enabled and args.window_disabled:
        raise ValueError("Specify only one of --window_enabled or --window_disabled.")

    enabled = base_window.enabled
    if args.window_enabled:
        enabled = True
    if args.window_disabled:
        enabled = False

    window_tokens = base_window.window_tokens if args.window_tokens is None else int(args.window_tokens)
    top_k_per_window = base_window.top_k_per_window if args.top_k_per_window is None else int(args.top_k_per_window)

    if window_tokens <= 0:
        raise ValueError(f"--window_tokens must be > 0, got {window_tokens}")
    if top_k_per_window <= 0:
        raise ValueError(f"--top_k_per_window must be > 0, got {top_k_per_window}")

    window_cfg = WindowSelectConfig(
        enabled=enabled,
        window_tokens=window_tokens,
        top_k_per_window=top_k_per_window,
        preflip_select_metric=base_window.preflip_select_metric,
    )

    cfg = ExperimentConfig(
        model=ModelConfig(
            model_name=args.model,
            attn_implementation=None,  # 入っていれば "flash_attention_2" に変更可
            device_map=None,
            dtype="auto",
        ),
        prompt=prompt_cfg,
        gen=GenConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ),
        data=DataConfig(
            split=args.split,
            idxs=idxs,
            max_questions=args.max_questions,
        ),
        run=RunConfig(
            seed=args.seed,
            n_rollouts_per_question=args.rollouts,
            out_dir=args.out_dir,
            run_id=args.run_id,
            save_level=args.save_level,
            save_decoded=args.save_decoded,
            save_think_rest=args.save_think_rest,
            save_prompt=args.save_prompt,
        ),
        window=window_cfg,  # <-- add
        contrib=ContributionConfig(method="prefix"),
        norm=NormalizationConfig(scope="rollout", method="zscore", clip=None, separate_correctness=False),
        softmax=SoftmaxConfig(tau=1.0, stable=True, mask_invalid=True),
        reward=RewardConfig(reward_correct=1.0, reward_incorrect=-1.0, flip_on_incorrect=True),
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
