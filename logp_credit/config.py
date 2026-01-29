# logp_credit/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
import json

import torch


# -------------------------
# Utilities
# -------------------------
def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_dtype_str(dtype: str) -> str:
    d = dtype.strip().lower()
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    if d not in aliases:
        raise ValueError(f"Unknown dtype string: {dtype!r}. Use one of {sorted(aliases)}.")
    return aliases[d]


def detect_dtype(device: str) -> str:
    # Prefer bf16 on CUDA; fall back to fp32 on CPU.
    if device == "cuda":
        return "bfloat16"
    return "float32"


def torch_dtype(dtype: str, device: Optional[str] = None) -> torch.dtype:
    """
    Convert dtype string to torch.dtype. If dtype='auto', device is required.
    """
    d = normalize_dtype_str(dtype)
    if d == "auto":
        if device is None:
            raise ValueError("torch_dtype(dtype='auto') requires device.")
        d = detect_dtype(device)

    if d == "bfloat16":
        return torch.bfloat16
    if d == "float16":
        return torch.float16
    if d == "float32":
        return torch.float32
    raise RuntimeError(f"Unhandled dtype after normalization: {d}")


def clamp_nonneg(name: str, v: int) -> int:
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


# -------------------------
# Config dataclasses
# -------------------------
@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen3-0.6B"

    # Use "auto" to detect at runtime.
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: str = "auto"  # "auto" | "bfloat16" | "float16" | "float32" | aliases: bf16/fp16/fp32

    # Transformers load/runtime knobs
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"
    device_map: Optional[str] = None           # e.g. "auto"
    trust_remote_code: bool = False

    # Optional runtime optimizations
    torch_compile: bool = False
    use_amp: bool = True

    def resolved_device(self) -> str:
        return detect_device() if self.device == "auto" else self.device

    def resolved_dtype_str(self) -> str:
        dev = self.resolved_device()
        d = normalize_dtype_str(self.dtype)
        return detect_dtype(dev) if d == "auto" else d

    def resolved_torch_dtype(self) -> torch.dtype:
        dev = self.resolved_device()
        return torch_dtype(self.dtype, device=dev)


@dataclass(frozen=True)
class PromptConfig:
    # Former prompt/constants.py lives here.
    hash_prefix: str = "#### "     # NOTE: keep trailing space as part of format
    think_open: str = "<think>\n"
    think_close: str = "</think>"

    placeholder: str = "[REMOVED]"

    # If None, your prompt/build.py can generate a default system prompt.
    system_prompt: Optional[str] = None

    add_generation_prompt: bool = True
    enable_thinking: bool = True

    ensure_think_open: bool = False
    max_prompt_tail_check: int = 400

    def default_system_prompt(self) -> str:
        """
        Default system prompt aligned to your current build_prompt_text().
        Kept here so that prompt/build.py can simply call cfg.prompt.default_system_prompt()
        when cfg.prompt.system_prompt is None.
        """
        hp = self.hash_prefix
        return (
            "You are a helpful assistant.\n"
            "Solve the problem step by step.\n"
            f"Put your final answer on its own line in the format: \n{hp} <answer>.\n"
            "Do NOT put the final answer inside <think>.\n"
            "\n"
            "Example:\n"
            "<think>\n"
            "Compute 60% of 5 is 3, so second song is 8, total is 13.\n"
            "</think>\n"
            f"{hp} 13"
        )


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    do_sample: bool = True
    use_cache: bool = True

    # "eos" means use tok.eos_token_id
    pad_token_id_strategy: Literal["eos", "none"] = "eos"

    repetition_penalty: Optional[float] = None
    stop_on_eos: bool = False

    def __post_init__(self):
        clamp_nonneg("max_new_tokens", self.max_new_tokens)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1], got {self.top_p}")
        clamp_nonneg("top_k", self.top_k)
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    split: str = "test"

    question_field: str = "question"
    answer_field: str = "answer"
    id_field: Optional[str] = None

    # If provided, use these indices; otherwise take first max_questions.
    idxs: Optional[List[int]] = None
    max_questions: Optional[int] = None

    def __post_init__(self):
        if self.idxs is not None:
            for i in self.idxs:
                if i < 0:
                    raise ValueError(f"idxs must be non-negative. Got {i}")
        if self.max_questions is not None and self.max_questions <= 0:
            raise ValueError(f"max_questions must be > 0, got {self.max_questions}")


@dataclass(frozen=True)
class SegmentationConfig:
    method: Literal["period"] = "period"
    max_segments: Optional[int] = 64
    truncate_policy: Literal["head", "tail"] = "head"
    keep_empty_think: bool = True

    def __post_init__(self):
        if self.max_segments is not None and self.max_segments <= 0:
            raise ValueError(f"max_segments must be > 0, got {self.max_segments}")


@dataclass(frozen=True)
class ContributionConfig:
    """
    Contribution estimation config.
    NOTE: score_target is fixed to GT answer logP by design.
    """
    method: Literal["prefix", "loo"] = "prefix"

    # LOO-only
    loo_mode: Literal["delete", "placeholder"] = "placeholder"
    max_loo_segments: int = 64

    # Fixed for this project
    score_target: Literal["gt"] = "gt"

    include_empty_prefix: bool = True

    def __post_init__(self):
        clamp_nonneg("max_loo_segments", self.max_loo_segments)


@dataclass(frozen=True)
class NormalizationConfig:
    scope: Literal["rollout", "question", "batch"] = "rollout"
    method: Literal["zscore", "robust", "none"] = "zscore"
    eps: float = 1e-6
    clip: Optional[float] = None
    # If True, compute normalization statistics separately for correct/incorrect groups
    # within the chosen scope. (Normalization only; softmax remains per-rollout.)
    separate_correctness: bool = False

    def __post_init__(self):
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.clip is not None and self.clip <= 0:
            raise ValueError(f"clip must be > 0, got {self.clip}")


@dataclass(frozen=True)
class SoftmaxConfig:
    tau: float = 1.0
    stable: bool = True     # max-shift etc.
    mask_invalid: bool = True

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")


@dataclass(frozen=True)
class RewardConfig:
    reward_correct: float = 1.0
    reward_incorrect: float = -1.0

    # If True: incorrect uses x_signed = -x_norm (i.e., flip AFTER normalization)
    flip_on_incorrect: bool = True

    assignment: Literal["proportional"] = "proportional"  # r_i = R * w_i

    min_reward_clip: Optional[float] = None
    max_reward_clip: Optional[float] = None

    def __post_init__(self):
        if self.min_reward_clip is not None and self.max_reward_clip is not None:
            if self.min_reward_clip > self.max_reward_clip:
                raise ValueError("min_reward_clip must be <= max_reward_clip")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 1
    seed_strategy: Literal["base_plus_offsets"] = "base_plus_offsets"

    n_rollouts_per_question: int = 8

    out_dir: str = "runs/logp_credit"
    run_id: Optional[str] = None  # set at runtime if None

    # Save policy
    save_level: Literal["full", "light", "minimal"] = "light"
    save_prompt: bool = False
    save_decoded: bool = True
    save_think_rest: bool = True

    # Resume/overwrite
    resume: bool = True
    overwrite: bool = False

    # Future parallelization
    num_workers: int = 0

    def __post_init__(self):
        clamp_nonneg("seed", self.seed)
        if self.n_rollouts_per_question <= 0:
            raise ValueError(f"n_rollouts_per_question must be > 0, got {self.n_rollouts_per_question}")
        clamp_nonneg("num_workers", self.num_workers)
        if not self.out_dir:
            raise ValueError("out_dir must be non-empty")


@dataclass(frozen=True)
class PlotConfig:
    enable: bool = False
    per_rollout: bool = True
    per_question: bool = False
    aggregate: bool = False
    max_plots_per_question: int = 4

    def __post_init__(self):
        clamp_nonneg("max_plots_per_question", self.max_plots_per_question)


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    contrib: ContributionConfig = field(default_factory=ContributionConfig)
    norm: NormalizationConfig = field(default_factory=NormalizationConfig)
    softmax: SoftmaxConfig = field(default_factory=SoftmaxConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    run: RunConfig = field(default_factory=RunConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    # -------- convenience ----------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure dtype is normalized for reproducible configs
        d["model"]["dtype"] = self.model.resolved_dtype_str()
        d["model"]["device"] = self.model.resolved_device()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def get_system_prompt(self) -> str:
        return self.prompt.system_prompt if self.prompt.system_prompt is not None else self.prompt.default_system_prompt()
