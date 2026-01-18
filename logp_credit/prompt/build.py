# logp_credit/prompt/build.py
from __future__ import annotations

from logp_credit.config import ExperimentConfig


def build_prompt_text(tok, question: str, cfg: ExperimentConfig) -> str:
    system = cfg.get_system_prompt()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=cfg.prompt.add_generation_prompt,
        enable_thinking=cfg.prompt.enable_thinking,
    )


def strip_prompt_from_decoded(full_decoded: str, prompt_text: str) -> str:
    idx = full_decoded.find(prompt_text)
    if idx >= 0:
        return full_decoded[idx + len(prompt_text):].lstrip()
    return full_decoded
