# logp_credit/text/extract.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Literal


_R_HASH_LINE_MODE = Literal["hash_line", "numeric_last_line", "none", "empty"]

_NUMERIC_ONLY_LINE_RE = re.compile(r"^\s*-?\d[\d,]*\.?\d*\s*$")


@dataclass(frozen=True)
class RationaleSplit:
    """
    rationale_text: answer部分を除いた本文（segment化対象）
    answer_tail:    取り除いた最終回答部分（デバッグ用）
    mode:           どのルールで取り除いたか
    """
    rationale_text: str
    answer_tail: str
    mode: _R_HASH_LINE_MODE


def extract_rationale_text(
    assistant: str,
    *,
    hash_prefix: str = "#### ",
    strip_think_tags: bool = True,
) -> RationaleSplit:
    """
    Build rationale text from assistant output by removing the final answer part.

    Rules (priority):
      1) If a '####' answer line exists, cut at the LAST such line:
           rationale = text_before_last_hash_line
           answer_tail = last_hash_line_and_after
      2) Else, if the last non-empty line is numeric-only (e.g., "13"), drop that last line.
      3) Else, keep the whole text as rationale.

    Notes:
      - hash_prefix may contain trailing space (e.g., "#### "), we match by `hash_prefix.strip()`.
      - Designed for enable_thinking=False, but can optionally strip <think> tags if they appear.
    """
    if not assistant or not assistant.strip():
        return RationaleSplit(rationale_text="", answer_tail="", mode="empty")

    text = assistant.strip()

    # Optional: remove think tags (sometimes models still output them)
    if strip_think_tags:
        # Remove the tags themselves; keep inner content.
        text = re.sub(r"</?think>\s*", "", text).strip()

    # ---- (1) hash line exists: cut at the LAST hash line ----
    marker = hash_prefix.strip()  # "####"
    lines = text.splitlines()

    last_hash_i: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(rf"^\s*{re.escape(marker)}\s*", ln):
            last_hash_i = i

    if last_hash_i is not None:
        rationale = "\n".join(lines[:last_hash_i]).rstrip()
        answer_tail = "\n".join(lines[last_hash_i:]).strip()
        return RationaleSplit(rationale_text=rationale, answer_tail=answer_tail, mode="hash_line")

    # ---- (2) no hash line: drop numeric-only last line (fallback case) ----
    # Remove trailing empty lines first
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return RationaleSplit(rationale_text="", answer_tail="", mode="empty")

    last = lines[-1].strip()
    if _NUMERIC_ONLY_LINE_RE.match(last):
        rationale = "\n".join(lines[:-1]).rstrip()
        answer_tail = last
        return RationaleSplit(rationale_text=rationale, answer_tail=answer_tail, mode="numeric_last_line")

    # ---- (3) otherwise: keep whole text ----
    return RationaleSplit(rationale_text=text, answer_tail="", mode="none")


def normalize_numeric_str(s: str) -> str:
    s = s.strip().replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    if not m:
        return s.strip()
    v = m[-1]
    v = v.lstrip("0") or "0"
    return v


def extract_final_after_hashes(text: str) -> Optional[str]:
    matches = re.findall(r"####\s*([^\n\r]*)", text)
    if not matches:
        return None
    cand = matches[-1].strip()
    if not cand:
        return None
    return normalize_numeric_str(cand)


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    if not nums:
        return None
    last = nums[-1].replace(",", "").strip()
    if last.endswith("."):
        last = last[:-1]
    last = last.lstrip("0") or "0"
    return last
