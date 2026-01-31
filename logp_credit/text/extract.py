# logp_credit/text/extract.py
from __future__ import annotations
import re
from typing import Optional


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
