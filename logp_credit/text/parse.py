# logp_credit/text/parse.py
from __future__ import annotations
import re
from typing import Tuple


def parse_think_and_rest(text: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)$", text, flags=re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = re.search(r"(.*?)</think>(.*)$", text, flags=re.S)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()
    return "", text.strip()
