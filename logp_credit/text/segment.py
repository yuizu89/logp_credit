# logp_credit/text/segment.py
from __future__ import annotations

import re
from typing import List


_DOT_PLACEHOLDER = "∯"  # 文中に出にくい文字なら何でもOK


def split_segments_by_period(text: str) -> List[str]:
    """
    Split thinking text into segments by sentence periods, with small heuristics:
      - protect common abbreviations (Dr., e.g., etc.)
      - protect initials like U.S.
      - split on whitespace after '.'
      - merge list-like segments "1." with the next chunk
    """
    text = (text or "").strip()
    if not text:
        return []

    # whitespace normalize
    t = re.sub(r"\s+", " ", text)

    # 1) protect common abbreviations
    for abbr in ["Mr.", "Mrs.", "Ms.", "Dr.", "Inc.", "Ltd.", "e.g.", "i.e."]:
        t = t.replace(abbr, abbr.replace(".", _DOT_PLACEHOLDER))

    # 2) protect initials like U.S. / U.K.
    def _protect_initials(m: re.Match) -> str:
        return m.group(0).replace(".", _DOT_PLACEHOLDER)

    t = re.sub(r"\b(?:[A-Z]\.){2,}", _protect_initials, t)

    # 3) split: keep period in the segment
    parts = re.split(r"(?<=\.)\s+", t)

    # 4) restore placeholders
    parts = [p.replace(_DOT_PLACEHOLDER, ".").strip() for p in parts if p.strip()]

    # 5) merge list-number-only segments like "1."
    merged: List[str] = []
    i = 0
    while i < len(parts):
        seg = parts[i]
        if re.fullmatch(r"\(?\d+\)?\.", seg) and i + 1 < len(parts):
            merged.append(seg + " " + parts[i + 1])
            i += 2
        else:
            merged.append(seg)
            i += 1

    return merged
