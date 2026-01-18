import re
from typing import Tuple

def parse_think_and_rest(text: str) -> Tuple[str, str]:
    # last <think>...</think>
    ms = list(re.finditer(r"<think>(.*?)</think>", text, flags=re.S))
    if ms:
        m = ms[-1]
        think = m.group(1).strip()
        rest  = text[m.end():].strip()
        return think, rest

    # fallback: last </think>
    j = text.rfind("</think>")
    if j >= 0:
        think = text[:j].strip()
        rest  = text[j+len("</think>"):].strip()
        return think, rest

    return "", text.strip()