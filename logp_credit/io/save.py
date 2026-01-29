# logp_credit/io/save.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from logp_credit.schema.records import SegmentRecord, record_to_dict, segment_record_columns


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def records_to_dataframe(records: List[SegmentRecord]) -> pd.DataFrame:
    rows = [record_to_dict(r) for r in records]
    df = pd.DataFrame(rows)
    # stable column order
    cols = segment_record_columns()
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    return df[cols]


def save_records_csv(records: List[SegmentRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df = records_to_dataframe(records)
    df.to_csv(path, index=False)


def save_records_parquet(records: List[SegmentRecord], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df = records_to_dataframe(records)
    df.to_parquet(path, index=False)
