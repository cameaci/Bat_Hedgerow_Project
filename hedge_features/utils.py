from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha1_bytes(data: bytes, length: int = 16) -> str:
    return hashlib.sha1(data).hexdigest()[:length]


def sha1_text(text: str, length: int = 16) -> str:
    return sha1_bytes(text.encode("utf-8"), length=length)


def stable_json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def dataframe_fingerprint(df, *, length: int = 16) -> str:
    import pandas as pd

    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    hashed = pd.util.hash_pandas_object(normalized, index=True, categorize=False)
    payload = stable_json_dumps(
        {
            "columns": list(normalized.columns),
            "index_name": normalized.index.name,
        }
    ).encode("utf-8") + hashed.to_numpy().tobytes()
    return sha1_bytes(payload, length=length)


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
