from __future__ import annotations

import io
from pathlib import Path


def read_acoustic_table(path: str | Path):
    """Read an acoustic detections table from CSV or XLSX."""
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        data = p.read_bytes()
        tried: list[str] = []
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            tried.append(encoding)
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ValueError(f"Could not parse acoustic CSV '{p.name}': {exc}") from exc
        raise ValueError(f"Could not decode acoustic CSV '{p.name}'. Tried encodings: {', '.join(tried)}.")
    if suffix == ".xlsx":
        try:
            return pd.read_excel(p, engine="openpyxl")
        except ImportError as exc:
            raise RuntimeError("XLSX acoustic import requires openpyxl.") from exc
        except Exception as exc:
            raise ValueError(f"Could not parse acoustic XLSX '{p.name}': {exc}") from exc
    raise ValueError(f"Unsupported acoustic table format '{suffix}'. Use .csv or .xlsx.")
