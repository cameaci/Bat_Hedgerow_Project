from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any


def read_attribute_table(path: str | Path):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _read_csv_bytes(p.read_bytes(), source_name=p.name)
    if suffix == ".xlsx":
        return _read_xlsx_bytes(p.read_bytes(), source_name=p.name)
    raise ValueError(f"Unsupported table format '{suffix}'. Use .csv or .xlsx.")


def read_uploaded_attribute_table(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    name = str(getattr(uploaded_file, "name", "uploaded"))
    data = uploaded_file.getvalue()
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return _read_csv_bytes(data, source_name=name)
    if suffix == ".xlsx":
        return _read_xlsx_bytes(data, source_name=name)
    raise ValueError(f"Unsupported upload format '{suffix}'. Please upload a CSV or XLSX file.")


def dataframe_to_csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_xlsx_bytes(df) -> bytes:
    buffer = io.BytesIO()
    with _excel_writer(buffer) as writer:
        df.to_excel(writer, index=False, sheet_name="screened_results")
    return buffer.getvalue()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _read_csv_bytes(data: bytes, *, source_name: str):
    import pandas as pd

    tried: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        tried.append(encoding)
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError(f"Could not parse CSV '{source_name}': {exc}") from exc
    raise ValueError(
        f"Could not decode CSV '{source_name}'. Tried encodings: {', '.join(tried)}."
    )


def _read_xlsx_bytes(data: bytes, *, source_name: str):
    import pandas as pd

    try:
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "XLSX support requires openpyxl. Install `openpyxl` and try again."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not parse XLSX '{source_name}': {exc}") from exc


def _excel_writer(buffer: io.BytesIO):
    import pandas as pd

    try:
        return pd.ExcelWriter(buffer, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "XLSX export requires openpyxl. Install `openpyxl` and try again."
        ) from exc

