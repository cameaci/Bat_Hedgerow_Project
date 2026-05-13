from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load feature profiles.") from exc


PACKAGE_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


def _load_text_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise InputValidationError(f"Unsupported profile format: {path.suffix}")
    if not isinstance(data, dict):
        raise InputValidationError(f"Profile must be a mapping object: {path}")
    return data


def resolve_profile(profile_name: str, profile_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    if profile_path is not None:
        path = Path(profile_path)
        if not path.exists():
            raise InputValidationError(f"Profile file not found: {path}")
        return _load_text_file(path), path

    candidates = [
        PACKAGE_PROFILE_DIR / f"{profile_name}.yaml",
        PACKAGE_PROFILE_DIR / f"{profile_name}.yml",
        PACKAGE_PROFILE_DIR / f"{profile_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_text_file(candidate), candidate
    raise InputValidationError(
        f"Profile '{profile_name}' not found. Looked in {PACKAGE_PROFILE_DIR}"
    )

