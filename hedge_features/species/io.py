from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import ensure_parent_dir


def write_species_artifacts(
    training_result,
    *,
    output_dir: str | Path,
    framework_dir: str | Path | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    ensure_parent_dir(out_dir / "placeholder")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(str(training_result.summary["species_name"]))

    model_path = out_dir / f"species_{slug}_model.json"
    model_card_path = out_dir / f"species_{slug}_model_card.json"
    model_card_md_path = out_dir / f"species_{slug}_model_card.md"
    domain_path = out_dir / f"species_{slug}_domain.json"
    summary_path = out_dir / f"species_{slug}_training_summary.json"

    training_result.model_artifact["model_card_artifact"] = model_card_path.name
    training_result.model_artifact["domain_artifact"] = domain_path.name

    model_path.write_text(json.dumps(training_result.model_artifact, indent=2), encoding="utf-8")
    model_card_path.write_text(json.dumps(training_result.model_card, indent=2), encoding="utf-8")
    model_card_md_path.write_text(_render_model_card_markdown(training_result.model_card), encoding="utf-8")
    domain_path.write_text(json.dumps(training_result.domain_of_applicability, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(training_result.summary, indent=2), encoding="utf-8")

    if framework_dir is not None:
        _update_framework_manifest_species_targets(Path(framework_dir), species_name=str(training_result.summary["species_name"]))

    return {
        "species_model": str(model_path),
        "model_card_json": str(model_card_path),
        "model_card_md": str(model_card_md_path),
        "domain_of_applicability": str(domain_path),
        "training_summary": str(summary_path),
    }


def _update_framework_manifest_species_targets(framework_dir: Path, *, species_name: str) -> None:
    manifest_path = framework_dir / "framework_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = {str(x) for x in (payload.get("species_targets_available") or [])}
    targets.add(str(species_name))
    payload["species_targets_available"] = sorted(targets)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_model_card_markdown(model_card: dict[str, Any]) -> str:
    lines = [
        f"# Species Model Card: {model_card.get('species_name', '')}",
        "",
        f"- Framework: `{model_card.get('framework_name', '')}` `{model_card.get('framework_version', '')}`",
        f"- Feature profile: `{model_card.get('compatible_feature_profile_name', '')}`",
        "",
        "## Training Data",
        "",
        f"- Rows: {model_card.get('training_data', {}).get('row_count')}",
        f"- Positive rows: {model_card.get('training_data', {}).get('positive_count')}",
        f"- Positive rate: {model_card.get('training_data', {}).get('positive_rate')}",
        "",
        "## Fit Metrics",
        "",
        "```json",
        json.dumps(model_card.get("fit_metrics", {}), indent=2),
        "```",
        "",
        "## Cross Validation",
        "",
        "```json",
        json.dumps(model_card.get("cross_validation", {}), indent=2),
        "```",
        "",
        "## Geography Holdout",
        "",
        "```json",
        json.dumps(model_card.get("geography_holdout", {}), indent=2),
        "```",
        "",
        "## Limitations",
        "",
    ]
    for item in model_card.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _slugify(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "species"
