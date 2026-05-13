from __future__ import annotations


def initialise_review_candidates(candidates_gdf):
    gdf = candidates_gdf.copy()
    defaults = {
        "final_selected_flag": 0,
        "final_selection_rank": None,
        "final_selection_status": "eligible_unselected",
        "review_override_action": "",
        "review_override_rationale": "",
        "review_override_sequence": None,
    }
    for col, default in defaults.items():
        if col not in gdf.columns:
            gdf[col] = default

    for idx, row in gdf.iterrows():
        if int(row.get("selected_flag", 0) or 0) == 1:
            gdf.at[idx, "final_selected_flag"] = 1
            gdf.at[idx, "final_selection_rank"] = row.get("selection_rank")
            gdf.at[idx, "final_selection_status"] = "auto_selected"
        elif str(row.get("planning_status", "")).strip() == "ineligible":
            gdf.at[idx, "final_selection_status"] = "ineligible"
        else:
            gdf.at[idx, "final_selection_status"] = "eligible_unselected"
    return gdf


def get_final_selected_candidates(candidates_gdf):
    selected = candidates_gdf[candidates_gdf["final_selected_flag"].astype(int) == 1].copy()
    if selected.empty:
        return selected
    return selected.sort_values("final_selection_rank", kind="mergesort").reset_index(drop=True)


def build_review_summary(
    candidates_gdf,
    *,
    audit_log: list[dict[str, object]] | None = None,
    detector_budget: int | None = None,
):
    final_selected = get_final_selected_candidates(candidates_gdf)
    status_counts = {
        str(k): int(v)
        for k, v in candidates_gdf["final_selection_status"].astype("string").fillna("NA").value_counts().items()
    }
    summary = {
        "final_selected_count": int(len(final_selected)),
        "final_selected_candidate_ids": final_selected["candidate_id"].astype(str).tolist(),
        "final_selection_status_counts": status_counts,
        "override_count": int(len(audit_log or [])),
        "audit_log": list(audit_log or []),
    }
    if detector_budget is not None:
        summary["detector_budget"] = int(detector_budget)
        summary["budget_gap"] = int(detector_budget) - int(len(final_selected))
    return summary


def apply_review_override(
    candidates_gdf,
    *,
    action: str,
    remove_candidate_id: str | None,
    add_candidate_id: str | None,
    rationale: str,
    sequence_no: int,
):
    gdf = candidates_gdf.copy()
    action = str(action).strip().lower()
    rationale = str(rationale).strip()
    if not rationale:
        raise ValueError("Override rationale is required.")
    if action not in {"replace", "remove"}:
        raise ValueError(f"Unsupported review action '{action}'.")
    if remove_candidate_id is None:
        raise ValueError("A selected candidate must be chosen for removal.")

    remove_idx = _require_candidate_index(gdf, remove_candidate_id)
    if int(gdf.at[remove_idx, "final_selected_flag"] or 0) != 1:
        raise ValueError("The selected candidate to remove is not currently in the final detector set.")

    replacement_rank = int(gdf.at[remove_idx, "final_selection_rank"])
    gdf.at[remove_idx, "final_selected_flag"] = 0
    gdf.at[remove_idx, "final_selection_rank"] = None
    gdf.at[remove_idx, "final_selection_status"] = "manual_removed"
    gdf.at[remove_idx, "review_override_action"] = "remove"
    gdf.at[remove_idx, "review_override_rationale"] = rationale
    gdf.at[remove_idx, "review_override_sequence"] = int(sequence_no)

    if action == "replace":
        if add_candidate_id is None:
            raise ValueError("A replacement candidate must be chosen.")
        add_idx = _require_candidate_index(gdf, add_candidate_id)
        if add_idx == remove_idx:
            raise ValueError("Replacement candidate must differ from the removed candidate.")
        if int(gdf.at[add_idx, "final_selected_flag"] or 0) == 1:
            raise ValueError("Replacement candidate is already selected.")
        gdf.at[add_idx, "final_selected_flag"] = 1
        gdf.at[add_idx, "final_selection_rank"] = replacement_rank
        gdf.at[add_idx, "final_selection_status"] = "manual_added"
        gdf.at[add_idx, "review_override_action"] = "replace_in"
        gdf.at[add_idx, "review_override_rationale"] = rationale
        gdf.at[add_idx, "review_override_sequence"] = int(sequence_no)
    else:
        add_candidate_id = None

    _recompute_final_selection_ranks(gdf)
    audit_entry = {
        "sequence_no": int(sequence_no),
        "action": action,
        "removed_candidate_id": str(remove_candidate_id),
        "added_candidate_id": str(add_candidate_id) if add_candidate_id is not None else None,
        "rationale": rationale,
    }
    return gdf, audit_entry


def _require_candidate_index(gdf, candidate_id: str) -> object:
    matches = gdf.index[gdf["candidate_id"].astype(str) == str(candidate_id)].tolist()
    if not matches:
        raise ValueError(f"Candidate '{candidate_id}' was not found.")
    return matches[0]


def _recompute_final_selection_ranks(gdf) -> None:
    selected = gdf[gdf["final_selected_flag"].astype(int) == 1].copy()
    if selected.empty:
        return
    selected = selected.sort_values(
        by=["final_selection_rank", "selection_rank", "candidate_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    rank = 1
    for idx in selected.index:
        gdf.at[idx, "final_selection_rank"] = int(rank)
        if str(gdf.at[idx, "final_selection_status"]).strip() == "eligible_unselected":
            gdf.at[idx, "final_selection_status"] = "auto_selected"
        rank += 1
