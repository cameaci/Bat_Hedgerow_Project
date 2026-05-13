import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point  # noqa: E402

from hedge_features.planning.review import (  # noqa: E402
    apply_review_override,
    build_review_summary,
    get_final_selected_candidates,
    initialise_review_candidates,
)


def _candidates():
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["cand_1", "cand_2", "cand_3", "cand_4"],
            "selected_flag": [1, 1, 0, 0],
            "selection_rank": [1, 2, None, None],
            "planning_status": ["selected", "selected", "eligible_unselected", "ineligible"],
        },
        geometry=[
            Point(0, 0),
            Point(100, 0),
            Point(200, 0),
            Point(300, 0),
        ],
        crs="EPSG:27700",
    )


def test_initialise_review_candidates_maps_auto_selection_and_statuses():
    review_gdf = initialise_review_candidates(_candidates())

    assert review_gdf["final_selected_flag"].astype(int).tolist() == [1, 1, 0, 0]
    assert review_gdf["final_selection_status"].astype(str).tolist() == [
        "auto_selected",
        "auto_selected",
        "eligible_unselected",
        "ineligible",
    ]


def test_apply_review_override_replace_swaps_selected_candidate_and_tracks_audit():
    review_gdf = initialise_review_candidates(_candidates())

    updated_gdf, audit_entry = apply_review_override(
        review_gdf,
        action="replace",
        remove_candidate_id="cand_1",
        add_candidate_id="cand_3",
        rationale="Improve spatial spread near a separate corridor.",
        sequence_no=1,
    )

    final_ids = get_final_selected_candidates(updated_gdf)["candidate_id"].astype(str).tolist()
    assert final_ids == ["cand_3", "cand_2"]
    assert updated_gdf.loc[updated_gdf["candidate_id"] == "cand_1", "final_selection_status"].iloc[0] == "manual_removed"
    assert updated_gdf.loc[updated_gdf["candidate_id"] == "cand_3", "final_selection_status"].iloc[0] == "manual_added"
    assert audit_entry == {
        "sequence_no": 1,
        "action": "replace",
        "removed_candidate_id": "cand_1",
        "added_candidate_id": "cand_3",
        "rationale": "Improve spatial spread near a separate corridor.",
    }


def test_build_review_summary_reports_final_counts_and_budget_gap():
    review_gdf = initialise_review_candidates(_candidates())
    updated_gdf, audit_entry = apply_review_override(
        review_gdf,
        action="remove",
        remove_candidate_id="cand_2",
        add_candidate_id=None,
        rationale="Unsafe mounting location.",
        sequence_no=2,
    )

    summary = build_review_summary(updated_gdf, audit_log=[audit_entry], detector_budget=2)

    assert summary["final_selected_count"] == 1
    assert summary["final_selected_candidate_ids"] == ["cand_1"]
    assert summary["override_count"] == 1
    assert summary["budget_gap"] == 1
    assert summary["final_selection_status_counts"]["manual_removed"] == 1
