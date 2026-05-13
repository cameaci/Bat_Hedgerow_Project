from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .io import prepare_working_gdf, read_input_geodata
from .planning import (
    PlanningSettings,
    TARGET_SPECS,
    apply_review_override,
    build_review_summary,
    get_final_selected_candidates,
    initialise_review_candidates,
    plan_static_detectors,
    write_planning_evidence_pack,
)
from .screening.io import dataframe_to_csv_bytes, dataframe_to_xlsx_bytes
from .ui_shell import render_info_card, render_page_hero, render_section_intro, render_side_note, render_step_chips


GEOSPATIAL_UPLOAD_TYPES = ["zip", "gpkg", "geojson", "json", "shp"]
PLANNER_ROUTE_OPTIONS = ["Upload enriched geospatial dataset", "Use existing GIS enrichment route output"]
TARGET_SCENARIO_OPTIONS = [("all_bats", "All bats")] + [
    (str(spec["slug"]), str(spec["label"]))
    for spec in TARGET_SPECS
]
STATUS_COLORS = {
    "auto_selected": [30, 92, 179, 210],
    "manual_added": [20, 140, 76, 220],
    "manual_removed": [196, 44, 44, 220],
    "eligible_unselected": [120, 125, 128, 140],
    "ineligible": [170, 170, 170, 80],
}


def render_planner_tab(st) -> None:
    render_page_hero(
        st,
        eyebrow="Step 3",
        title="Static Detector Planner",
        subtitle=(
            "Generate candidate detector locations, optimise for coverage and representation, review the result on a map, "
            "apply expert overrides, and export the final evidence pack."
        ),
    )
    render_step_chips(
        st,
        steps=["1. Source", "2. Configure", "3. Review Map", "4. Expert Override", "5. Evidence Pack"],
        current_step="1. Source",
    )

    top_cols = st.columns([2, 1, 1])
    default_route = st.session_state.get("planner_prefill_route", PLANNER_ROUTE_OPTIONS[0])
    route_index = PLANNER_ROUTE_OPTIONS.index(default_route) if default_route in PLANNER_ROUTE_OPTIONS else 0
    input_route = top_cols[0].radio(
        "Choose input route",
        PLANNER_ROUTE_OPTIONS,
        index=route_index,
        key="planner_input_route",
    )
    working_crs = top_cols[1].text_input("Working CRS", value="EPSG:27700", key="planner_working_crs")
    input_crs = top_cols[2].text_input("Input CRS (only if missing)", value="", key="planner_input_crs")

    source_payload = _resolve_planner_source(
        st=st,
        input_route=input_route,
        working_crs=working_crs,
        input_crs=input_crs.strip() or None,
    )

    if source_payload is not None:
        context_cols = st.columns([2, 1])
        with context_cols[0]:
            render_section_intro(
                st,
                title="Planner Workflow",
                subtitle=(
                    "The planner is designed as a reviewable sequence rather than a black box: choose the source, configure detector rules, "
                    "review spatial outputs, document overrides, then export the evidence pack."
                ),
            )
        with context_cols[1]:
            render_info_card(
                st,
                title="Current Source",
                body=str(source_payload.get("source_name", "No source loaded")),
                chips=[
                    f"Route: {source_payload.get('route', '')}",
                    f"Profile: {source_payload.get('profile_name', '')}",
                ],
            )
    else:
        render_side_note(
            st,
            "The planner works best when you reuse the GIS Enrichment output from the current session. That keeps CRS, feature profile, and metadata aligned."
        )

    tab_setup, tab_map, tab_opt, tab_review, tab_exports = st.tabs(
        ["1. Source & Settings", "2. Candidate Map", "3. Optimisation", "4. Expert Review", "5. Evidence Pack"]
    )

    with tab_setup:
        _render_project_setup(st, source_payload=source_payload)

    last_run = st.session_state.get("planner_last_run")
    if last_run is None:
        with tab_map:
            st.info("Run the planner from Project Setup to review candidates on the map.")
        with tab_opt:
            st.info("Planner optimisation outputs will appear here after a run.")
        with tab_review:
            st.info("Expert review becomes available after a planner run.")
        with tab_exports:
            st.info("Reviewed planner exports become available after a planner run.")
        return

    review_state = _ensure_review_state(st, last_run=last_run)
    reviewed_candidates = review_state["candidates_gdf"]
    review_summary = build_review_summary(
        reviewed_candidates,
        audit_log=review_state["audit_log"],
        detector_budget=last_run["settings"].detector_budget,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidates", int(len(last_run["candidates_gdf"])))
    metric_cols[1].metric("Auto-selected", int(len(last_run["selected_gdf"])))
    metric_cols[2].metric("Final selected", int(review_summary["final_selected_count"]))
    metric_cols[3].metric("Overrides", int(review_summary["override_count"]))

    with tab_map:
        _render_candidate_map(
            st,
            source_gdf=last_run["source_gdf"],
            candidates_gdf=reviewed_candidates,
            include_area_gdf=last_run.get("include_area_gdf"),
            exclude_area_gdf=last_run.get("exclude_area_gdf"),
        )
    with tab_opt:
        _render_optimisation_tab(
            st,
            last_run=last_run,
            reviewed_candidates=reviewed_candidates,
            review_summary=review_summary,
        )
    with tab_review:
        _render_review_tab(
            st,
            last_run=last_run,
            review_state=review_state,
            review_summary=review_summary,
        )
    with tab_exports:
        _render_exports_tab(
            st,
            last_run=last_run,
            reviewed_candidates=reviewed_candidates,
            review_summary=review_summary,
        )


def _resolve_planner_source(*, st, input_route: str, working_crs: str, input_crs: str | None) -> dict[str, Any] | None:
    st.session_state["planner_prefill_route"] = input_route
    if input_route == PLANNER_ROUTE_OPTIONS[0]:
        uploaded = st.file_uploader(
            "Upload enriched geospatial source",
            type=GEOSPATIAL_UPLOAD_TYPES,
            key="planner_upload_geospatial",
        )
        if uploaded is None:
            return None
        try:
            return _read_uploaded_planner_source(
                uploaded=uploaded,
                working_crs=working_crs,
                input_crs=input_crs,
            )
        except Exception as exc:
            st.error(f"Could not read planner source: {exc}")
            return None

    inapp = st.session_state.get("planner_inapp")
    if not inapp:
        st.warning("No in-app enrichment output is available yet. Run GIS Enrichment first or upload a geospatial source.")
        return None

    try:
        source_gdf, _, repaired_count = prepare_working_gdf(
            inapp["gdf"].copy(),
            working_crs=working_crs,
            id_column="hf_uid",
        )
    except Exception as exc:
        st.error(f"Could not prepare the in-app enrichment output for planning: {exc}")
        return None

    notes = []
    if repaired_count:
        notes.append(f"Repaired {repaired_count} invalid hedgerow geometries before planning.")
    st.caption("Using enriched geospatial output cached from the current Streamlit session.")
    return {
        "route": "in_app_enrichment",
        "source_name": str(inapp.get("source_name", "enriched.gpkg")),
        "gdf": source_gdf,
        "notes": notes,
        "profile_name": str(inapp.get("profile_name", "bats_v1")),
        "base_metadata": inapp.get("base_metadata"),
    }


def _read_uploaded_planner_source(*, uploaded, working_crs: str, input_crs: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hedge_features_planner_src_") as tmp_dir:
        temp_path = Path(tmp_dir) / str(uploaded.name)
        temp_path.write_bytes(uploaded.getvalue())
        raw_gdf, notes = read_input_geodata(temp_path, input_crs=input_crs)
        source_gdf, _, repaired_count = prepare_working_gdf(raw_gdf, working_crs=working_crs, id_column="hf_uid")
    if repaired_count:
        notes = list(notes) + [f"Repaired {repaired_count} invalid hedgerow geometries before planning."]
    return {
        "route": "uploaded_enriched_geodata",
        "source_name": str(uploaded.name),
        "gdf": source_gdf,
        "notes": list(notes),
        "profile_name": "bats_bankable_england_v2",
        "base_metadata": None,
    }


def _render_project_setup(st, *, source_payload: dict[str, Any] | None) -> None:
    if source_payload is None:
        st.info("Load an enriched hedgerow geospatial source to configure planning.")
        return

    source_gdf = source_payload["gdf"]
    geometry_name = source_gdf.geometry.name
    non_geom_columns = [col for col in source_gdf.columns if col != geometry_name]
    numeric_columns = _numeric_columns(source_gdf, ignore=[geometry_name])

    st.markdown("**Source Preview**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hedgerows", len(source_gdf))
    c2.metric("Columns", len(non_geom_columns))
    c3.metric("CRS", str(source_gdf.crs))
    c4.metric("Profile", str(source_payload.get("profile_name", "bats_v1")))
    st.caption(f"Source: {source_payload['source_name']}")
    metadata = source_payload.get("base_metadata") or {}
    feature_health_summary = (metadata.get("feature_health") or {}).get("summary", {})
    data_catalogue_summary = (metadata.get("data_catalogue") or {}).get("summary", {})
    with st.expander("Data readiness summary", expanded=False):
        st.json(
            {
                "guidance_regime_version": metadata.get("guidance_regime_version"),
                "dataset_catalogue_summary": data_catalogue_summary,
                "feature_health_summary": feature_health_summary,
            }
        )
    if source_payload.get("notes"):
        for note in source_payload["notes"]:
            st.caption(f"Note: {note}")
    st.dataframe(source_gdf.drop(columns=geometry_name).head(25), use_container_width=True)

    st.markdown("**Planner Settings**")
    col_a, col_b, col_c, col_d = st.columns(4)
    detector_budget = col_a.number_input(
        "Detector budget",
        min_value=1,
        max_value=500,
        value=12,
        step=1,
        key="planner_detector_budget",
    )
    candidate_spacing = col_b.number_input(
        "Candidate spacing (m)",
        min_value=10.0,
        max_value=2000.0,
        value=100.0,
        step=10.0,
        key="planner_candidate_spacing",
    )
    endpoint_offset = col_c.number_input(
        "Endpoint offset (m)",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=5.0,
        key="planner_endpoint_offset",
    )
    min_detector_spacing = col_d.number_input(
        "Min detector spacing (m)",
        min_value=0.0,
        max_value=5000.0,
        value=150.0,
        step=10.0,
        key="planner_min_detector_spacing",
    )

    col_e, col_f, col_g, col_h = st.columns(4)
    use_evidence_engine = col_e.checkbox(
        "Use planner evidence engine",
        value=True,
        key="planner_use_evidence_engine",
    )
    score_override = col_f.selectbox(
        "Score override column",
        options=["<auto>"] + numeric_columns,
        index=0,
        key="planner_score_override",
        help="Leave on <auto> to use the planner's own ecological evidence and utility score.",
    )
    access_flag_column = col_g.selectbox(
        "Access flag column",
        options=["<none>"] + non_geom_columns,
        index=0,
        key="planner_access_flag_column",
    )
    section_column = col_h.selectbox(
        "Section / route column",
        options=["<none>"] + non_geom_columns,
        index=0,
        key="planner_section_column",
    )

    scenario_labels = {slug: label for slug, label in TARGET_SCENARIO_OPTIONS}
    scenario_options = [slug for slug, _ in TARGET_SCENARIO_OPTIONS]
    target_scenario = st.selectbox(
        "Target scenario",
        options=scenario_options,
        index=0,
        format_func=lambda slug: scenario_labels.get(str(slug), str(slug)),
        key="planner_target_scenario",
        help="Choose All bats, a guild, or a priority species scenario for deterministic detector placement.",
    )

    rule_cols = st.columns(3)
    min_score_enabled = rule_cols[0].checkbox(
        "Apply minimum candidate score threshold",
        value=False,
        key="planner_min_score_enabled",
    )
    reject_overlit = rule_cols[1].checkbox(
        "Reject over-lit corridors",
        value=True,
        key="planner_reject_overlit",
    )
    reject_low_confidence = rule_cols[2].checkbox(
        "Reject low-confidence candidates",
        value=False,
        key="planner_reject_low_confidence",
    )
    min_score = None
    if min_score_enabled:
        min_score = st.number_input(
            "Minimum candidate score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            key="planner_min_score",
        )

    section_minimum_counts: dict[str, int] = {}
    if section_column != "<none>":
        unique_sections = _unique_string_values(source_gdf[section_column])
        if unique_sections:
            with st.expander("Section minimum quotas", expanded=False):
                selected_sections = st.multiselect(
                    "Sections with minimum quotas",
                    options=unique_sections,
                    default=[],
                    key="planner_section_min_targets",
                )
                if selected_sections:
                    quota_cols = st.columns(3)
                    for idx, section_name in enumerate(selected_sections):
                        quota = quota_cols[idx % 3].number_input(
                            f"Minimum detectors for {section_name}",
                            min_value=0,
                            max_value=int(detector_budget),
                            value=1,
                            step=1,
                            key=f"planner_section_min_{idx}",
                        )
                        if quota > 0:
                            section_minimum_counts[str(section_name)] = int(quota)

    with st.expander("Spatial constraints", expanded=False):
        st.caption("These polygons can be used to clip planning to the project area and block known exclusion zones.")
        include_upload = st.file_uploader(
            "Include area (optional polygon dataset)",
            type=GEOSPATIAL_UPLOAD_TYPES,
            key="planner_include_area_upload",
        )
        exclude_upload = st.file_uploader(
            "Exclude area (optional polygon dataset)",
            type=GEOSPATIAL_UPLOAD_TYPES,
            key="planner_exclude_area_upload",
        )

    if st.button("Run static detector planner", type="primary", key="planner_run_button"):
        include_area_gdf = None
        exclude_area_gdf = None
        try:
            if include_upload is not None:
                include_area_gdf = _read_uploaded_area(
                    uploaded=include_upload,
                    input_crs=st.session_state.get("planner_input_crs") or None,
                )
            if exclude_upload is not None:
                exclude_area_gdf = _read_uploaded_area(
                    uploaded=exclude_upload,
                    input_crs=st.session_state.get("planner_input_crs") or None,
                )
        except Exception as exc:
            st.error(f"Could not read a spatial constraint layer: {exc}")
            return

        settings = PlanningSettings(
            detector_budget=int(detector_budget),
            candidate_spacing_m=float(candidate_spacing),
            endpoint_offset_m=float(endpoint_offset),
            min_detector_spacing_m=float(min_detector_spacing),
            use_evidence_engine=bool(use_evidence_engine),
            score_column=None if score_override == "<auto>" else score_override,
            target_scenario=str(target_scenario),
            guidance_regime_version=str(metadata.get("guidance_regime_version", "bct4_ne2025_england_v1")),
            min_score=float(min_score) if min_score is not None else None,
            access_flag_column=None if access_flag_column == "<none>" else access_flag_column,
            section_column=None if section_column == "<none>" else section_column,
            section_minimum_counts=section_minimum_counts,
            reject_overlit_candidates=bool(reject_overlit),
            reject_low_confidence_candidates=bool(reject_low_confidence),
            deterministic_output=True,
        )

        try:
            result = plan_static_detectors(
                source_gdf,
                settings=settings,
                include_area_gdf=include_area_gdf,
                exclude_area_gdf=exclude_area_gdf,
                hedge_id_column="hf_uid",
            )
        except Exception as exc:
            st.error(f"Planner run failed: {exc}")
            return

        st.session_state["planner_last_run"] = {
            "source_route": source_payload["route"],
            "source_name": source_payload["source_name"],
            "source_metadata": source_payload.get("base_metadata"),
            "source_gdf": source_gdf.copy(),
            "include_area_gdf": include_area_gdf.copy() if include_area_gdf is not None else None,
            "exclude_area_gdf": exclude_area_gdf.copy() if exclude_area_gdf is not None else None,
            "settings": settings,
            "candidates_gdf": result.candidates_gdf,
            "selected_gdf": result.selected_gdf,
            "run_summary": result.run_summary,
        }
        st.session_state["planner_review"] = {
            "candidates_gdf": initialise_review_candidates(result.candidates_gdf),
            "audit_log": [],
        }
        st.success("Planner run complete. Open Candidate Map, Optimisation, Expert Review, and Exports to continue.")
        st.json(result.run_summary)


def _read_uploaded_area(*, uploaded, input_crs: str | None):
    with tempfile.TemporaryDirectory(prefix="hedge_features_planner_area_") as tmp_dir:
        temp_path = Path(tmp_dir) / str(uploaded.name)
        temp_path.write_bytes(uploaded.getvalue())
        area_gdf, _ = read_input_geodata(temp_path, input_crs=input_crs)
        return area_gdf


def _ensure_review_state(st, *, last_run: dict[str, Any]) -> dict[str, Any]:
    review_state = st.session_state.get("planner_review")
    if review_state is None:
        review_state = {
            "candidates_gdf": initialise_review_candidates(last_run["candidates_gdf"]),
            "audit_log": [],
        }
        st.session_state["planner_review"] = review_state
    return review_state


def _render_candidate_map(st, *, source_gdf, candidates_gdf, include_area_gdf=None, exclude_area_gdf=None) -> None:
    status_values = _unique_string_values(candidates_gdf["final_selection_status"])
    default_statuses = [status for status in status_values if status != "ineligible"] or status_values
    selected_statuses = st.multiselect(
        "Candidate statuses on map",
        options=status_values,
        default=default_statuses,
        key="planner_map_status_filter",
    )
    filtered = candidates_gdf[candidates_gdf["final_selection_status"].astype(str).isin(selected_statuses)].copy()

    col_map, col_side = st.columns([3, 2])
    with col_map:
        _render_pydeck_map(
            st,
            source_gdf=source_gdf,
            candidates_gdf=filtered,
            include_area_gdf=include_area_gdf,
            exclude_area_gdf=exclude_area_gdf,
        )
    with col_side:
        st.markdown("**Legend**")
        st.write(
            {
                "auto_selected": "Automatic planner selection",
                "manual_added": "Added during expert review",
                "manual_removed": "Removed during expert review",
                "eligible_unselected": "Eligible candidate not in final set",
                "ineligible": "Blocked by constraints",
            }
        )
        st.metric("Mapped candidates", len(filtered))
        st.metric("Final selected", int((filtered["final_selected_flag"].astype(int) == 1).sum()))
        _render_candidate_inspection(st, candidates_gdf=candidates_gdf)


def _render_pydeck_map(st, *, source_gdf, candidates_gdf, include_area_gdf=None, exclude_area_gdf=None) -> None:
    try:
        import pydeck as pdk
    except Exception as exc:
        st.warning(f"Map rendering is unavailable because pydeck could not be imported: {exc}")
        return

    source_wgs84 = source_gdf.to_crs("EPSG:4326")
    candidates_wgs84 = candidates_gdf.to_crs("EPSG:4326").copy()
    candidates_wgs84["longitude"] = candidates_wgs84.geometry.x
    candidates_wgs84["latitude"] = candidates_wgs84.geometry.y
    candidates_wgs84["point_color"] = candidates_wgs84["final_selection_status"].map(STATUS_COLORS).apply(
        lambda value: value if isinstance(value, list) else [120, 125, 128, 140]
    )
    candidates_wgs84["point_radius"] = candidates_wgs84["final_selected_flag"].astype(int).map({1: 24.0, 0: 12.0})
    candidates_wgs84["hover_score"] = candidates_wgs84.get(
        "planning_priority_score",
        candidates_wgs84["candidate_score"],
    ).astype(float).round(3)
    candidates_wgs84["hover_confidence"] = candidates_wgs84.get("evidence_confidence_level", "").astype(str)
    candidates_wgs84["hover_guild"] = candidates_wgs84.get("eco_primary_guild", "").astype(str)
    candidates_wgs84["hover_target"] = candidates_wgs84.get("planning_target_label", "").astype(str)
    candidates_wgs84["hover_domain"] = candidates_wgs84.get("planning_target_domain_status", "").astype(str)
    candidates_wgs84["hover_reason"] = (
        candidates_wgs84.get("review_override_rationale", "")
        .astype(str)
        .replace({"nan": "", "None": ""})
    )

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            data=source_wgs84.to_json(),
            pickable=False,
            stroked=True,
            filled=False,
            get_line_color=[104, 94, 84, 150],
            get_line_width=3,
        )
    ]
    if include_area_gdf is not None and len(include_area_gdf) > 0:
        include_wgs84 = include_area_gdf.to_crs("EPSG:4326")
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=include_wgs84.to_json(),
                pickable=False,
                stroked=True,
                filled=True,
                get_line_color=[34, 139, 34, 160],
                get_fill_color=[34, 139, 34, 28],
                get_line_width=2,
            )
        )
    if exclude_area_gdf is not None and len(exclude_area_gdf) > 0:
        exclude_wgs84 = exclude_area_gdf.to_crs("EPSG:4326")
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=exclude_wgs84.to_json(),
                pickable=False,
                stroked=True,
                filled=True,
                get_line_color=[196, 44, 44, 180],
                get_fill_color=[196, 44, 44, 32],
                get_line_width=2,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=candidates_wgs84,
            get_position=["longitude", "latitude"],
            get_fill_color="point_color",
            get_radius="point_radius",
            radius_units="meters",
            stroked=True,
            get_line_color=[40, 40, 40, 120],
            line_width_min_pixels=1,
            pickable=True,
        )
    )

    min_x, min_y, max_x, max_y = source_wgs84.total_bounds
    center_x = float((min_x + max_x) / 2.0)
    center_y = float((min_y + max_y) / 2.0)
    zoom = _bounds_zoom(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
    deck = pdk.Deck(
        map_style=None,
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_y, longitude=center_x, zoom=zoom, pitch=0),
        tooltip={
            "html": (
                "<b>{candidate_id}</b><br/>"
                "Hedge: {source_hf_uid}<br/>"
                "Status: {final_selection_status}<br/>"
                "Priority: {hover_score}<br/>"
                "Target: {hover_target}<br/>"
                "Guild: {hover_guild}<br/>"
                "Domain: {hover_domain}<br/>"
                "Confidence: {hover_confidence}<br/>"
                "Review note: {hover_reason}"
            ),
            "style": {"backgroundColor": "#1f2933", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


def _render_candidate_inspection(st, *, candidates_gdf) -> None:
    st.markdown("**Candidate Inspection**")
    if len(candidates_gdf) == 0:
        st.info("No candidates available.")
        return

    options = candidates_gdf["candidate_id"].astype(str).tolist()
    candidate_id = st.selectbox("Inspect candidate", options=options, key="planner_inspect_candidate_id")
    row = candidates_gdf.loc[candidates_gdf["candidate_id"].astype(str) == str(candidate_id)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", str(row.get("final_selection_status", "")))
    c2.metric("Priority", f"{float(row.get('planning_priority_score', row.get('candidate_score', 0.0))):.3f}")
    c3.metric("Guild", str(row.get("eco_primary_guild", "unknown")))
    c4.metric("Target", str(row.get("planning_target_label", "All bats")))

    st.write(
        {
            "source_hf_uid": str(row.get("source_hf_uid", "")),
            "candidate_chainage_m": float(row.get("candidate_chainage_m", 0.0)),
            "selection_rank": _safe_int(row.get("selection_rank")),
            "final_selection_rank": _safe_int(row.get("final_selection_rank")),
            "planning_status": str(row.get("planning_status", "")),
            "constraint_reason_codes": str(row.get("constraint_reason_codes", "")),
            "planning_target_model_status": str(row.get("planning_target_model_status", "")),
            "planning_target_domain_status": str(row.get("planning_target_domain_status", "")),
            "planning_target_reason_codes": str(row.get("planning_target_reason_codes", "")),
            "data_quality_state": str(row.get("data_quality_state", "")),
            "evidence_confidence_level": str(row.get("evidence_confidence_level", "")),
            "evidence_reason_codes": str(row.get("evidence_reason_codes", "")),
            "review_override_rationale": str(row.get("review_override_rationale", "")),
        }
    )

    with st.expander("Candidate record", expanded=False):
        st.write(_row_snapshot(row))


def _render_optimisation_tab(st, *, last_run: dict[str, Any], reviewed_candidates, review_summary: dict[str, Any]) -> None:
    import pandas as pd

    auto_selected = last_run["selected_gdf"]
    run_summary = last_run["run_summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", int(run_summary.get("candidate_count", len(reviewed_candidates))))
    c2.metric(
        "Eligible",
        int(run_summary.get("eligible_count", (reviewed_candidates["eligible_for_selection"].astype(int) == 1).sum())),
    )
    c3.metric("Auto-selected", int(run_summary.get("selected_count", len(auto_selected))))
    c4.metric("Final selected", int(review_summary["final_selected_count"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Overrides", int(review_summary["override_count"]))
    c6.metric("Budget gap", int(review_summary.get("budget_gap", 0)))
    c7.metric(
        "Route units covered",
        int(auto_selected["optimization_route_unit"].astype("string").nunique()) if len(auto_selected) else 0,
    )
    c8.metric(
        "High-risk corridors selected",
        int((auto_selected["optimization_high_risk_flag"].astype(int) == 1).sum()) if len(auto_selected) else 0,
    )

    info_cols = st.columns(2)
    info_cols[0].metric("Target scenario", str(run_summary.get("planning_target_scenario", "all_bats")))
    info_cols[1].metric("Guidance regime", str(run_summary.get("guidance_regime_version", "n/a")))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Final selection statuses**")
        status_df = pd.DataFrame(
            [{"status": key, "count": int(value)} for key, value in review_summary["final_selection_status_counts"].items()]
        )
        if not status_df.empty:
            st.bar_chart(status_df.set_index("status")["count"])
        else:
            st.info("No status counts available.")
    with col_b:
        st.markdown("**Selected primary guilds**")
        guild_counts = run_summary.get("optimizer", {}).get("selected_primary_guild_counts", {})
        guild_df = pd.DataFrame([{"guild": key, "count": int(value)} for key, value in guild_counts.items()])
        if not guild_df.empty:
            st.bar_chart(guild_df.set_index("guild")["count"])
        else:
            st.info("No guild counts available.")

    st.markdown("**Automatic selection summary**")
    optimizer_summary = run_summary.get("optimizer", {})
    if optimizer_summary:
        st.json(optimizer_summary)
    else:
        st.info("Optimizer summary is not available for this run.")

    st.markdown("**Automatic detector set**")
    preview_cols = [
        "selection_rank",
        "candidate_id",
        "source_hf_uid",
        "optimization_route_unit",
        "optimization_primary_guild",
        "planning_priority_score",
        "optimizer_marginal_gain",
        "optimizer_gain_route_coverage",
        "optimizer_gain_habitat_representation",
        "optimizer_gain_high_risk_coverage",
        "optimizer_gain_uncertainty_reduction",
        "optimizer_penalty_redundancy",
    ]
    preview_cols = [col for col in preview_cols if col in auto_selected.columns]
    st.dataframe(auto_selected[preview_cols], use_container_width=True)

    st.markdown("**Constraint reason frequencies**")
    reason_freq = _split_code_frequencies(reviewed_candidates.get("constraint_reason_codes"))
    if reason_freq:
        reason_df = pd.DataFrame([{"reason_code": key, "count": int(value)} for key, value in reason_freq.items()])
        st.dataframe(reason_df, use_container_width=True)
    else:
        st.info("No constraint reason codes were triggered.")


def _render_review_tab(st, *, last_run: dict[str, Any], review_state: dict[str, Any], review_summary: dict[str, Any]) -> None:
    reviewed_candidates = review_state["candidates_gdf"]
    final_selected = get_final_selected_candidates(reviewed_candidates)

    st.markdown("**Current final detector set**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final selected", int(review_summary["final_selected_count"]))
    c2.metric("Overrides", int(review_summary["override_count"]))
    c3.metric("Budget gap", int(review_summary.get("budget_gap", 0)))

    preview_cols = [
        "final_selection_rank",
        "candidate_id",
        "source_hf_uid",
        "final_selection_status",
        "planning_priority_score",
        "eco_primary_guild",
        "review_override_rationale",
    ]
    preview_cols = [col for col in preview_cols if col in final_selected.columns]
    st.dataframe(final_selected[preview_cols], use_container_width=True)

    reset_col, _ = st.columns([1, 3])
    if reset_col.button("Reset review to automatic planner selection", key="planner_review_reset"):
        st.session_state["planner_review"] = {
            "candidates_gdf": initialise_review_candidates(last_run["candidates_gdf"]),
            "audit_log": [],
        }
        st.rerun()

    st.markdown("**Apply expert override**")
    remove_options = final_selected["candidate_id"].astype(str).tolist()
    eligible_additions = reviewed_candidates[
        (reviewed_candidates["eligible_for_selection"].astype(int) == 1)
        & (reviewed_candidates["final_selected_flag"].astype(int) == 0)
    ]["candidate_id"].astype(str).tolist()

    if not remove_options:
        st.info("No selected candidates are available for review.")
        return

    action_options = ["replace", "remove"] if eligible_additions else ["remove"]
    if not eligible_additions:
        st.caption("No eligible unselected candidates are currently available for replacement, so only removal is available.")

    with st.form("planner_review_form"):
        action = st.selectbox("Override action", options=action_options, key="planner_review_action")
        remove_candidate_id = st.selectbox(
            "Selected candidate to remove",
            options=remove_options,
            key="planner_review_remove_candidate",
        )
        add_candidate_id = None
        if action == "replace":
            add_candidate_id = st.selectbox(
                "Replacement candidate",
                options=eligible_additions,
                key="planner_review_add_candidate",
                help="Only eligible, currently unselected candidates are available as replacements.",
            )
        rationale = st.text_area(
            "Override rationale",
            height=120,
            key="planner_review_rationale",
            help="Required. This rationale is stored in the audit trail and export summary.",
        )
        submit = st.form_submit_button("Apply override")

    if submit:
        sequence_no = int(len(review_state["audit_log"]) + 1)
        try:
            updated_gdf, audit_entry = apply_review_override(
                reviewed_candidates,
                action=action,
                remove_candidate_id=remove_candidate_id,
                add_candidate_id=add_candidate_id,
                rationale=rationale,
                sequence_no=sequence_no,
            )
        except Exception as exc:
            st.error(f"Could not apply override: {exc}")
            return

        st.session_state["planner_review"] = {
            "candidates_gdf": updated_gdf,
            "audit_log": list(review_state["audit_log"]) + [audit_entry],
        }
        st.rerun()

    st.markdown("**Override audit trail**")
    audit_log = review_state["audit_log"]
    if audit_log:
        st.dataframe(audit_log, use_container_width=True)
    else:
        st.info("No manual overrides have been applied yet.")


def _render_exports_tab(st, *, last_run: dict[str, Any], reviewed_candidates, review_summary: dict[str, Any]) -> None:
    candidates_export_df = _gdf_to_export_df(reviewed_candidates)
    final_selected = get_final_selected_candidates(reviewed_candidates)
    final_selected_export_df = _gdf_to_export_df(final_selected)
    evidence_pack_files = _build_evidence_pack_files(
        last_run=last_run,
        reviewed_candidates=reviewed_candidates,
        review_summary=review_summary,
    )
    base_stem = Path(str(last_run.get("source_name", "static_planner"))).stem or "static_planner"
    pack_zip_bytes = _zip_named_files(evidence_pack_files)

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download screened GPKG",
        data=evidence_pack_files["screened_gpkg"][1],
        file_name=evidence_pack_files["screened_gpkg"][0],
        mime="application/geopackage+sqlite3",
        key="planner_dl_screened_gpkg",
    )
    col3.download_button(
        "Download run manifest",
        data=evidence_pack_files["run_manifest"][1],
        file_name=evidence_pack_files["run_manifest"][0],
        mime="application/json",
        key="planner_dl_manifest_json",
    )
    col2.download_button(
        "Download candidate GPKG",
        data=evidence_pack_files["candidate_gpkg"][1],
        file_name=evidence_pack_files["candidate_gpkg"][0],
        mime="application/geopackage+sqlite3",
        key="planner_dl_candidate_gpkg",
    )

    col4, col5, col6 = st.columns(3)
    col4.download_button(
        "Download chosen detector set",
        data=evidence_pack_files["chosen_detector_set"][1],
        file_name=evidence_pack_files["chosen_detector_set"][0],
        mime="application/geopackage+sqlite3",
        key="planner_dl_selected_gpkg",
    )
    col5.download_button(
        "Download evidence report",
        data=evidence_pack_files["evidence_report"][1],
        file_name=evidence_pack_files["evidence_report"][0],
        mime="text/markdown",
        key="planner_dl_evidence_report",
    )
    col6.download_button(
        "Download evidence pack ZIP",
        data=pack_zip_bytes,
        file_name=f"{base_stem}_evidence_pack.zip",
        mime="application/zip",
        key="planner_dl_evidence_pack_zip",
    )

    col10, col11, col12 = st.columns(3)
    col10.download_button(
        "Download data catalogue",
        data=evidence_pack_files["data_catalogue"][1],
        file_name=evidence_pack_files["data_catalogue"][0],
        mime="application/json",
        key="planner_dl_data_catalogue_json",
    )
    col11.download_button(
        "Download feature health",
        data=evidence_pack_files["feature_health"][1],
        file_name=evidence_pack_files["feature_health"][0],
        mime="application/json",
        key="planner_dl_feature_health_json",
    )
    col12.download_button(
        "Download method statement",
        data=evidence_pack_files["method_statement"][1],
        file_name=evidence_pack_files["method_statement"][0],
        mime="text/markdown",
        key="planner_dl_method_statement_md",
    )

    st.markdown("**Supplemental tabular exports**")
    col7, col8, col9 = st.columns(3)
    col7.download_button(
        "Download reviewed candidates (CSV)",
        data=dataframe_to_csv_bytes(candidates_export_df),
        file_name=f"{base_stem}_planner_candidates_reviewed.csv",
        mime="text/csv",
        key="planner_dl_candidates_csv",
    )
    try:
        col8.download_button(
            "Download reviewed candidates (XLSX)",
            data=dataframe_to_xlsx_bytes(candidates_export_df),
            file_name=f"{base_stem}_planner_candidates_reviewed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="planner_dl_candidates_xlsx",
        )
    except Exception as exc:
        col8.warning(f"Candidates XLSX export unavailable: {exc}")
    try:
        col9.download_button(
            "Download final detector set (XLSX)",
            data=dataframe_to_xlsx_bytes(final_selected_export_df),
            file_name=f"{base_stem}_planner_selected_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="planner_dl_selected_xlsx",
        )
    except Exception as exc:
        col9.warning(f"Final detector XLSX export unavailable: {exc}")

    st.markdown("**Final detector set preview**")
    preview_cols = [
        "final_selection_rank",
        "candidate_id",
        "source_hf_uid",
        "final_selection_status",
        "planning_priority_score",
        "eco_primary_guild",
        "review_override_rationale",
    ]
    preview_cols = [col for col in preview_cols if col in final_selected.columns]
    st.dataframe(final_selected[preview_cols], use_container_width=True)


def _build_evidence_pack_files(*, last_run: dict[str, Any], reviewed_candidates, review_summary: dict[str, Any]):
    final_selected = get_final_selected_candidates(reviewed_candidates)
    base_stem = Path(str(last_run.get("source_name", "static_planner"))).stem or "static_planner"
    with tempfile.TemporaryDirectory(prefix="hedge_features_planner_pack_") as tmp_dir:
        screened_path = Path(tmp_dir) / f"{base_stem}_screened.gpkg"
        written = write_planning_evidence_pack(
            _ui_run_result(last_run),
            screened_path,
            source_name=str(last_run.get("source_name", "planning_source")),
            source_metadata=last_run.get("source_metadata"),
            reviewed_candidates_gdf=reviewed_candidates,
            selected_gdf=final_selected,
            review_summary=review_summary,
        )
        return {
            key: (Path(path).name, Path(path).read_bytes())
            for key, path in written.items()
        }


def _ui_run_result(last_run: dict[str, Any]):
    class _RunResult:
        def __init__(self, payload: dict[str, Any]):
            self.candidates_gdf = payload["candidates_gdf"]
            self.selected_gdf = payload["selected_gdf"]
            self.run_summary = payload["run_summary"]
            self.screened_gdf = payload["source_gdf"]

    return _RunResult(last_run)


def _zip_named_files(named_files: dict[str, tuple[str, bytes]]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="hedge_features_planner_zip_") as tmp_dir:
        zip_path = Path(tmp_dir) / "evidence_pack.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for _, (file_name, data) in named_files.items():
                zf.writestr(file_name, data)
        return zip_path.read_bytes()


def _gdf_to_export_df(gdf):
    df = gdf.copy()
    df["geometry_wkt"] = df.geometry.to_wkt()
    return df.drop(columns=df.geometry.name)


def _numeric_columns(df, *, ignore: list[str]) -> list[str]:
    import pandas as pd

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
    return sorted(numeric_cols)


def _unique_string_values(series) -> list[str]:
    values = series.astype("string").dropna().unique().tolist()
    return sorted(str(value) for value in values if str(value).strip())


def _bounds_zoom(*, min_x: float, min_y: float, max_x: float, max_y: float) -> float:
    lon_span = max(abs(max_x - min_x), 1e-6)
    lat_span = max(abs(max_y - min_y), 1e-6)
    max_span = max(lon_span, lat_span)
    zoom = 8.5 - math.log(max_span, 2)
    return max(6.0, min(16.0, zoom))


def _row_snapshot(row) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in row.items():
        if key == "geometry":
            snapshot["geometry_wkt"] = value.wkt if value is not None else None
            continue
        if hasattr(value, "item"):
            try:
                snapshot[key] = value.item()
                continue
            except Exception:
                pass
        snapshot[key] = value
    return snapshot


def _split_code_frequencies(series) -> dict[str, int]:
    counts: dict[str, int] = {}
    if series is None:
        return counts
    for raw in series.astype("string").fillna(""):
        for token in str(raw).split("|"):
            token = token.strip()
            if not token:
                continue
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        result = int(value)
        return result
    except Exception:
        return None
