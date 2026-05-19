from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from .acoustics import AcousticValidationSettings, validate_acoustic_evidence
    from .io import read_input_geodata
    from .planning import PlanningSettings, plan_static_detectors
    from .v2 import (
        BHSACalibrationSettings,
        BHSAScoringSettings,
        StaticAcousticSummarySettings,
        V2EvidencePackSettings,
        build_v2_evidence_pack,
        calibrate_bhsa_weights,
        score_bhsa_table,
        summarise_static_acoustics,
        validate_project_dataset,
    )
except ImportError:  # Support `streamlit run hedge_features/ui_streamlit.py`
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hedge_features.acoustics import AcousticValidationSettings, validate_acoustic_evidence
    from hedge_features.io import read_input_geodata
    from hedge_features.planning import PlanningSettings, plan_static_detectors
    from hedge_features.v2 import (
        BHSACalibrationSettings,
        BHSAScoringSettings,
        StaticAcousticSummarySettings,
        V2EvidencePackSettings,
        build_v2_evidence_pack,
        calibrate_bhsa_weights,
        score_bhsa_table,
        summarise_static_acoustics,
        validate_project_dataset,
    )


PAGES = [
    "Project Setup",
    "Remote BHSA",
    "Evidence Confidence",
    "Acoustic Analysis",
    "Validation & Calibration",
    "Detector Deployment",
    "Evidence Pack",
]


def run_app():
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover
        raise RuntimeError("Streamlit is not installed. Install with `pip install -e .[ui]`.")

    st.set_page_config(
        page_title="Bat Hedgerow Intelligence Platform V2",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)
    _init_state(st)
    page = _sidebar(st)
    if page == "Project Setup":
        _render_project_setup(st)
    elif page == "Remote BHSA":
        _render_remote_bhsa(st)
    elif page == "Evidence Confidence":
        _render_confidence(st)
    elif page == "Acoustic Analysis":
        _render_acoustic(st)
    elif page == "Validation & Calibration":
        _render_validation_calibration(st)
    elif page == "Detector Deployment":
        _render_deployment(st)
    else:
        _render_evidence_pack(st)


def _init_state(st) -> None:
    defaults = {
        "v2_project_name": "",
        "v2_analyst": "",
        "v2_source_name": None,
        "v2_project_df": None,
        "v2_project_is_geospatial": False,
        "v2_readiness": None,
        "v2_acoustic_df": None,
        "v2_acoustic_source_name": None,
        "v2_bhsa": None,
        "v2_bhsa_summary": None,
        "v2_acoustic_summary_df": None,
        "v2_acoustic_run_summary": None,
        "v2_validation": None,
        "v2_calibration": None,
        "v2_planner": None,
        "v2_pack": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _sidebar(st) -> str:
    with st.sidebar:
        st.markdown("## BHIP V2")
        st.caption("UK bat specialist workflow for remote BHSA, validation, calibration, and static survey design.")
        page = st.radio("Workflow", PAGES, key="v2_page")
        st.divider()
        df = st.session_state.get("v2_project_df")
        bhsa = st.session_state.get("v2_bhsa")
        acoustic = st.session_state.get("v2_acoustic_summary_df")
        planner = st.session_state.get("v2_planner")
        st.markdown("### Case Status")
        st.caption(f"Source: {st.session_state.get('v2_source_name') or 'not loaded'}")
        st.caption(f"Hedgerows: {len(df) if df is not None else 0}")
        st.caption(f"BHSA assessed: {len(bhsa) if bhsa is not None else 0}")
        st.caption(f"Acoustic rows: {len(acoustic) if acoustic is not None else 0}")
        selected = len(planner.selected_gdf) if planner is not None else 0
        st.caption(f"Detector locations selected: {selected}")
    return page


def _render_project_setup(st) -> None:
    _page_title(
        st,
        "Project Setup",
        "Load the hedgerow case file, confirm identifiers and CRS, and record the project context before any BHSA decision is made.",
    )
    c1, c2 = st.columns(2)
    st.session_state["v2_project_name"] = c1.text_input(
        "Scheme / project name",
        value=st.session_state.get("v2_project_name", ""),
        placeholder="EGL section 4, EDN2, A5 WTC...",
    )
    st.session_state["v2_analyst"] = c2.text_input(
        "Analyst / reviewer",
        value=st.session_state.get("v2_analyst", ""),
    )
    uploaded = st.file_uploader(
        "Upload hedgerow table or geospatial layer",
        type=["csv", "xlsx", "gpkg", "geojson", "json", "zip", "shp"],
        key="v2_source_upload",
    )
    input_crs = st.text_input("Input CRS if the uploaded geospatial layer has no CRS", value="", key="v2_input_crs")
    if uploaded is not None and st.button("Load project dataset", type="primary"):
        try:
            df, is_geo = _read_uploaded_dataset(uploaded, input_crs=input_crs.strip() or None)
            st.session_state["v2_project_df"] = df
            st.session_state["v2_project_is_geospatial"] = is_geo
            st.session_state["v2_source_name"] = uploaded.name
            st.session_state["v2_readiness"] = validate_project_dataset(df)
            st.success("Project dataset loaded and checked.")
        except Exception as exc:
            st.error(f"Could not load project dataset: {exc}")

    acoustic_uploaded = st.file_uploader(
        "Optional: upload static detector CSV/XLSX for acoustic analysis",
        type=["csv", "xlsx"],
        key="v2_acoustic_upload",
    )
    if acoustic_uploaded is not None and st.button("Load acoustic table"):
        try:
            acoustic_df = _read_uploaded_table(acoustic_uploaded)
            st.session_state["v2_acoustic_df"] = acoustic_df
            st.session_state["v2_acoustic_source_name"] = acoustic_uploaded.name
            st.success("Acoustic table loaded.")
        except Exception as exc:
            st.error(f"Could not load acoustic table: {exc}")

    _render_project_dashboard(st)


def _render_project_dashboard(st) -> None:
    df = st.session_state.get("v2_project_df")
    readiness = st.session_state.get("v2_readiness")
    if df is None:
        st.info("Load a project dataset to unlock the V2 workflow.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hedgerows", len(df))
    m2.metric("Columns", len(df.columns))
    m3.metric("Readiness", readiness.get("status", "unknown") if readiness else "unknown")
    m4.metric("CRS", readiness.get("crs") or "not geospatial" if readiness else "unknown")
    if readiness:
        if readiness["issues"]:
            st.error("Blocking readiness issues: " + "; ".join(readiness["issues"]))
        if readiness["warnings"]:
            st.warning("Review warnings: " + "; ".join(readiness["warnings"]))
        with st.expander("Readiness report"):
            st.json(readiness)
    st.dataframe(df.head(30), use_container_width=True)


def _render_remote_bhsa(st) -> None:
    _page_title(
        st,
        "Remote BHSA",
        "Apply the BHSA scoring structure to field, proxy, or hybrid evidence and make confidence limitations explicit.",
    )
    df = st.session_state.get("v2_project_df")
    if df is None:
        st.info("Load a project dataset first.")
        return
    c1, c2 = st.columns([1, 2])
    mode = c1.radio("Assessment mode", ["hybrid", "field", "proxy"], horizontal=True)
    c2.caption(
        "Hybrid uses field scores where present and GIS proxies where field evidence is absent. Proxy mode is confidence-limited and flags SI6/SI7 for field verification."
    )
    if st.button("Run BHSA decision support", type="primary"):
        scored, summary = score_bhsa_table(df, settings=BHSAScoringSettings(mode=mode))
        st.session_state["v2_bhsa"] = scored
        st.session_state["v2_bhsa_summary"] = summary
        st.success("BHSA scoring complete.")
    scored = st.session_state.get("v2_bhsa")
    summary = st.session_state.get("v2_bhsa_summary")
    if scored is None:
        st.info("Run BHSA scoring to review suitability classes and survey implications.")
        return
    _bhsa_metrics(st, scored, summary)
    decision_cols = [
        col
        for col in [
            "hedgerow_id",
            "hf_uid",
            "section_id",
            "bhsa_score",
            "bhsa_class",
            "bhsa_survey_requirement",
            "bhsa_confidence_level",
            "field_verification_required",
            "bhsa_missing_reasons",
            "bhsa_notes",
        ]
        if col in scored.columns
    ]
    st.dataframe(scored[decision_cols].head(200), use_container_width=True)


def _render_confidence(st) -> None:
    _page_title(
        st,
        "Evidence Confidence",
        "Review which hedgerows can be defended from desk-based evidence and which need targeted field verification.",
    )
    readiness = st.session_state.get("v2_readiness")
    scored = st.session_state.get("v2_bhsa")
    if readiness:
        st.markdown("### Project readiness")
        st.json(readiness)
    if scored is None:
        st.info("Run Remote BHSA before reviewing confidence.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Field verification required", int(scored["field_verification_required"].astype(bool).sum()))
    c2.metric("Low confidence", int((scored["bhsa_confidence_level"].astype(str) == "Low").sum()))
    c3.metric("Incomplete", int((scored["bhsa_confidence_level"].astype(str) == "Incomplete").sum()))
    review = scored.loc[scored["field_verification_required"].astype(bool)].copy()
    cols = [c for c in ["hedgerow_id", "hf_uid", "section_id", "bhsa_class", "bhsa_confidence_level", "bhsa_missing_reasons", "bhsa_notes"] if c in review.columns]
    st.dataframe(review[cols].head(300), use_container_width=True)


def _render_acoustic(st) -> None:
    _page_title(
        st,
        "Acoustic Analysis",
        "Standardise static detector outputs into hedgerow-season-species summaries before validation or calibration.",
    )
    acoustic_df = st.session_state.get("v2_acoustic_df")
    if acoustic_df is None:
        st.info("Load an acoustic CSV/XLSX in Project Setup.")
        return
    columns = list(acoustic_df.columns)
    c1, c2, c3 = st.columns(3)
    hedge_col = c1.selectbox("Hedgerow id column", columns, index=_column_index(columns, ["hedgerow_id", "hf_uid", "hedge_id"]))
    species_col = c2.selectbox("Species/class column", columns, index=_column_index(columns, ["species", "class", "predicted_class"]))
    datetime_col = c3.selectbox("Datetime column", [""] + columns, index=_column_index([""] + columns, ["datetime", "timestamp", "start_time"]))
    c4, c5 = st.columns(2)
    activity_col = c4.selectbox("Pass/count column", [""] + columns, index=_column_index([""] + columns, ["passes", "calls", "count", "activity"]))
    season_col = c5.selectbox("Season column", [""] + columns, index=_column_index([""] + columns, ["survey_season", "season"]))
    if st.button("Summarise static acoustic evidence", type="primary"):
        summary_df, run_summary = summarise_static_acoustics(
            acoustic_df,
            settings=StaticAcousticSummarySettings(
                hedgerow_id_column=hedge_col,
                species_column=species_col,
                datetime_column=datetime_col or "datetime",
                activity_column=activity_col or None,
                season_column=season_col or None,
            ),
        )
        st.session_state["v2_acoustic_summary_df"] = summary_df
        st.session_state["v2_acoustic_run_summary"] = run_summary
        st.success("Acoustic summary complete.")
    summary_df = st.session_state.get("v2_acoustic_summary_df")
    run_summary = st.session_state.get("v2_acoustic_run_summary")
    if summary_df is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Summary rows", len(summary_df))
        m2.metric("Hedgerows with evidence", run_summary.get("hedgerow_count", 0))
        m3.metric("Missing effort fields", len(run_summary.get("missing_effort_metadata_fields", [])))
        if run_summary.get("qa_notes"):
            st.warning("; ".join(run_summary["qa_notes"]))
        st.dataframe(summary_df.head(300), use_container_width=True)


def _render_validation_calibration(st) -> None:
    _page_title(
        st,
        "Validation & Calibration",
        "Compare BHSA decisions against acoustic evidence and scaffold empirical weight calibration where paired data are sufficient.",
    )
    scored = st.session_state.get("v2_bhsa")
    acoustic_summary = st.session_state.get("v2_acoustic_summary_df")
    if scored is None:
        st.info("Run Remote BHSA first.")
        return
    combined = _combined_bhsa_acoustic(scored, acoustic_summary)
    if combined is not None and st.button("Run acoustic validation"):
        annotated, summary = validate_acoustic_evidence(
            combined,
            settings=AcousticValidationSettings(
                score_column="bhsa_score",
                acoustic_presence_column="acoustic_total_passes",
                low_score_threshold=1.70,
                high_score_threshold=2.40,
                id_column=_first_existing(combined, ["hedgerow_id", "hf_uid"]),
            ),
        )
        st.session_state["v2_validation"] = {"annotated": annotated, "summary": summary}
    validation = st.session_state.get("v2_validation")
    if validation:
        st.markdown("### Validation cases")
        st.json(validation["summary"])
    if combined is None:
        st.info("Load and summarise acoustic evidence to run validation. Calibration can still run if the BHSA table already contains labels.")
        calibration_source = scored
    else:
        calibration_source = combined
    activity_candidates = ["acoustic_total_passes", "acoustic_activity_sum", "total_calls", "baiv"]
    activity_col = _first_existing(calibration_source, activity_candidates)
    if st.button("Run BHSA weight calibration scaffold"):
        st.session_state["v2_calibration"] = calibrate_bhsa_weights(
            calibration_source,
            settings=BHSACalibrationSettings(
                label_column="high_activity_label" if "high_activity_label" in calibration_source.columns else None,
                activity_column=activity_col,
            ),
        )
    calibration = st.session_state.get("v2_calibration")
    if calibration:
        if calibration.get("do_not_use_calibrated_model"):
            st.warning("Calibration is not ready for use: " + "; ".join(calibration.get("warnings", [])))
        st.json(calibration)


def _render_deployment(st) -> None:
    _page_title(
        st,
        "Detector Deployment",
        "Optimise static detector placement using BHSA priority, access constraints, section coverage, and expert review outputs.",
    )
    scored = st.session_state.get("v2_bhsa")
    if scored is None or not st.session_state.get("v2_project_is_geospatial"):
        st.info("Run Remote BHSA on a geospatial hedgerow layer to optimise detector deployment.")
        return
    gdf = scored.copy()
    if "hf_uid" not in gdf.columns:
        id_col = _first_existing(gdf, ["hedgerow_id", "source_hf_uid"])
        if id_col:
            gdf["hf_uid"] = gdf[id_col].astype("string")
    c1, c2, c3 = st.columns(3)
    budget = c1.number_input("Available detectors", min_value=1, max_value=500, value=12, step=1)
    spacing = c2.number_input("Candidate spacing (m)", min_value=10.0, value=100.0, step=10.0)
    min_spacing = c3.number_input("Minimum detector spacing (m)", min_value=0.0, value=150.0, step=25.0)
    section_col = st.selectbox("Section coverage column", [""] + list(gdf.columns), index=_column_index([""] + list(gdf.columns), ["section_id", "section_name"]))
    if st.button("Optimise detector deployment", type="primary"):
        try:
            result = plan_static_detectors(
                gdf,
                settings=PlanningSettings(
                    detector_budget=int(budget),
                    candidate_spacing_m=float(spacing),
                    min_detector_spacing_m=float(min_spacing),
                    score_column="bhsa_score",
                    use_evidence_engine=False,
                    section_column=section_col or None,
                ),
                hedge_id_column="hf_uid",
            )
            st.session_state["v2_planner"] = result
            st.success("Detector deployment optimisation complete.")
        except Exception as exc:
            st.error(f"Could not optimise detector deployment: {exc}")
    result = st.session_state.get("v2_planner")
    if result is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates", len(result.candidates_gdf))
        m2.metric("Eligible", result.run_summary.get("eligible_count", 0))
        m3.metric("Selected", len(result.selected_gdf))
        m4.metric("High-risk selected", result.run_summary.get("optimizer", {}).get("selected_high_risk_corridors", 0))
        st.dataframe(result.selected_gdf.head(200), use_container_width=True)
        with st.expander("Planner run summary"):
            st.json(result.run_summary)


def _render_evidence_pack(st) -> None:
    _page_title(
        st,
        "Evidence Pack",
        "Export the V2 decision manifest and method statement for technical review, regulator discussion, or inquiry preparation.",
    )
    if st.button("Build V2 evidence pack", type="primary"):
        pack = build_v2_evidence_pack(
            bhsa_gdf=st.session_state.get("v2_bhsa"),
            readiness_report=st.session_state.get("v2_readiness"),
            acoustic_summary=st.session_state.get("v2_acoustic_summary_df"),
            calibration_summary=st.session_state.get("v2_calibration"),
            planner_summary=(
                st.session_state["v2_planner"].run_summary
                if st.session_state.get("v2_planner") is not None
                else None
            ),
            settings=V2EvidencePackSettings(
                project_name=st.session_state.get("v2_project_name") or "Unnamed scheme",
                analyst=st.session_state.get("v2_analyst") or "",
            ),
        )
        st.session_state["v2_pack"] = pack
        st.success("Evidence pack built.")
    pack = st.session_state.get("v2_pack")
    if pack:
        st.download_button(
            "Download V2 run manifest (JSON)",
            data=json.dumps(pack["manifest"], indent=2).encode("utf-8"),
            file_name="v2_run_manifest.json",
            mime="application/json",
        )
        st.download_button(
            "Download method statement (Markdown)",
            data=pack["method_statement_md"].encode("utf-8"),
            file_name="v2_method_statement.md",
            mime="text/markdown",
        )
        st.markdown(pack["method_statement_md"])


def _bhsa_metrics(st, scored, summary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hedgerows scored", summary.get("scored_row_count", 0))
    c2.metric("Excellent", summary.get("class_counts", {}).get("Excellent", 0))
    c3.metric("Good", summary.get("class_counts", {}).get("Good", 0))
    c4.metric("Verification required", summary.get("field_verification_required_count", 0))
    with st.expander("BHSA run summary"):
        st.json(summary)


def _combined_bhsa_acoustic(scored, acoustic_summary):
    if acoustic_summary is None or acoustic_summary.empty:
        return None
    import pandas as pd

    id_col = _first_existing(scored, ["hedgerow_id", "hf_uid"])
    acoustic_id = _first_existing(acoustic_summary, ["hedgerow_id", "hf_uid", "hedge_id"])
    if id_col is None or acoustic_id is None:
        return None
    totals = (
        acoustic_summary.groupby(acoustic_id, dropna=False)["acoustic_total_passes"]
        .sum()
        .reset_index()
        .rename(columns={acoustic_id: id_col})
    )
    return scored.merge(totals, how="left", on=id_col).assign(
        acoustic_total_passes=lambda df: pd.to_numeric(df["acoustic_total_passes"], errors="coerce").fillna(0.0)
    )


def _read_uploaded_dataset(uploaded, *, input_crs: str | None):
    suffix = Path(str(uploaded.name)).suffix.lower()
    if suffix in {".csv", ".xlsx"}:
        return _read_uploaded_table(uploaded), False
    with tempfile.TemporaryDirectory(prefix="bhip_v2_upload_") as tmp_dir:
        path = Path(tmp_dir) / str(uploaded.name)
        path.write_bytes(uploaded.getvalue())
        gdf, _ = read_input_geodata(path, input_crs=input_crs)
    return gdf, True


def _read_uploaded_table(uploaded):
    import pandas as pd

    suffix = Path(str(uploaded.name)).suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def _first_existing(df, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _column_index(columns: list[str], preferred: list[str]) -> int:
    lower = {str(col).lower(): idx for idx, col in enumerate(columns)}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    return 0


def _page_title(st, title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)


def _inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1440px; padding-top: 1.25rem; }
        [data-testid="stMetric"] { border: 1px solid rgba(40, 56, 48, 0.12); padding: 0.65rem; border-radius: 6px; background: #ffffff; }
        .stDataFrame { border: 1px solid rgba(40, 56, 48, 0.08); }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_app()
