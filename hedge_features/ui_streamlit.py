from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from .io import read_input_geodata
    from .planning import PlanningSettings, plan_static_detectors
    from .v2 import (
        BHSACalibrationSettings,
        BHSAScoringSettings,
        StaticAcousticSummarySettings,
        V2EvidencePackSettings,
        ValidationDiagnosticsSettings,
        build_v2_evidence_pack,
        build_validation_diagnostics,
        calibrate_bhsa_weights,
        parse_acoustic_survey_table,
        score_bhsa_table,
        summarise_static_acoustics,
        validate_project_dataset,
    )
except ImportError:  # Support `streamlit run hedge_features/ui_streamlit.py`
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hedge_features.io import read_input_geodata
    from hedge_features.planning import PlanningSettings, plan_static_detectors
    from hedge_features.v2 import (
        BHSACalibrationSettings,
        BHSAScoringSettings,
        StaticAcousticSummarySettings,
        V2EvidencePackSettings,
        ValidationDiagnosticsSettings,
        build_v2_evidence_pack,
        build_validation_diagnostics,
        calibrate_bhsa_weights,
        parse_acoustic_survey_table,
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
        "v2_acoustic_parse_audit": None,
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
    bhsa = st.session_state.get("v2_bhsa")
    acoustic = st.session_state.get("v2_acoustic_summary_df")
    planner = st.session_state.get("v2_planner")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Hedgerows", len(df))
    m2.metric("BHSA assessed", len(bhsa) if bhsa is not None else 0)
    m3.metric("Verification required", int(bhsa["field_verification_required"].astype(bool).sum()) if bhsa is not None else 0)
    m4.metric("Acoustic evidence", acoustic["hedgerow_id"].nunique() if acoustic is not None and "hedgerow_id" in acoustic else 0)
    m5.metric("Detectors selected", len(planner.selected_gdf) if planner is not None else 0)
    m6.metric("Readiness", readiness.get("status", "unknown") if readiness else "unknown")
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
    st.warning(
        "Remote limitation: SI6 woody species diversity and SI7 wet ditch cannot be verified from desk-based data alone. "
        "Proxy or hybrid scores using those indices should retain field verification unless field evidence is supplied."
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
    section_col = _first_existing(scored, ["section_id", "section", "section_name"])
    if section_col and "bhsa_class" in scored.columns:
        st.markdown("### Survey effort implication")
        effort = (
            scored.groupby([section_col, "bhsa_class"], dropna=False)
            .size()
            .reset_index(name="hedgerow_count")
            .sort_values([section_col, "bhsa_class"])
        )
        st.dataframe(effort, use_container_width=True)
    st.markdown("### Specialist evidence review")
    id_col = _first_existing(scored, ["hedgerow_id", "hf_uid", "source_hf_uid"])
    if id_col:
        selected_id = st.selectbox("Review hedgerow", scored[id_col].astype("string").tolist())
        row = scored.loc[scored[id_col].astype("string") == str(selected_id)].iloc[0]
        st.dataframe(_bhsa_explanation_table(row), use_container_width=True)
    st.info(
        "Technical meeting position: defend the BHSA class, survey-effort implication, confidence level, and field verification reason together. "
        "Do not defend low-confidence proxy SI6/SI7 as equivalent to field survey evidence."
    )


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
    st.info(
        "Technical meeting position: this table is the field verification schedule. It separates defensible desk-based BHSA outputs from confidence-limited decisions that should not be used to reduce survey effort without review."
    )


def _render_acoustic(st) -> None:
    _page_title(
        st,
        "Acoustic Analysis",
        "Standardise static detector outputs into hedgerow-season-species summaries before validation or calibration.",
    )
    st.caption("Accepted input is structured survey-result data: hedgerow/detector/night/species/pass-count/season/effort metadata. Audio files are out of scope.")
    upload_col, paste_col = st.columns(2)
    acoustic_upload = upload_col.file_uploader(
        "Upload static detector CSV/XLSX",
        type=["csv", "xlsx"],
        key="v2_acoustic_upload_analysis",
    )
    pasted_table = paste_col.text_area(
        "Paste acoustic survey table",
        height=140,
        placeholder="hedgerow_id,survey_night,season,species,passes,detector_id,detector_model,microphone_height_m,qa_status",
    )
    if st.button("Load / parse acoustic evidence"):
        try:
            if acoustic_upload is not None:
                acoustic_df = _read_uploaded_table(acoustic_upload)
                st.session_state["v2_acoustic_source_name"] = acoustic_upload.name
            elif pasted_table.strip():
                acoustic_df, audit = parse_acoustic_survey_table(pasted_table)
                st.session_state["v2_acoustic_parse_audit"] = audit
                st.session_state["v2_acoustic_source_name"] = "pasted table"
            else:
                acoustic_df = None
            if acoustic_df is not None:
                st.session_state["v2_acoustic_df"] = acoustic_df
                st.success("Acoustic evidence loaded.")
        except Exception as exc:
            st.error(f"Could not parse acoustic evidence: {exc}")

    acoustic_df = st.session_state.get("v2_acoustic_df")
    if acoustic_df is None:
        st.info("Upload or paste a static detector survey-result table to start acoustic analysis.")
        return
    columns = list(acoustic_df.columns)
    st.markdown("### Mapping review")
    c1, c2, c3 = st.columns(3)
    hedge_col = c1.selectbox("Hedgerow id column", columns, index=_column_index(columns, ["hedgerow_id", "hf_uid", "hedge_id"]))
    species_col = c2.selectbox("Species/class column", columns, index=_column_index(columns, ["species", "class", "predicted_class"]))
    datetime_col = c3.selectbox("Survey night/date column", [""] + columns, index=_column_index([""] + columns, ["survey_night", "datetime", "timestamp", "start_time", "date"]))
    c4, c5, c6 = st.columns(3)
    activity_col = c4.selectbox("Pass/count column", [""] + columns, index=_column_index([""] + columns, ["passes", "calls", "count", "activity"]))
    season_col = c5.selectbox("Season column", [""] + columns, index=_column_index([""] + columns, ["survey_season", "season"]))
    detector_col = c6.selectbox("Detector id column", [""] + columns, index=_column_index([""] + columns, ["detector_id", "static_id", "location_id"]))
    c7, c8, c9 = st.columns(3)
    detector_model_col = c7.selectbox("Detector model column", [""] + columns, index=_column_index([""] + columns, ["detector_model", "recorder_model", "model"]))
    mic_height_col = c8.selectbox("Microphone height column", [""] + columns, index=_column_index([""] + columns, ["microphone_height_m", "mic_height_m", "microphone_height"]))
    qa_col = c9.selectbox("QA status column", [""] + columns, index=_column_index([""] + columns, ["qa_status", "verified", "manual_qa_status"]))
    if st.button("Summarise static acoustic evidence", type="primary"):
        summary_df, run_summary = summarise_static_acoustics(
            acoustic_df,
            settings=StaticAcousticSummarySettings(
                hedgerow_id_column=hedge_col,
                species_column=species_col,
                datetime_column=datetime_col or "datetime",
                activity_column=activity_col or None,
                season_column=season_col or None,
                detector_id_column=detector_col or None,
                detector_model_column=detector_model_col or None,
                microphone_height_column=mic_height_col or None,
                qa_status_column=qa_col or None,
            ),
        )
        st.session_state["v2_acoustic_summary_df"] = summary_df
        st.session_state["v2_acoustic_run_summary"] = run_summary
        st.session_state["v2_acoustic_parse_audit"] = run_summary.get("column_audit")
        st.success("Acoustic summary complete.")
    summary_df = st.session_state.get("v2_acoustic_summary_df")
    run_summary = st.session_state.get("v2_acoustic_run_summary")
    audit = st.session_state.get("v2_acoustic_parse_audit")
    if audit:
        with st.expander("Column mapping audit", expanded=False):
            st.json(audit)
    if summary_df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Summary rows", len(summary_df))
        m2.metric("Hedgerows with evidence", run_summary.get("hedgerow_count", 0))
        m3.metric("Missing effort fields", len(run_summary.get("missing_effort_metadata_fields", [])))
        m4.metric("BAIV-ready rows", int(summary_df["acoustic_baiv_ready"].sum()) if "acoustic_baiv_ready" in summary_df else 0)
        if run_summary.get("qa_notes"):
            st.warning("; ".join(run_summary["qa_notes"]))
        if "survey_season" in summary_df.columns:
            season_totals = summary_df.groupby("survey_season", dropna=False)["acoustic_total_passes"].sum().reset_index()
            st.bar_chart(season_totals.set_index("survey_season"))
        st.dataframe(summary_df.head(300), use_container_width=True)
        st.info(
            "Technical meeting position: defend the acoustic evidence only after detector effort, microphone height, detector model, and QA comparability have been checked. The app standardises survey-result tables; it does not classify bat calls."
        )


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
    if combined is not None and st.button("Run validation diagnostics"):
        id_col = _first_existing(combined, ["hedgerow_id", "hf_uid"]) or "hedgerow_id"
        st.session_state["v2_validation"] = build_validation_diagnostics(
            combined,
            settings=ValidationDiagnosticsSettings(
                hedgerow_id_column=id_col,
                score_column="bhsa_score",
                class_column="bhsa_class",
                acoustic_activity_column="acoustic_total_passes",
            ),
        )
    validation = st.session_state.get("v2_validation")
    if validation:
        st.markdown("### BHSA class vs acoustic activity")
        metrics = validation.get("metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sensitivity", _display_metric(metrics.get("sensitivity")))
        c2.metric("Specificity", _display_metric(metrics.get("specificity")))
        c3.metric("Precision", _display_metric(metrics.get("precision")))
        c4.metric("Accuracy", _display_metric(metrics.get("accuracy")))
        c5.metric("ROC / AUC", _display_metric(metrics.get("auc")))
        matrix = validation.get("confusion_matrix", {})
        if matrix:
            import pandas as pd

            st.dataframe(pd.DataFrame([matrix]), use_container_width=True)
        if validation.get("class_activity_summary"):
            import pandas as pd

            class_summary = pd.DataFrame(validation["class_activity_summary"])
            st.dataframe(class_summary, use_container_width=True)
            if {"bhsa_class", "acoustic_total_activity"}.issubset(class_summary.columns):
                st.bar_chart(class_summary.set_index("bhsa_class")["acoustic_total_activity"])
        examples = validation.get("examples", {})
        if examples.get("high_score_no_acoustic_evidence"):
            st.markdown("#### High-score / no acoustic evidence")
            import pandas as pd

            st.dataframe(pd.DataFrame(examples["high_score_no_acoustic_evidence"]), use_container_width=True)
        if examples.get("low_score_positive_acoustic_evidence"):
            st.markdown("#### Low-score / positive acoustic evidence")
            import pandas as pd

            st.dataframe(pd.DataFrame(examples["low_score_positive_acoustic_evidence"]), use_container_width=True)
        if validation.get("caveats"):
            st.warning("; ".join(validation["caveats"]))
        st.info(
            "Technical meeting position: use these diagnostics to identify under-specified survey risk, over-specified effort, and cases needing ecologist review. Do not present AUC as transferability evidence where sample size or class balance caveats remain."
        )
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
            st.warning("Calibration is not ready for use: " + "; ".join(calibration.get("warnings", calibration.get("reliability_warnings", []))))
        else:
            st.success("Calibration output is ready for technical review. Equal-prior BHSA remains the default until reviewed.")
        weights = calibration.get("fitted_weights")
        if weights:
            import pandas as pd

            st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: "evidence_adjusted_weight"}), use_container_width=True)
        with st.expander("Calibration audit"):
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
            validation_diagnostics=st.session_state.get("v2_validation"),
            calibration_summary=st.session_state.get("v2_calibration"),
            planner_summary=(
                st.session_state["v2_planner"].run_summary
                if st.session_state.get("v2_planner") is not None
                else None
            ),
            detector_deployment=(
                st.session_state["v2_planner"].selected_gdf
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
        for key, table in pack.get("tables", {}).items():
            if key == "reviewer_override_log":
                continue
            st.download_button(
                f"Download {key.replace('_', ' ')} (CSV)",
                data=_df_to_csv_bytes(table),
                file_name=f"v2_{key}.csv",
                mime="text/csv",
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


def _bhsa_explanation_table(row):
    import pandas as pd

    labels = {
        "si1": "Height",
        "si2": "Width",
        "si3": "Gappiness",
        "si4": "Arable field margin",
        "si5": "Trees present",
        "si6": "Woody species diversity",
        "si7": "Wet ditch",
    }
    field_sources = {
        "si1": ["si1_height_m", "bhsa_height_m", "hedge_height_m", "height_m"],
        "si2": ["si2_width_m", "bhsa_width_m", "hedge_width_m", "width_m"],
        "si3": ["si3_gappiness_pct", "bhsa_gappiness_pct", "gappiness_pct", "gap_pct"],
        "si4": ["si4_arable_margin_m", "bhsa_arable_margin_m", "arable_margin_m"],
        "si5": ["si5_tree_count_50m", "bhsa_trees_per_50m", "trees_per_50m"],
        "si6": ["si6_woody_species_count_20m", "bhsa_woody_species_count_20m", "woody_species_count_20m"],
        "si7": ["si7_wet_ditch_present", "bhsa_wet_ditch_present", "wet_ditch_present"],
    }
    proxy_sources = {
        "si1": ["hedge_struct_height_mean_5m", "hedge_struct_height_p90_5m"],
        "si2": ["hedge_struct_width_proxy_m"],
        "si3": ["hedge_struct_gap_fraction_10m", "hedge_struct_canopy_continuity_10m"],
        "si4": ["buf100_worldcover_cropland_pct", "buf250_worldcover_cropland_pct"],
        "si5": ["hedge_struct_tree_standard_pct_10m", "buf100_worldcover_tree_pct", "mhb_corridor10_tree_pct"],
        "si6": ["buf100_worldcover_tree_pct", "buf100_phi_broadleaved_woodland_pct", "dist_awi_ancwood_m"],
        "si7": ["buf100_os_river_density_m_per_ha", "mhb_water_dist_m", "dist_os_river_m", "buf100_worldcover_wetland_pct"],
    }
    rows = []
    for key, label in labels.items():
        source = row.get(f"bhsa_{key}_source")
        reason = ""
        if key in {"si6", "si7"} and source != "field":
            reason = "Cannot be remotely verified; field check required."
        elif row.get(f"bhsa_{key}_confidence") in {"Low", "Missing"}:
            reason = "Confidence-limited proxy or missing evidence."
        rows.append(
            {
                "index": key.upper(),
                "attribute": label,
                "field value": _first_row_value(row, field_sources[key]),
                "proxy value": _first_row_value(row, proxy_sources[key]),
                "score": row.get(f"bhsa_{key}_score"),
                "chosen evidence source": source,
                "confidence": row.get(f"bhsa_{key}_confidence"),
                "field verification reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _first_row_value(row, names: list[str]):
    for name in names:
        if name in row.index:
            value = row.get(name)
            try:
                import pandas as pd

                if pd.isna(value):
                    continue
            except Exception:
                if value is None:
                    continue
            return value
    return ""


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


def _display_metric(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _df_to_csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


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
