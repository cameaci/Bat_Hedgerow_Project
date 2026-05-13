from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from .deps import require_geopandas
    from .models import RunOptions
    from .planning import TARGET_SPECS
    from .pipeline import run_enrichment
    from .ui_planner import render_planner_tab
    from .ui_shell import (
        inject_global_styles,
        render_info_card,
        render_page_hero,
        render_section_intro,
        render_sidebar_navigation,
        render_side_note,
        render_step_chips,
    )
    from .screening import (
        ScreeningSettings,
        get_bundled_framework_names,
        load_framework_bundle,
        screen_dataframe,
    )
    from .screening.engine import (
        LOW_CONFIDENCE_ACTION,
        ScreeningInputError,
        align_predictors,
        build_column_audit,
    )
    from .screening.io import (
        dataframe_to_csv_bytes,
        dataframe_to_xlsx_bytes,
        json_bytes,
        read_uploaded_attribute_table,
    )
except ImportError:  # Support `streamlit run hedge_features/ui_streamlit.py`
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hedge_features.deps import require_geopandas
    from hedge_features.models import RunOptions
    from hedge_features.planning import TARGET_SPECS
    from hedge_features.pipeline import run_enrichment
    from hedge_features.ui_planner import render_planner_tab
    from hedge_features.ui_shell import (
        inject_global_styles,
        render_info_card,
        render_page_hero,
        render_section_intro,
        render_sidebar_navigation,
        render_side_note,
        render_step_chips,
    )
    from hedge_features.screening import (
        ScreeningSettings,
        get_bundled_framework_names,
        load_framework_bundle,
        screen_dataframe,
    )
    from hedge_features.screening.engine import (
        LOW_CONFIDENCE_ACTION,
        ScreeningInputError,
        align_predictors,
        build_column_audit,
    )
    from hedge_features.screening.io import (
        dataframe_to_csv_bytes,
        dataframe_to_xlsx_bytes,
        json_bytes,
        read_uploaded_attribute_table,
    )


def _test_credentials(
    *,
    earthdata_token: str | None,
    eog_username: str | None,
    eog_password: str | None,
) -> dict[str, str]:
    import urllib.error
    import urllib.request

    results: dict[str, str] = {}
    if earthdata_token:
        req = urllib.request.Request(
            "https://cmr.earthdata.nasa.gov/search/collections.json?page_size=1",
            headers={"Authorization": f"Bearer {earthdata_token}", "User-Agent": "hedge-features/0.1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                results["earthdata"] = f"ok ({resp.status})"
        except urllib.error.HTTPError as exc:
            results["earthdata"] = f"http_error ({exc.code})"
        except Exception as exc:  # pragma: no cover
            results["earthdata"] = f"error ({type(exc).__name__})"

    if eog_username and eog_password:
        results["eog"] = "provided (validation endpoint not implemented in v1)"

    if not results:
        results["status"] = "no_credentials_provided"
    return results


def run_app():
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover
        raise RuntimeError("Streamlit is not installed. Install with `pip install -e .[ui]`.")

    st.set_page_config(page_title="Hedge Features", layout="wide", initial_sidebar_state="expanded")
    inject_global_styles(st)
    _init_session_state(st)
    workflow_state = _workflow_state(st)
    page = render_sidebar_navigation(
        st,
        workflow_state=workflow_state,
        active_page=st.session_state.get("ui_active_page", "Overview"),
    )
    st.session_state["ui_active_page"] = page

    if page == "Overview":
        _render_overview_page(st, workflow_state=workflow_state)
    elif page == "GIS Enrichment":
        _render_enrichment_tab(st)
    elif page == "Data Readiness":
        _render_data_readiness_page(st)
    elif page == "GIS-only Bat Screening":
        _render_screening_tab(st)
    elif page == "Species Strategy":
        _render_species_strategy_page(st)
    else:
        render_planner_tab(st)


def _init_session_state(st) -> None:
    st.session_state.setdefault("ui_active_page", "Overview")
    st.session_state.setdefault("enrichment_last_run", None)
    st.session_state.setdefault("screening_inapp", None)
    st.session_state.setdefault("screening_last_run", None)
    st.session_state.setdefault("screening_prefill_route", "Upload enriched CSV/XLSX")
    st.session_state.setdefault("planner_inapp", None)
    st.session_state.setdefault("planner_last_run", None)
    st.session_state.setdefault("planner_review", None)
    st.session_state.setdefault("planner_prefill_route", "Upload enriched geospatial dataset")


def _workflow_state(st) -> dict[str, dict[str, Any]]:
    enrichment = st.session_state.get("enrichment_last_run")
    screening = st.session_state.get("screening_last_run")
    planner = st.session_state.get("planner_last_run")
    return {
        "GIS Enrichment": {
            "ready": enrichment is not None,
            "status_text": (
                f"Ready: {enrichment['source_name']}" if enrichment is not None else "No enriched dataset cached in this session"
            ),
        },
        "Data Readiness": {
            "ready": enrichment is not None,
            "status_text": (
                _data_readiness_status_text(enrichment) if enrichment is not None else "No enrichment metadata cached in this session"
            ),
        },
        "GIS-only Bat Screening": {
            "ready": screening is not None,
            "status_text": (
                f"Ready: {len(screening['results_df'])} rows screened" if screening is not None else "No screening run cached in this session"
            ),
        },
        "Species Strategy": {
            "ready": True,
            "status_text": _species_strategy_status_text(planner),
        },
        "Static Detector Planner": {
            "ready": planner is not None,
            "status_text": (
                f"Ready: {len(planner['selected_gdf'])} detector locations selected" if planner is not None else "No detector planning run cached in this session"
            ),
        },
    }


def _data_readiness_status_text(enrichment: dict[str, Any]) -> str:
    metadata = _decode_json_bytes(enrichment.get("metadata_bytes"))
    feature_health = metadata.get("feature_health", {}) if isinstance(metadata, dict) else {}
    summary = feature_health.get("summary", {}) if isinstance(feature_health, dict) else {}
    high_null = len(summary.get("high_null_features", [])) if isinstance(summary.get("high_null_features"), list) else 0
    if high_null:
        return f"Review needed: {high_null} near-empty features"
    return "Ready: data catalogue and feature health available"


def _species_strategy_status_text(planner: dict[str, Any] | None) -> str:
    if planner is None:
        return f"Configured: {len(TARGET_SPECS)} guild/species targets available"
    run_summary = planner.get("run_summary", {})
    scenario = str(run_summary.get("planning_target_scenario", "all_bats"))
    return f"Last planner scenario: {scenario}"


def _decode_json_bytes(payload: bytes | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {}


def _render_overview_page(st, *, workflow_state: dict[str, dict[str, Any]]) -> None:
    render_page_hero(
        st,
        eyebrow="WSP Survey Workflow",
        title="Bat Survey Decision Support",
        subtitle=(
            "Use the app as a single guided workflow: enrich hedgerows, screen risk and survey priority, then plan "
            "static detector locations and export a defensible evidence pack."
        ),
    )
    render_step_chips(
        st,
        steps=["1. Enrich", "2. Screen", "3. Plan", "4. Evidence Pack"],
        current_step="1. Enrich",
    )

    top_cols = st.columns([2, 1])
    with top_cols[0]:
        render_section_intro(
            st,
            title="How To Use This App",
            subtitle=(
                "The previous interface exposed too many controls at once. This redesign focuses on a clearer sequence, "
                "persistent session status, and explicit outputs at each stage."
            ),
        )
    with top_cols[1]:
        render_side_note(
            st,
            "Best practice for project teams: run GIS Enrichment once per source layer, review GIS-only screening as a triage step, then finalise detector placement in the planner."
        )

    card_cols = st.columns(3)
    with card_cols[0]:
        render_info_card(
            st,
            title="1. GIS Enrichment",
            body=(
                "Start here. Create the GIS feature layer that every downstream workflow depends on."
            ),
            chips=["Input hedgerows", workflow_state["GIS Enrichment"]["status_text"]],
        )
        if st.button("Open GIS Enrichment", key="nav_open_enrichment"):
            st.session_state["ui_active_page"] = "GIS Enrichment"
            st.rerun()
    with card_cols[1]:
        render_info_card(
            st,
            title="2. GIS-only Bat Screening",
            body=(
                "Use enriched tables to rank survey priority, inspect confidence, and review reason codes before committing to field effort."
            ),
            chips=["Triage output", workflow_state["GIS-only Bat Screening"]["status_text"]],
        )
        if st.button("Open Screening", key="nav_open_screening"):
            st.session_state["ui_active_page"] = "GIS-only Bat Screening"
            st.rerun()
    with card_cols[2]:
        render_info_card(
            st,
            title="3. Static Detector Planner",
            body=(
                "Generate candidate detector points, optimise coverage, review on a map, and export the evidence pack."
            ),
            chips=["Decision support", workflow_state["Static Detector Planner"]["status_text"]],
        )
        if st.button("Open Planner", key="nav_open_planner"):
            st.session_state["ui_active_page"] = "Static Detector Planner"
            st.rerun()

    secondary_cols = st.columns(2)
    with secondary_cols[0]:
        render_info_card(
            st,
            title="Data Readiness",
            body=(
                "Review dataset availability, proxy and fallback use, outside-coverage warnings, and feature-health summaries before trusting downstream outputs."
            ),
            chips=["Data trust", workflow_state["Data Readiness"]["status_text"]],
        )
        if st.button("Open Data Readiness", key="nav_open_data_readiness"):
            st.session_state["ui_active_page"] = "Data Readiness"
            st.rerun()
    with secondary_cols[1]:
        render_info_card(
            st,
            title="Species Strategy",
            body=(
                "Inspect which guilds, species, and grouped taxa are supported, provisional, or grouped in the bankable planning path."
            ),
            chips=["Species logic", workflow_state["Species Strategy"]["status_text"]],
        )
        if st.button("Open Species Strategy", key="nav_open_species_strategy"):
            st.session_state["ui_active_page"] = "Species Strategy"
            st.rerun()

    st.divider()
    summary_cols = st.columns(5)
    enrichment = st.session_state.get("enrichment_last_run")
    screening = st.session_state.get("screening_last_run")
    planner = st.session_state.get("planner_last_run")
    summary_cols[0].metric("Enriched datasets in session", 1 if enrichment is not None else 0)
    summary_cols[1].metric("Readiness pages", 2)
    summary_cols[2].metric("Rows screened in session", len(screening["results_df"]) if screening is not None else 0)
    summary_cols[3].metric("Selected detector locations", len(planner["selected_gdf"]) if planner is not None else 0)
    summary_cols[4].metric("Species targets configured", len(TARGET_SPECS))


def _render_data_readiness_page(st) -> None:
    render_page_hero(
        st,
        eyebrow="Trust Layer",
        title="Data Readiness",
        subtitle=(
            "Review what was measured, what is missing, what is proxy-based, and what sits outside authoritative coverage before relying on any ranking or detector placement output."
        ),
    )
    last_run = st.session_state.get("enrichment_last_run")
    if last_run is None:
        st.info("Run GIS Enrichment first to populate the data catalogue and feature-health summaries.")
        return

    metadata = _decode_json_bytes(last_run.get("metadata_bytes"))
    data_catalogue = metadata.get("data_catalogue", {}) if isinstance(metadata, dict) else {}
    feature_health = metadata.get("feature_health", {}) if isinstance(metadata, dict) else {}
    data_summary = data_catalogue.get("summary", {}) if isinstance(data_catalogue, dict) else {}
    health_summary = feature_health.get("summary", {}) if isinstance(feature_health, dict) else {}

    metric_cols = st.columns(4)
    metric_cols[0].metric("Enabled datasets", int(data_summary.get("enabled_dataset_count", 0)))
    metric_cols[1].metric("Configured datasets", int(data_summary.get("dataset_count", 0)))
    metric_cols[2].metric("High-null features", len(health_summary.get("high_null_features", [])))
    metric_cols[3].metric("Guidance regime", metadata.get("guidance_regime_version", "n/a"))

    section_cols = st.columns(2)
    with section_cols[0]:
        render_section_intro(
            st,
            title="Dataset Catalogue",
            subtitle="These source states distinguish available inputs from authenticated, manual, missing, or unused inputs.",
        )
        datasets = data_catalogue.get("datasets", [])
        if datasets:
            st.dataframe(datasets, use_container_width=True)
        else:
            st.info("No dataset catalogue entries are available.")
    with section_cols[1]:
        render_section_intro(
            st,
            title="Feature Health Summary",
            subtitle="These summaries highlight feature families with weak support, high null rates, or proxy and fallback risk.",
        )
        st.json(health_summary or {"status": "not_available"})

    render_section_intro(
        st,
        title="High-null Features",
        subtitle="Features listed here are nearly empty in the current run and should be treated cautiously in downstream analysis.",
    )
    high_null = health_summary.get("high_null_features", [])
    if high_null:
        st.write(high_null)
    else:
        st.success("No feature columns crossed the high-null threshold in the cached enrichment run.")


def _render_species_strategy_page(st) -> None:
    render_page_hero(
        st,
        eyebrow="Target Logic",
        title="Species Strategy",
        subtitle=(
            "The bankable planner ships guild outputs for all candidates, ready interim species outputs for priority taxa, and grouped outputs where acoustic separation is weak."
        ),
    )
    planner_last_run = st.session_state.get("planner_last_run")
    if planner_last_run is not None:
        run_summary = planner_last_run.get("run_summary", {})
        st.info(
            f"Last planner scenario: {run_summary.get('planning_target_scenario', 'all_bats')} | "
            f"Guidance regime: {run_summary.get('guidance_regime_version', 'n/a')}"
        )

    target_rows = [
        {
            "slug": spec["slug"],
            "label": spec["label"],
            "kind": spec["kind"],
            "group": spec["group"],
            "model_status": spec["model_status"],
        }
        for spec in TARGET_SPECS
    ]
    ready_count = sum(1 for row in target_rows if row["model_status"] == "Ready")
    grouped_count = sum(1 for row in target_rows if row["model_status"] == "Grouped taxon")
    interim_count = sum(1 for row in target_rows if row["model_status"] == "Interim model")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Ready guilds", ready_count)
    metric_cols[1].metric("Interim species", interim_count)
    metric_cols[2].metric("Grouped taxa", grouped_count)

    render_section_intro(
        st,
        title="Target Catalogue",
        subtitle="Use guild scenarios for broad planning coverage. Use interim species scenarios to test defensible differences in detector placement without overclaiming certainty.",
    )
    st.dataframe(target_rows, use_container_width=True)

    render_section_intro(
        st,
        title="Interpretation Rules",
        subtitle="These are the operating assumptions used in the bankable planning path.",
    )
    st.write(
        [
            "Guild outputs are available for every candidate row.",
            "Priority standalone species are provisional and literature-driven until WSP calibration data is loaded.",
            "Myotis spp. and Plecotus spp. remain grouped because acoustic separation is weaker and grouped interpretation is more defensible in v1.",
            "Scenario changes should change detector placement deterministically when the ecological evidence differs materially by target.",
        ]
    )


def _render_enrichment_tab(st) -> None:
    render_page_hero(
        st,
        eyebrow="Step 1",
        title="GIS Enrichment",
        subtitle=(
            "Create the enriched hedgerow layer used by every downstream workflow. Keep this step simple: upload the source layer, "
            "confirm CRS/settings, run enrichment, then reuse the output in screening and planning."
        ),
    )
    render_step_chips(
        st,
        steps=["1. Source", "2. Core Settings", "3. Optional Inputs", "4. Run & Outputs"],
        current_step="1. Source",
    )

    uploaded = st.file_uploader(
        "Upload hedgerow dataset",
        type=["zip", "gpkg", "geojson", "json", "shp"],
        key="enrich_upload",
    )
    last_run = st.session_state.get("enrichment_last_run")
    if uploaded is not None:
        st.caption(f"Current upload: {uploaded.name}")

    tab_source, tab_settings, tab_optional, tab_outputs = st.tabs(
        ["1. Source", "2. Core Settings", "3. Optional Inputs", "4. Run & Outputs"]
    )

    with tab_source:
        render_section_intro(
            st,
            title="Source Setup",
            subtitle="Upload the hedgerow geometry once. Use British National Grid as the working CRS unless you have a project-specific reason not to.",
        )
        source_cols = st.columns([2, 1])
        with source_cols[0]:
            profile_options = ["bats_bankable_england_v2", "bats_v1"]
            profile_name = st.selectbox("Profile", options=profile_options, index=0, key="enrich_profile")
            working_crs = st.text_input("Working CRS", value="EPSG:27700", key="enrich_working_crs")
            input_crs = st.text_input("Input CRS (only if missing in file)", value="", key="enrich_input_crs")
            export_crs = st.text_input(
                "Export CRS (blank keeps working CRS; use 'input' to restore)",
                value="",
                key="enrich_export_crs",
            )
        with source_cols[1]:
            render_info_card(
                st,
                title="Recommended Defaults",
                body=(
                    "For bankable England-first planning: use `bats_bankable_england_v2`, keep `EPSG:27700`, enable auto-fetch for the open baseline, and review Data Readiness before moving to the planner."
                ),
                chips=["Profile: bats_bankable_england_v2", "CRS: EPSG:27700", "Auto-fetch: on"],
            )

    with tab_settings:
        render_section_intro(
            st,
            title="Core Settings",
            subtitle="These are the defaults most users should care about. Advanced inputs stay in the next tab.",
        )
        config_cols = st.columns(2)
        auto_fetch = config_cols[0].checkbox("Auto-fetch open datasets", value=True, key="enrich_auto_fetch")
        drop_all_null_features = config_cols[0].checkbox(
            "Drop derived columns that are all null",
            value=True,
            key="enrich_drop_null_features",
        )
        frozen_datasets_only = config_cols[0].checkbox(
            "Production mode (frozen datasets only)",
            value=bool(st.session_state.get("enrich_profile", "bats_bankable_england_v2") == "bats_bankable_england_v2"),
            key="enrich_frozen_datasets_only",
            help="Recommended for bankable production runs. Reuses only cached/local snapshots and blocks live fetches.",
        )
        enable_temporal_features = config_cols[1].checkbox(
            "Enable temporal features (weather/moon/deployment)",
            value=True,
            key="enrich_temporal",
        )
        enable_roost_microhabitat_proxies = config_cols[1].checkbox(
            "Enable roost and microhabitat proxy features",
            value=True,
            key="enrich_proxies",
        )
        cache_dir = st.text_input("Cache directory (optional)", value="", key="enrich_cache_dir")
        render_side_note(
            st,
            "If you are producing a screening/planning input for the current session, the enriched output will be cached automatically and offered as the default source downstream."
        )

    with tab_optional:
        render_section_intro(
            st,
            title="Optional Inputs",
            subtitle="Only open the sections you need. Most users should leave these on defaults unless a project has special data or credential requirements.",
        )
        with st.expander("Deployment and temporal settings", expanded=False):
            deployment_start_col = st.text_input("Deployment start column", value="", key="enrich_dep_start")
            deployment_end_col = st.text_input("Deployment end column", value="", key="enrich_dep_end")
            deployment_timezone = st.text_input(
                "Deployment timezone", value="Europe/London", key="enrich_dep_tz"
            )
            weather_backend = st.selectbox(
                "Weather backend",
                options=["open_meteo"],
                index=0,
                key="enrich_weather_backend",
            )
            min_night_overlap_minutes = st.number_input(
                "Min night overlap (minutes)",
                min_value=1,
                max_value=720,
                value=30,
                step=5,
                key="enrich_min_night_overlap",
            )

        with st.expander("Credentials for authenticated data sources", expanded=False):
            earthdata_token = st.text_input("NASA Earthdata token", value="", type="password", key="enrich_earthdata")
            eog_username = st.text_input("EOG username", value="", key="enrich_eog_user")
            eog_password = st.text_input("EOG password", value="", type="password", key="enrich_eog_pass")
            cred_disabled = not (earthdata_token.strip() or (eog_username.strip() and eog_password))
            if st.button(
                "Test credentials",
                disabled=cred_disabled,
                help="Lightweight validation only; does not store secrets.",
                key="enrich_test_creds",
            ):
                try:
                    results = _test_credentials(
                        earthdata_token=earthdata_token.strip() or None,
                        eog_username=eog_username.strip() or None,
                        eog_password=eog_password or None,
                    )
                    st.json(results)
                except Exception as exc:
                    st.exception(exc)

        with st.expander("Local dataset path overrides", expanded=False):
            st.caption("Leave blank to use automatic open-data fetching. Paths here override automatic sources.")
            ds_roads = st.text_input("OS Open Roads path", value="", key="enrich_ds_roads")
            ds_rivers = st.text_input("OS Open Rivers path", value="", key="enrich_ds_rivers")
            ds_worldcover = st.text_input("WorldCover path", value="", key="enrich_ds_worldcover")
            ds_viirs = st.text_input("VIIRS path", value="", key="enrich_ds_viirs")
            ds_copdem = st.text_input("Copernicus DEM path", value="", key="enrich_ds_copdem")
            ds_phi = st.text_input("NE Priority Habitat path", value="", key="enrich_ds_phi")
            ds_awi = st.text_input("NE Ancient Woodland path", value="", key="enrich_ds_awi")
            ds_lidar_dtm = st.text_input("EA LiDAR DTM path", value="", key="enrich_ds_lidar_dtm")
            ds_lidar_dsm = st.text_input("EA LiDAR DSM path", value="", key="enrich_ds_lidar_dsm")
            ds_living_england = st.text_input("Living England Habitat path", value="", key="enrich_ds_living_england")
            ds_project_lighting = st.text_input("Project lighting assets path", value="", key="enrich_ds_project_lighting")

    with tab_outputs:
        render_section_intro(
            st,
            title="Run Enrichment And Manage Outputs",
            subtitle="Run enrichment here, then reuse the cached result in GIS-only screening and static detector planning without re-uploading the layer.",
        )
        run_cols = st.columns([2, 1])
        with run_cols[0]:
            if st.button("Run GIS enrichment", type="primary", disabled=uploaded is None, key="enrich_run"):
                if uploaded is None:
                    st.warning("Upload a dataset first.")
                    return

                with tempfile.TemporaryDirectory(prefix="hedge_features_ui_") as tmp_dir:
                    tmp = Path(tmp_dir)
                    input_path = tmp / uploaded.name
                    input_path.write_bytes(uploaded.getbuffer())
                    output_path = tmp / "enriched.gpkg"
                    dataset_overrides = {
                        k: v
                        for k, v in {
                            "os_open_roads": st.session_state.get("enrich_ds_roads", ""),
                            "os_open_rivers": st.session_state.get("enrich_ds_rivers", ""),
                            "worldcover": st.session_state.get("enrich_ds_worldcover", ""),
                            "viirs_nightlights": st.session_state.get("enrich_ds_viirs", ""),
                            "copdem": st.session_state.get("enrich_ds_copdem", ""),
                            "ne_phi": st.session_state.get("enrich_ds_phi", ""),
                            "ne_awi": st.session_state.get("enrich_ds_awi", ""),
                            "ea_lidar_dtm": st.session_state.get("enrich_ds_lidar_dtm", ""),
                            "ea_lidar_dsm": st.session_state.get("enrich_ds_lidar_dsm", ""),
                            "living_england_habitat": st.session_state.get("enrich_ds_living_england", ""),
                            "project_lighting_assets": st.session_state.get("enrich_ds_project_lighting", ""),
                        }.items()
                        if str(v).strip()
                    }
                    options = RunOptions(
                        input_path=input_path,
                        output_path=output_path,
                        profile_name=profile_name,
                        working_crs=working_crs,
                        input_crs=input_crs.strip() or None,
                        export_crs=export_crs.strip() or None,
                        cache_dir=Path(cache_dir).expanduser() if cache_dir.strip() else None,
                        auto_fetch=auto_fetch,
                        credentials={
                            k: v
                            for k, v in {
                                "earthdata_token": st.session_state.get("enrich_earthdata", "").strip(),
                                "eog_username": st.session_state.get("enrich_eog_user", "").strip(),
                                "eog_password": st.session_state.get("enrich_eog_pass", ""),
                            }.items()
                            if v
                        },
                        drop_all_null_feature_columns=drop_all_null_features,
                        deployment_start_column=st.session_state.get("enrich_dep_start", "").strip() or None,
                        deployment_end_column=st.session_state.get("enrich_dep_end", "").strip() or None,
                        deployment_timezone=st.session_state.get("enrich_dep_tz", "").strip() or "Europe/London",
                        weather_backend=st.session_state.get("enrich_weather_backend", "open_meteo"),
                        min_night_overlap_minutes=int(st.session_state.get("enrich_min_night_overlap", 30)),
                        enable_temporal_features=enable_temporal_features,
                        enable_roost_microhabitat_proxies=enable_roost_microhabitat_proxies,
                        dataset_overrides=dataset_overrides,
                        write_csv=True,
                        frozen_datasets_only=bool(frozen_datasets_only),
                    )
                    try:
                        result = run_enrichment(options)
                    except Exception as exc:
                        st.exception(exc)
                        return

                    metadata_path = output_path.with_name("METADATA.json")
                    data_catalogue_path = output_path.with_name("DATA_CATALOGUE.json")
                    feature_health_path = output_path.with_name("FEATURE_HEALTH.json")
                    csv_path = output_path.with_suffix(".csv")
                    _store_enrichment_last_run(
                        st=st,
                        source_name=uploaded.name,
                        profile_name=profile_name,
                        result=result,
                        gpkg_bytes=output_path.read_bytes(),
                        csv_bytes=csv_path.read_bytes() if csv_path.exists() else None,
                        metadata_bytes=metadata_path.read_bytes() if metadata_path.exists() else None,
                        data_catalogue_bytes=data_catalogue_path.read_bytes() if data_catalogue_path.exists() else None,
                        feature_health_bytes=feature_health_path.read_bytes() if feature_health_path.exists() else None,
                    )
                    _store_inapp_enrichment_for_screening(
                        st=st,
                        output_path=output_path,
                        profile_name=profile_name,
                        result=result,
                        metadata_path=metadata_path if metadata_path.exists() else None,
                    )
                    st.success("Enrichment complete. The result is now available in screening and planning.")
            render_side_note(
                st,
                "After a successful run, the enriched GeoPackage is cached in this Streamlit session and becomes the default downstream source."
            )
        with run_cols[1]:
            if last_run is not None:
                render_info_card(
                    st,
                    title="Latest Enrichment Run",
                    body=f"Source: {last_run['source_name']}",
                    chips=[
                        f"Profile: {last_run['profile_name']}",
                        "Cached for screening",
                        "Cached for planning",
                    ],
                )
            else:
                render_info_card(
                    st,
                    title="No Enrichment Output Yet",
                    body="Run enrichment once to unlock one-click screening and planning inputs for the current session.",
                )

        last_run = st.session_state.get("enrichment_last_run")
        if last_run is not None:
            _render_enrichment_outputs(st, last_run=last_run)


def _store_inapp_enrichment_for_screening(
    *,
    st,
    output_path: Path,
    profile_name: str,
    result: dict[str, Any],
    metadata_path: Path | None,
) -> None:
    try:
        gpd = require_geopandas()
        enriched_gdf = gpd.read_file(output_path)
    except Exception as exc:
        st.warning(
            f"Enrichment output was created, but it could not be loaded into the screening tab cache: {type(exc).__name__}"
        )
        return

    base_metadata = None
    if metadata_path and metadata_path.exists():
        try:
            base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            base_metadata = None

    st.session_state["screening_inapp"] = {
        "source_name": "enriched.gpkg",
        "profile_name": profile_name,
        "result_summary": result,
        "gdf": enriched_gdf,
        "table_df": enriched_gdf.drop(columns=enriched_gdf.geometry.name).copy(),
        "base_metadata": base_metadata,
    }
    st.session_state["planner_inapp"] = {
        "source_name": "enriched.gpkg",
        "profile_name": profile_name,
        "result_summary": result,
        "gdf": enriched_gdf,
        "table_df": enriched_gdf.drop(columns=enriched_gdf.geometry.name).copy(),
        "base_metadata": base_metadata,
    }
    st.session_state["planner_prefill_route"] = "Use existing GIS enrichment route output"


def _store_enrichment_last_run(
    *,
    st,
    source_name: str,
    profile_name: str,
    result: dict[str, Any],
    gpkg_bytes: bytes,
    csv_bytes: bytes | None,
    metadata_bytes: bytes | None,
    data_catalogue_bytes: bytes | None,
    feature_health_bytes: bytes | None,
) -> None:
    st.session_state["enrichment_last_run"] = {
        "source_name": source_name,
        "profile_name": profile_name,
        "result": result,
        "gpkg_bytes": gpkg_bytes,
        "csv_bytes": csv_bytes,
        "metadata_bytes": metadata_bytes,
        "data_catalogue_bytes": data_catalogue_bytes,
        "feature_health_bytes": feature_health_bytes,
    }


def _render_enrichment_outputs(st, *, last_run: dict[str, Any]) -> None:
    render_section_intro(
        st,
        title="Outputs",
        subtitle="These downloads persist for the current Streamlit session so you do not need to rerun enrichment after minor UI changes.",
    )
    result = last_run["result"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", last_run["source_name"])
    c2.metric("Profile", last_run["profile_name"])
    outputs_ready = sum(
        1
        for key in ("gpkg_bytes", "csv_bytes", "metadata_bytes", "data_catalogue_bytes", "feature_health_bytes")
        if last_run.get(key) is not None
    )
    c3.metric("Outputs ready", outputs_ready)
    metadata = _decode_json_bytes(last_run.get("metadata_bytes"))
    c4.metric("Guidance regime", metadata.get("guidance_regime_version", "n/a"))

    st.json(result)

    download_cols = st.columns(5)
    download_cols[0].download_button(
        "Download enriched GeoPackage",
        data=last_run["gpkg_bytes"],
        file_name="enriched.gpkg",
        mime="application/geopackage+sqlite3",
        key="enrich_dl_gpkg_persisted",
    )
    if last_run.get("csv_bytes") is not None:
        download_cols[1].download_button(
            "Download enriched CSV",
            data=last_run["csv_bytes"],
            file_name="enriched.csv",
            mime="text/csv",
            key="enrich_dl_csv_persisted",
        )
    if last_run.get("metadata_bytes") is not None:
        download_cols[2].download_button(
            "Download metadata",
            data=last_run["metadata_bytes"],
            file_name="METADATA.json",
            mime="application/json",
            key="enrich_dl_metadata_persisted",
        )
    if last_run.get("data_catalogue_bytes") is not None:
        download_cols[3].download_button(
            "Download data catalogue",
            data=last_run["data_catalogue_bytes"],
            file_name="DATA_CATALOGUE.json",
            mime="application/json",
            key="enrich_dl_data_catalogue_persisted",
        )
    if last_run.get("feature_health_bytes") is not None:
        download_cols[4].download_button(
            "Download feature health",
            data=last_run["feature_health_bytes"],
            file_name="FEATURE_HEALTH.json",
            mime="application/json",
            key="enrich_dl_feature_health_persisted",
        )

    action_cols = st.columns(2)
    if action_cols[0].button("Open GIS-only Screening with in-app enrichment", key="enrich_to_screening"):
        st.session_state["screening_prefill_route"] = "Use existing GIS enrichment route output"
        st.session_state["ui_active_page"] = "GIS-only Bat Screening"
        st.rerun()
    if action_cols[1].button("Open Static Detector Planner with in-app enrichment", key="enrich_to_planner"):
        st.session_state["planner_prefill_route"] = "Use existing GIS enrichment route output"
        st.session_state["ui_active_page"] = "Static Detector Planner"
        st.rerun()


def _render_screening_tab(st) -> None:
    render_page_hero(
        st,
        eyebrow="Step 2",
        title="GIS-only Bat Screening",
        subtitle=(
            "Use enriched hedgerow features to rank survey priority, review confidence, and inspect explainability before you commit static detector effort."
        ),
    )
    render_step_chips(
        st,
        steps=["1. Choose Source", "2. Configure Framework", "3. Run Screening", "4. Review Results", "5. Export"],
        current_step="1. Choose Source",
    )

    framework_names = get_bundled_framework_names() or ["bats_screening_v1"]
    framework_name = framework_names[0]
    mode = "Default"
    default_route = st.session_state.get("screening_prefill_route", "Upload enriched CSV/XLSX")
    route_options = ["Upload enriched CSV/XLSX", "Use existing GIS enrichment route output"]
    last_run = st.session_state.get("screening_last_run")
    source_payload = None
    settings = ScreeningSettings(mode=st.session_state.get("screen_mode", "Default"))

    tab_input, tab_settings, tab_run, tab_results, tab_exports = st.tabs(
        ["1. Input", "2. Settings", "3. Run", "4. Results", "5. Exports"]
    )

    framework = None
    try:
        framework = load_framework_bundle(
            st.session_state.get("screen_framework_name", "bats_bankable_england_v2")
            if "screen_framework_name" in st.session_state
            else "bats_bankable_england_v2"
        )
    except Exception as exc:
        st.error(f"Could not load framework artefacts: {exc}")

    with tab_input:
        render_section_intro(
            st,
            title="Input Route",
            subtitle="Start with an enriched table. The easiest path is to reuse the GIS Enrichment output already cached in this session.",
        )
        input_cols = st.columns([2, 1])
        framework_name = input_cols[0].selectbox(
            "Framework package",
            options=framework_names,
            index=framework_names.index("bats_bankable_england_v2") if "bats_bankable_england_v2" in framework_names else 0,
            key="screen_framework_name",
        )
        mode = input_cols[1].radio("Mode", ["Default", "Advanced"], horizontal=False, key="screen_mode")
        route_index = route_options.index(default_route) if default_route in route_options else 0
        input_route = st.radio("Choose input route", route_options, index=route_index, key="screen_input_route")
        source_payload = _resolve_screening_source(st, input_route)
        if framework is not None:
            side_cols = st.columns([2, 1])
            with side_cols[1]:
                render_info_card(
                    st,
                    title="Framework Snapshot",
                    body=(
                        f"{framework.manifest.name} {framework.manifest.version}. Compatible with {framework.manifest.compatible_feature_profile_name}."
                    ),
                    chips=[
                        f"Predictors: {len(framework.feature_registry.predictor_order)}",
                        f"Required: {len(framework.feature_registry.required_predictors)}",
                    ],
                )
        if source_payload is None:
            st.info("Load an enriched table or use the in-session enrichment output to continue.")
        else:
            df_input = source_payload["table_df"]
            st.caption(
                f"Rows: {len(df_input)} | Columns: {len(df_input.columns)} | Source: {source_payload['source_name']}"
            )
            st.dataframe(df_input.head(50), use_container_width=True)

    if source_payload is None:
        with tab_settings:
            st.info("Choose an input source first.")
        with tab_run:
            st.info("Choose an input source first.")
        with tab_results:
            st.info("Run screening to review results.")
        with tab_exports:
            st.info("Run screening to unlock exports.")
        return

    df_input = source_payload["table_df"]
    with tab_settings:
        render_section_intro(
            st,
            title="Framework And Policy Settings",
            subtitle="Default mode is the recommended route for most teams. Only move to Advanced when you need explicit policy, threshold, or species-module control.",
        )
        settings = _render_screening_settings(
            st=st,
            mode=st.session_state.get("screen_mode", "Default"),
            framework=framework,
            source_payload=source_payload,
        )
        if framework is not None:
            with st.expander("Framework registry summary", expanded=False):
                st.write(
                    {
                        "predictor_count": len(framework.feature_registry.predictor_order),
                        "required_predictors": framework.feature_registry.required_predictors,
                        "strict_gis_prefixes": framework.feature_registry.strict_gis_prefixes,
                        "forbidden_exact": framework.feature_registry.forbidden_exact,
                        "coverage_flags": framework.feature_registry.coverage_flags,
                    }
                )

    pre_audit = None
    if framework is not None:
        try:
            pre_audit = _build_pre_audit(df_input, framework=framework, settings=settings)
        except Exception as exc:
            st.warning(f"Column audit pre-check failed: {exc}")

    with tab_run:
        render_section_intro(
            st,
            title="Run Screening",
            subtitle="Review the pre-run audit, then apply the framework. The output remains a prioritisation score and confidence layer, not a guaranteed probability.",
        )
        _render_column_audit_panel(st, pre_audit)
        if st.button("Apply GIS-only bat screening framework", type="primary", key="screen_run_btn"):
            if framework is None:
                st.error("Framework artefacts are not available, so screening cannot run.")
            else:
                _run_screening(
                    st=st,
                    df_input=df_input,
                    source_payload=source_payload,
                    framework=framework,
                    settings=settings,
                )

    last_run = st.session_state.get("screening_last_run")
    with tab_results:
        if last_run is not None:
            _render_screening_results(
                st,
                last_run=last_run,
                framework=framework,
                source_payload=source_payload,
                include_downloads=False,
            )
        else:
            st.info("Run screening to review results.")
    with tab_exports:
        if last_run is not None:
            render_section_intro(
                st,
                title="Exports",
                subtitle="Use CSV/XLSX for tabular review and GeoPackage when you need the screened results back in GIS.",
            )
            _render_downloads(st, last_run=last_run)
        else:
            st.info("Run screening to unlock exports.")


def _resolve_screening_source(st, input_route: str) -> dict[str, Any] | None:
    if input_route == "Upload enriched CSV/XLSX":
        uploaded = st.file_uploader(
            "Upload pre-enriched table (.csv or .xlsx)",
            type=["csv", "xlsx"],
            key="screen_upload_table",
        )
        if uploaded is None:
            return None
        try:
            df = read_uploaded_attribute_table(uploaded)
        except Exception as exc:
            st.error(str(exc))
            return None
        return {
            "route": "uploaded_enriched_table",
            "source_name": uploaded.name,
            "table_df": df,
            "gdf": None,
            "base_metadata": None,
            "profile_name": "bats_v1",
        }

    inapp = st.session_state.get("screening_inapp")
    if not inapp:
        st.warning("No in-app enrichment output is available yet. Run the GIS Enrichment tab first.")
        return None
    st.caption("Using enriched output cached from the current Streamlit session.")
    return {
        "route": "in_app_enrichment",
        "source_name": str(inapp.get("source_name", "enriched.gpkg")),
        "table_df": inapp["table_df"],
        "gdf": inapp.get("gdf"),
        "base_metadata": inapp.get("base_metadata"),
        "profile_name": str(inapp.get("profile_name", "bats_v1")),
    }


def _render_screening_settings(*, st, mode: str, framework, source_payload: dict[str, Any]) -> ScreeningSettings:
    st.markdown("**Screening Settings**")

    if framework is None:
        return ScreeningSettings(mode=mode)

    thresholds = framework.thresholds
    policy_options = list(thresholds.policy_thresholds.keys()) or ["Recall-first"]
    strictness_options = list(framework.confidence_rules.strictness_profiles.keys()) or ["Standard"]
    species_available = bool(framework.manifest.species_targets_available)
    is_geospatial_route = source_payload.get("route") == "in_app_enrichment" and source_payload.get("gdf") is not None

    if mode == "Default":
        st.info(
            "Default mode uses strict GIS-only predictors, Recall-first policy, packaged thresholds, and "
            "confidence override (low confidence rows are not auto-deprioritised)."
        )
        if is_geospatial_route:
            st.caption("Geospatial route detected: screened GPKG export will be available after screening.")
        return ScreeningSettings(
            mode="Default",
            policy="Recall-first" if "Recall-first" in policy_options else policy_options[0],
            confidence_strictness="Standard" if "Standard" in strictness_options else strictness_options[0],
            strict_gis_only=True,
            predictor_inclusion_mode="Strict GIS-only",
            use_packaged_band_thresholds=True,
            allow_profile_mismatch=False,
            species_module_enabled=False,
        )

    col1, col2, col3 = st.columns(3)
    policy = col1.selectbox(
        "Decision policy",
        options=policy_options,
        index=policy_options.index("Recall-first") if "Recall-first" in policy_options else 0,
        key="screen_adv_policy",
    )
    strictness = col2.selectbox(
        "Confidence strictness",
        options=strictness_options,
        index=strictness_options.index("Standard") if "Standard" in strictness_options else 0,
        key="screen_adv_strictness",
    )
    predictor_inclusion_mode = col3.selectbox(
        "Predictor inclusion",
        options=["Strict GIS-only", "Relaxed"],
        index=0,
        key="screen_adv_predictor_mode",
        help="Relaxed mode only changes audit tolerance/reporting. Scoring remains aligned to the versioned framework registry.",
    )

    col4, col5, col6 = st.columns(3)
    band_source = col4.selectbox(
        "Band threshold source",
        options=["Use packaged defaults", "Custom thresholds"],
        index=0,
        key="screen_adv_band_source",
    )
    custom_band_low = None
    custom_band_high = None
    if band_source == "Custom thresholds":
        custom_band_low = col5.number_input(
            "Low -> Medium cutoff",
            min_value=0.0,
            max_value=1.0,
            value=float(thresholds.band_low_upper),
            step=0.01,
            key="screen_adv_band_low",
        )
        custom_band_high = col6.number_input(
            "Medium -> High cutoff",
            min_value=0.0,
            max_value=1.0,
            value=float(thresholds.band_high_lower),
            step=0.01,
            key="screen_adv_band_high",
        )
    else:
        col5.caption(f"Low/Medium: {thresholds.band_low_upper:.2f}")
        col6.caption(f"Medium/High: {thresholds.band_high_lower:.2f}")

    col7, col8, col9 = st.columns(3)
    use_custom_policy_threshold = col7.checkbox(
        "Custom policy threshold",
        value=False,
        key="screen_adv_use_custom_policy_th",
    )
    custom_policy_threshold = None
    if use_custom_policy_threshold:
        custom_policy_threshold = col8.number_input(
            "Policy threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(thresholds.policy_thresholds.get(policy, 0.5)),
            step=0.01,
            key="screen_adv_policy_th",
        )
    else:
        col8.caption(f"Policy threshold: {thresholds.policy_thresholds.get(policy, 0.5):.2f}")

    allow_profile_mismatch = col9.checkbox(
        "Allow profile mismatch (warn only)",
        value=False,
        key="screen_adv_profile_mismatch",
        help="Only use if you know the uploaded table was enriched with a compatible feature profile.",
    )

    col10, col11 = st.columns(2)
    species_enabled = col10.checkbox(
        "Species-target screening module",
        value=False,
        disabled=not species_available,
        key="screen_adv_species_module",
        help="Optional module is off by default and only available when compatible species artefacts are installed.",
    )
    if not species_available:
        col10.caption("No compatible species artefacts installed in this framework package.")
    if is_geospatial_route:
        col11.caption("Geospatial route detected: screened GPKG export available after screening.")

    if predictor_inclusion_mode == "Relaxed":
        st.warning(
            "Relaxed mode is an advanced audit setting. The screening score still uses the strict versioned framework "
            "predictor registry to avoid leakage and drift."
        )

    return ScreeningSettings(
        mode="Advanced",
        policy=policy,
        confidence_strictness=strictness,
        strict_gis_only=(predictor_inclusion_mode == "Strict GIS-only"),
        predictor_inclusion_mode=predictor_inclusion_mode,
        use_packaged_band_thresholds=(band_source == "Use packaged defaults"),
        custom_band_low_upper=float(custom_band_low) if custom_band_low is not None else None,
        custom_band_high_lower=float(custom_band_high) if custom_band_high is not None else None,
        custom_policy_threshold=float(custom_policy_threshold) if custom_policy_threshold is not None else None,
        allow_profile_mismatch=allow_profile_mismatch,
        species_module_enabled=species_enabled,
    )


def _build_pre_audit(df, *, framework, settings: ScreeningSettings):
    audit = build_column_audit(df, framework=framework, settings=settings)
    try:
        alignment = align_predictors(df, framework=framework)
        X = alignment.predictor_df
        if len(X.columns) > 0:
            cfg = framework.confidence_rules.strictness_profiles.get(
                settings.confidence_strictness,
                framework.confidence_rules.strictness_profiles.get("Standard", {}),
            )
            severe_threshold = float(
                cfg.get("severe_missingness_threshold", cfg.get("low_coverage_threshold", 0.45))
            )
            coverage_pct = X.notna().sum(axis=1).astype("float64") / float(len(X.columns))
            audit.severe_missingness_rows = int((coverage_pct < severe_threshold).sum())
    except Exception:
        pass
    return audit


def _render_column_audit_panel(st, audit) -> None:
    st.markdown("**Column Audit**")
    if audit is None:
        st.info("Column audit will appear after a valid enriched table is loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total columns", audit.total_columns)
    c2.metric("GIS predictors detected", len(audit.detected_gis_predictor_columns))
    c3.metric("Missing expected features", len(audit.missing_expected_features))
    c4.metric("Severe missingness rows", audit.severe_missingness_rows)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Rows", audit.total_rows)
    c6.metric("Excluded columns", len(audit.excluded_columns))
    c7.metric("Status columns found", len(audit.status_columns_found))
    c8.metric("Coverage flags found", len(audit.coverage_flags_found))

    with st.expander("Audit details", expanded=False):
        st.write({"gis_feature_families_detected": audit.gis_feature_families_detected})
        st.write({"missing_required_features": audit.missing_required_features})
        st.write({"missing_expected_features_sample": audit.missing_expected_features[:50]})
        st.write({"status_columns_found": audit.status_columns_found})
        st.write({"coverage_flags_found": audit.coverage_flags_found})
        st.write({"excluded_columns": audit.excluded_columns})
        st.write({"extra_columns_sample": audit.extra_columns[:100]})


def _run_screening(*, st, df_input, source_payload: dict[str, Any], framework, settings: ScreeningSettings) -> None:
    try:
        run = screen_dataframe(
            df_input,
            framework=framework,
            settings=settings,
            prediction_route=source_payload["route"],
            feature_profile_name=source_payload.get("profile_name", framework.manifest.compatible_feature_profile_name),
            input_label=source_payload.get("source_name"),
        )
    except ScreeningInputError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Screening failed: {exc}")
        return

    if settings.species_module_enabled and not framework.manifest.species_targets_available:
        st.warning("Species-target screening is enabled in settings, but no compatible species artefacts are installed.")

    st.session_state["screening_last_run"] = {
        "results_df": run.results_df,
        "run_summary": run.run_summary,
        "warnings": run.warnings,
        "column_audit": run.column_audit,
        "source_route": source_payload["route"],
        "source_name": source_payload["source_name"],
        "base_metadata": source_payload.get("base_metadata"),
        "source_gdf": source_payload.get("gdf"),
        "source_table_columns": list(df_input.columns),
        "settings": settings,
        "framework_name": framework.manifest.name,
        "framework_version": framework.manifest.version,
    }
    st.success("GIS-only screening complete.")
    if run.warnings:
        for msg in run.warnings:
            st.warning(msg)


def _render_screening_results(
    st,
    *,
    last_run: dict[str, Any],
    framework,
    source_payload: dict[str, Any],
    include_downloads: bool = True,
) -> None:
    import pandas as pd

    results_df = last_run["results_df"]
    summary = last_run["run_summary"]
    audit = last_run["column_audit"]

    st.divider()
    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows screened", len(results_df))
    c2.metric("High priority", int((results_df["survey_priority_band"] == "High").sum()))
    c3.metric(
        "Low confidence rows",
        int((results_df["confidence_level"] == "Low").sum()),
        help="These rows are not auto-deprioritised.",
    )
    c4.metric(
        "Review-required rows",
        int((results_df["recommended_action"] == LOW_CONFIDENCE_ACTION).sum()),
    )

    st.caption(
        "Scores are ranking scores (`survey_priority_score`) for prioritisation, not calibrated probabilities. "
        "Confidence and reason codes are shown separately and can override automated deprioritisation."
    )

    _render_column_audit_panel(st, audit)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Score distribution**")
        hist = _score_histogram_df(results_df["survey_priority_score"])
        if not hist.empty:
            st.bar_chart(hist.set_index("bin_label")["count"])
        else:
            st.info("No scores to plot.")

    with col_b:
        st.markdown("**Recommended action by confidence**")
        ctab = pd.crosstab(results_df["recommended_action"], results_df["confidence_level"])
        if not ctab.empty:
            st.bar_chart(ctab)
        else:
            st.info("No action/confidence counts to plot.")

    st.markdown("**Reason code frequencies**")
    reason_freq = summary.get("reason_code_frequencies", {})
    if reason_freq:
        reason_df = pd.DataFrame(
            [{"reason_code": k, "count": int(v)} for k, v in reason_freq.items()]
        )
        st.dataframe(reason_df, use_container_width=True)
    else:
        st.info("No reason codes triggered.")

    st.markdown("**Preview (final outputs)**")
    preview_cols = _preferred_preview_columns(results_df.columns)
    st.dataframe(results_df[preview_cols].head(100), use_container_width=True)

    _render_row_inspection(st, results_df=results_df, framework=framework)
    if include_downloads:
        _render_downloads(st, last_run=last_run)


def _preferred_preview_columns(columns) -> list[str]:
    preferred = [
        "hf_uid",
        "survey_priority_score",
        "survey_priority_band",
        "confidence_level",
        "reason_codes",
        "recommended_action",
        "gis_feature_coverage_pct",
        "missing_required_feature_count",
        "major_reason_code_count",
    ]
    out = [c for c in preferred if c in columns]
    for c in columns:
        if c not in out:
            out.append(c)
        if len(out) >= 20:
            break
    return out


def _score_histogram_df(score_series):
    import pandas as pd

    if len(score_series) == 0:
        return pd.DataFrame(columns=["bin_label", "count"])
    bins = pd.interval_range(start=0, end=1, periods=10, closed="left")
    cut = pd.cut(score_series.astype(float), bins=bins)
    counts = cut.value_counts(sort=False).fillna(0).astype(int)
    labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in counts.index]
    return pd.DataFrame({"bin_label": labels, "count": counts.values})


def _render_row_inspection(st, *, results_df, framework) -> None:
    st.markdown("**Row-level inspection**")
    if len(results_df) == 0:
        st.info("No rows available.")
        return

    id_col = "hf_uid" if "hf_uid" in results_df.columns else None
    if id_col:
        options = results_df[id_col].astype(str).tolist()
        selected = st.selectbox("Inspect hedgerow", options=options, key="screen_row_inspect_uid")
        row = results_df.loc[results_df[id_col].astype(str) == str(selected)].iloc[0]
    else:
        idx_options = [str(i) for i in results_df.index.tolist()]
        selected = st.selectbox("Inspect row index", options=idx_options, key="screen_row_inspect_idx")
        row = results_df.iloc[int(selected)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{float(row['survey_priority_score']):.3f}")
    c2.metric("Band", str(row["survey_priority_band"]))
    c3.metric("Confidence", str(row["confidence_level"]))
    c4.metric("Action", str(row["recommended_action"]))

    reason_codes = [x for x in str(row.get("reason_codes", "")).split("|") if x]
    st.write({"reason_codes": reason_codes})
    st.write(
        {
            "gis_feature_coverage_pct": float(row.get("gis_feature_coverage_pct", 0.0)),
            "missing_required_feature_count": int(row.get("missing_required_feature_count", 0)),
            "major_reason_code_count": int(row.get("major_reason_code_count", 0)),
        }
    )

    with st.expander("Feature coverage summary", expanded=False):
        predictor_names = []
        if framework is not None:
            predictor_names = framework.feature_registry.predictor_order
        present = [c for c in predictor_names if c in results_df.columns and not _is_missing_value(row.get(c))]
        missing = []
        for c in predictor_names:
            if c not in results_df.columns:
                missing.append(c)
                continue
            val = row.get(c)
            if _is_missing_value(val):
                missing.append(c)
        st.write(
            {
                "predictor_count": len(predictor_names),
                "present_predictor_values": len(present),
                "missing_predictor_values": len(missing),
                "missing_predictor_sample": missing[:30],
            }
        )
        st.write({"key_feature_values": _row_key_feature_snapshot(row)})

    with st.expander("Explainability (fallback summary)", expanded=False):
        st.caption(
            "SHAP-style explanations are not bundled in this framework package. "
            "Fallback summary shows missing/unusual features and key feature values used in scoring."
        )
        st.write({"top_missing_or_flagged_context": _row_flagged_context(row)})


def _row_key_feature_snapshot(row) -> dict[str, Any]:
    keys = [
        "geom_length_m",
        "geom_sinuosity",
        "net_degree_max",
        "dist_os_road_m",
        "dist_os_river_m",
        "buf100_worldcover_tree_pct",
        "buf100_worldcover_built_pct",
        "roostpx_struct_proxy_score",
        "mhb_roost_proxy_score",
        "survey_priority_score",
        "survey_priority_band",
        "confidence_level",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        if key in row.index:
            out[key] = row[key]
    return out


def _row_flagged_context(row) -> list[str]:
    msgs: list[str] = []
    if str(row.get("reason_codes", "")).strip():
        msgs.append(f"reason_codes={row['reason_codes']}")
    if float(row.get("gis_feature_coverage_pct", 0.0)) < 0.45:
        msgs.append("Low GIS feature coverage for scoring.")
    if int(row.get("missing_required_feature_count", 0)) > 0:
        msgs.append("One or more required predictors are missing.")
    return msgs


def _is_missing_value(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return value is None


def _render_downloads(st, *, last_run: dict[str, Any]) -> None:
    st.markdown("**Downloads**")
    results_df = last_run["results_df"]
    run_summary = last_run["run_summary"]
    source_name = str(last_run.get("source_name", "screened"))

    base_stem = Path(source_name).stem or "screened_results"
    csv_name = f"{base_stem}_screened.csv"
    xlsx_name = f"{base_stem}_screened.xlsx"

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download screened results (CSV)",
        data=dataframe_to_csv_bytes(results_df),
        file_name=csv_name,
        mime="text/csv",
        key="screen_dl_csv",
    )
    try:
        xlsx_bytes = dataframe_to_xlsx_bytes(results_df)
        col2.download_button(
            "Download screened results (XLSX)",
            data=xlsx_bytes,
            file_name=xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="screen_dl_xlsx",
        )
    except Exception as exc:
        col2.warning(f"XLSX export unavailable: {exc}")

    col3.download_button(
        "Download run summary JSON",
        data=json_bytes(run_summary),
        file_name=f"{base_stem}_screening_run_summary.json",
        mime="application/json",
        key="screen_dl_run_summary",
    )

    source_gdf = last_run.get("source_gdf")
    if source_gdf is not None:
        with st.expander("Geospatial outputs (Route 2)", expanded=False):
            try:
                screened_gdf = _build_screened_geodataframe(last_run)
                gpkg_bytes = _gdf_to_gpkg_bytes(screened_gdf)
                st.download_button(
                    "Download screened GeoPackage (screening columns appended)",
                    data=gpkg_bytes,
                    file_name=f"{base_stem}_screened.gpkg",
                    mime="application/geopackage+sqlite3",
                    key="screen_dl_gpkg",
                )
            except Exception as exc:
                st.warning(f"Could not prepare screened GeoPackage: {exc}")

            combined_metadata = _merge_metadata_with_screening(last_run)
            if combined_metadata is not None:
                st.download_button(
                    "Download metadata (enrichment + screening)",
                    data=json_bytes(combined_metadata),
                    file_name=f"{base_stem}_METADATA_screening.json",
                    mime="application/json",
                    key="screen_dl_metadata",
                )


def _build_screened_geodataframe(last_run: dict[str, Any]):
    gdf = last_run["source_gdf"].copy()
    results_df = last_run["results_df"]
    source_cols = set(last_run.get("source_table_columns") or [])
    appended_cols = [c for c in results_df.columns if c not in source_cols]
    for col in appended_cols:
        gdf[col] = results_df[col].values
    return gdf


def _gdf_to_gpkg_bytes(gdf) -> bytes:
    with tempfile.TemporaryDirectory(prefix="hedge_features_screened_gpkg_") as tmp_dir:
        out_path = Path(tmp_dir) / "screened.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        return out_path.read_bytes()


def _merge_metadata_with_screening(last_run: dict[str, Any]) -> dict[str, Any] | None:
    base = last_run.get("base_metadata")
    if base is None:
        return {"screening": last_run["run_summary"]}
    merged = dict(base)
    merged["screening"] = last_run["run_summary"]
    return merged


if __name__ == "__main__":  # pragma: no cover
    run_app()
