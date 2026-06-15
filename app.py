"""England Bat Hedgerow Suitability Index — single-page Streamlit app.

Upload a hedgerow shapefile -> get a ranked, scored list of which hedgerows to
prioritise for static bat surveys, with a map and downloads. Weights have sensible
defaults (collapsed "Fine-tuning" panel); results can be filtered/sorted by any
sub-index.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from hedge_features.exceptions import InputValidationError
from hedge_features.utils import sha1_text
from hsi import config
from hsi.calibration import calibrate, join_activity
from hsi.explain import explain_hedgerow, sensitivity
from hsi.pipeline import run_features
from hsi.report import build_method_statement, build_run_metadata, write_outputs
from hsi.score import apply_scoring
from hsi.uploads import ACCEPTED_UPLOAD_EXTENSIONS, save_uploaded_geodata, uploaded_files_fingerprint

st.set_page_config(page_title="Bat Hedgerow HSI (England)", layout="wide")

CATEGORY_COLORS = {
    "Excellent": "#1a9641",
    "Good": "#fdae61",
    "Poor": "#d7191c",
    "Incomplete": "#9e9e9e",
}
CONFIDENCE_COLORS = {
    "High": "#1a9641",
    "Medium": "#fdae61",
    "Low": "#d7191c",
    "Incomplete": "#9e9e9e",
}

SI_TABLE_COLS = [f"hsi_{k}_score" for k in config.SI_KEYS]


# --------------------------------------------------------------------------------------
# Sidebar — inputs and fine-tuning
# --------------------------------------------------------------------------------------

def sidebar_controls():
    # Calibration can suggest weights; apply them before the sliders are instantiated.
    pending = st.session_state.pop("_pending_si_weights", None)
    if pending:
        for k, v in pending.items():
            st.session_state[f"w_{k}"] = float(v)

    st.sidebar.header("1 · Input")
    uploaded = st.sidebar.file_uploader(
        "Hedgerow layer",
        type=ACCEPTED_UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
        help=(
            "Upload one GeoPackage, GeoJSON or zipped shapefile. For a direct shapefile upload, "
            "select its matching .shp, .shx and .dbf files together (plus .prj if available)."
        ),
    )
    input_crs = st.sidebar.text_input(
        "Input CRS (only if the file has no .prj)", value="", placeholder="e.g. EPSG:27700"
    ).strip() or None
    allow_live = st.sidebar.toggle(
        "Fetch live open data (OSM / WorldCover / Natural England)",
        value=True,
        help="If off, only locally pre-downloaded data in data/ is used.",
    )

    st.sidebar.header("2 · Fine-tuning")
    with st.sidebar.expander("Weights & blend (defaults are sensible)", expanded=False):
        st.caption("You normally won't need to change these.")
        alpha = st.slider(
            "Structure vs context (alpha)", 0.0, 1.0, config.DEFAULT_ALPHA, 0.05,
            help="1.0 = pure WSP structural score; 0.0 = pure landscape context.",
        )
        st.markdown("**SI weights (structural)**")
        si_weights = {
            k: st.slider(f"{config.SI_LABELS[k]}", 0.0, 3.0, config.DEFAULT_SI_WEIGHTS[k], 0.1, key=f"w_{k}")
            for k in config.SI_KEYS
        }
        st.markdown("**Context weights**")
        ctx_weights = {
            k: st.slider(config.CONTEXT_LABELS[k], 0.0, 1.0, config.DEFAULT_CONTEXT_WEIGHTS[k], 0.05, key=f"w_{k}")
            for k in config.CONTEXT_KEYS
        }
        if st.button("Reset to defaults"):
            for k in list(st.session_state.keys()):
                if k.startswith("w_"):
                    del st.session_state[k]
            st.rerun()

    settings = config.ScoreSettings(si_weights=si_weights, context_weights=ctx_weights, alpha=alpha)
    return uploaded, input_crs, allow_live, settings


# --------------------------------------------------------------------------------------
# Heavy GIS stage (cached in session_state)
# --------------------------------------------------------------------------------------

def compute_features(uploaded_files, input_crs, allow_live):
    upload_key = uploaded_files_fingerprint(uploaded_files)
    key = sha1_text(f"{upload_key}:{input_crs}:{allow_live}", length=20)
    if st.session_state.get("features_key") == key:
        return st.session_state["features"]

    with st.status("Running GIS feature extraction…", expanded=True) as status:
        st.write("Reading hedgerows and acquiring open data (this can take a minute)…")
        with tempfile.TemporaryDirectory(prefix="hsi_upload_") as tmp_dir:
            source_path = save_uploaded_geodata(uploaded_files, tmp_dir)
            result = run_features(source_path, input_crs=input_crs, allow_live_fetch=allow_live)
        status.update(label=f"Features ready for {len(result.gdf)} hedgerows.", state="complete")

    st.session_state["features"] = result
    st.session_state["features_key"] = key
    return result


# --------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------

def render_map(gdf, *, color_field: str = "hsi_wsp_category", color_map: dict | None = None):
    color_map = color_map or CATEGORY_COLORS
    try:
        import folium
        from streamlit_folium import st_folium
    except Exception:
        st.info("Install folium + streamlit-folium to see the map. Showing centroids instead.")
        pts = gdf.to_crs("EPSG:4326")
        st.map(pd.DataFrame({"lat": pts.geometry.centroid.y, "lon": pts.geometry.centroid.x}))
        return

    wgs = gdf.to_crs("EPSG:4326")
    centroid = wgs.geometry.union_all().centroid if hasattr(wgs.geometry, "union_all") else wgs.geometry.unary_union.centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles="CartoDB positron")

    def style_fn(feat):
        val = feat["properties"].get(color_field, "Incomplete")
        return {"color": color_map.get(val, "#9e9e9e"), "weight": 4, "opacity": 0.9}

    tooltip_fields = [c for c in ["hf_uid", "hsi_priority_rank", "hsi_priority", "hsi_wsp_category",
                                  "hsi_confidence_level", "hsi_survey_requirement"] if c in wgs.columns]
    keep = tooltip_fields + ["geometry"]
    folium.GeoJson(
        wgs[keep].to_json(),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),
    ).add_to(fmap)
    legend = "&nbsp;".join(f'<span style="color:{c}">&#9632;</span> {name}' for name, c in color_map.items())
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:6px 10px;border-radius:4px;font-size:12px">{legend}</div>'
    ))
    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    st.title("🦇 Bat Hedgerow Suitability Index — England")
    st.caption(
        "Upload hedgerows → get a prioritised list for static bat surveys. "
        "Transparent WSP/HyNet 7-index scoring (SI1–SI7) plus a landscape-context layer."
    )

    uploaded, input_crs, allow_live, settings = sidebar_controls()

    if not uploaded:
        st.info(
            "⬅️ Upload a hedgerow layer to begin. For full structural scoring (height, width, "
            "gappiness, trees) drop EA 1 m LiDAR tiles into `data/lidar/dtm` and `data/lidar/dsm` "
            "— see `data/README.md`. Without LiDAR the tool still ranks hedgerows using land-cover, "
            "water, woodland, roost and darkness context, and flags the WSP category as Incomplete."
        )
        return

    try:
        features = compute_features(uploaded, input_crs, allow_live)
    except InputValidationError as exc:
        st.error(str(exc))
        return
    scored = apply_scoring(features.gdf, settings=settings)

    # ---- result filters ----
    st.sidebar.header("3 · Result filters")
    categories = sorted(scored["hsi_wsp_category"].dropna().unique().tolist())
    sel_categories = st.sidebar.multiselect("WSP category", categories, default=categories)
    confidences = sorted(scored["hsi_confidence_level"].dropna().unique().tolist())
    sel_conf = st.sidebar.multiselect("Confidence", confidences, default=confidences)
    only_no_fieldcheck = st.sidebar.toggle("Hide rows needing field verification", value=False)

    sort_options = {
        "Survey priority (recommended)": "hsi_priority",
        "WSP score": "hsi_wsp_score",
        **{f"{config.SI_LABELS[k]} (SI{i+1})": f"hsi_{k}_score" for i, k in enumerate(config.SI_KEYS)},
        **{config.CONTEXT_LABELS[k]: k for k in config.CONTEXT_KEYS},
    }
    sort_label = st.sidebar.selectbox("Prioritise by", list(sort_options.keys()), index=0)
    sort_col = sort_options[sort_label]

    view = scored
    if sel_categories:
        view = view[view["hsi_wsp_category"].isin(sel_categories)]
    if sel_conf:
        view = view[view["hsi_confidence_level"].isin(sel_conf)]
    if only_no_fieldcheck and "field_verification_required" in view.columns:
        view = view[~view["field_verification_required"].astype(bool)]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=False, na_position="last")

    # ---- summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hedgerows", len(scored))
    c2.metric("Shown", len(view))
    excellent = int((scored["hsi_wsp_category"] == "Excellent").sum())
    c3.metric("Excellent", excellent)
    incomplete = int((scored["hsi_wsp_category"] == "Incomplete").sum())
    c4.metric("Incomplete (need LiDAR/field)", incomplete)

    tab_table, tab_map, tab_explain, tab_calibrate, tab_data, tab_method = st.tabs(
        ["📋 Ranked table", "🗺️ Map", "🔍 Explain", "🎯 Calibrate", "🔌 Data sources", "📄 Method & downloads"]
    )

    with tab_table:
        display_cols = (
            ["hf_uid", "hsi_priority_rank", "hsi_priority", "hsi_wsp_category", "hsi_wsp_score"]
            + SI_TABLE_COLS
            + ["hsi_confidence_level", "hsi_survey_requirement"]
        )
        display_cols = [c for c in display_cols if c in view.columns]
        st.dataframe(
            view[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=560,
        )

    with tab_map:
        color_choice = st.radio("Colour by", ["WSP category", "Confidence"], horizontal=True)
        if len(view):
            if color_choice == "Confidence":
                render_map(view, color_field="hsi_confidence_level", color_map=CONFIDENCE_COLORS)
            else:
                render_map(view, color_field="hsi_wsp_category", color_map=CATEGORY_COLORS)
        else:
            st.warning("No hedgerows match the current filters.")

    with tab_explain:
        st.subheader("Why did a hedgerow score the way it did?")
        pool = view if len(view) else scored
        id_list = pool.sort_values("hsi_priority_rank")["hf_uid"].tolist()
        if not id_list:
            st.warning("No hedgerows to explain.")
        else:
            sel = st.selectbox("Hedgerow", id_list)
            expl = explain_hedgerow(scored, sel, settings=settings)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Priority", expl["priority"])
            m2.metric("Structure A", expl["structural_A"])
            m3.metric("Context B", expl["context_B"])
            m4.metric("WSP category", expl["wsp_category"])
            contrib = pd.DataFrame(expl["contributions"])
            st.caption("Each bar is that factor's additive share of the final priority (bars sum to the priority).")
            st.bar_chart(contrib.set_index("factor")["contribution"])
            with st.expander("Contribution detail"):
                st.dataframe(contrib, use_container_width=True, hide_index=True)
            st.markdown("**Global sensitivity** — how much moving each weight changes the rankings:")
            sens = pd.DataFrame(sensitivity(scored, settings=settings))
            st.bar_chart(sens.set_index("factor")["impact"])

    with tab_calibrate:
        st.subheader("Validate / calibrate against survey activity")
        st.caption(
            "Upload a table with a column matching one of your hedgerow attributes (an ID or name) plus an "
            "activity/count column (e.g. crossing-point counts). The tool reports how well the HSI tracks "
            "your data and can suggest re-tuned weights."
        )
        up = st.file_uploader("Activity table (CSV)", type=["csv"], key="cal_csv")
        if up is not None:
            adf = pd.read_csv(up)
            st.dataframe(adf.head(), use_container_width=True, hide_index=True)
            cols = list(adf.columns)
            shared = [c for c in cols if c in scored.columns]
            join_col = st.selectbox("Join column (must also exist in your hedgerow layer)", shared or cols)
            act_col = st.selectbox("Activity / count column", [c for c in cols if c != join_col])
            optimize = st.checkbox("Also suggest re-tuned SI weights", value=False)
            if st.button("Run calibration"):
                if join_col not in scored.columns:
                    st.error(f"'{join_col}' is not a column in your hedgerow layer; pick a shared key.")
                else:
                    joined = join_activity(scored, adf, activity_col=act_col, join_col=join_col)
                    matched = int(joined["activity"].notna().sum())
                    st.write(f"Matched **{matched}** of {len(scored)} hedgerows.")
                    report = calibrate(joined, activity_col="activity", settings=settings, optimize=optimize)
                    if "error" in report:
                        st.warning(report["error"])
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Spearman (priority)", report["spearman_priority_vs_activity"])
                        c2.metric("Spearman (structure A)", report["spearman_structuralA_vs_activity"])
                        c3.metric("AUC (presence)", report["auc_priority_vs_presence"])
                        st.info(report["interpretation"])
                        if "suggested_si_weights" in report:
                            st.write("**Suggested SI weights** (validate before adopting):")
                            st.json(report["suggested_si_weights"])
                            st.caption(report.get("optimize_note", ""))
                            if st.button("Apply suggested weights to the fine-tuning panel"):
                                st.session_state["_pending_si_weights"] = report["suggested_si_weights"]
                                st.rerun()

    with tab_data:
        st.write("Which data sources returned data for this AOI:")
        st.table(
            __import__("pandas").DataFrame(
                [{"dataset": k, "status": v} for k, v in sorted(features.dataset_status.items())]
            )
        )
        with st.expander("Run notes"):
            for note in features.notes:
                st.write("•", note)

    with tab_method:
        metadata = build_run_metadata(
            settings=settings,
            dataset_status=features.dataset_status,
            dataset_metadata=features.dataset_metadata,
            feature_count=len(scored),
            notes=features.notes,
        )
        method_md = build_method_statement(scored, settings=settings, dataset_status=features.dataset_status)

        out_dir = Path(tempfile.mkdtemp(prefix="hsi_out_"))
        out_path = out_dir / "hedgerow_hsi.gpkg"
        written = write_outputs(view if len(view) else scored, out_path, metadata=metadata)

        st.subheader("Downloads")
        col_a, col_b, col_c = st.columns(3)
        if "gpkg" in written:
            col_a.download_button("GeoPackage (.gpkg)", Path(written["gpkg"]).read_bytes(), file_name="hedgerow_hsi.gpkg")
        if "csv" in written:
            col_b.download_button("Attributes (.csv)", Path(written["csv"]).read_bytes(), file_name="hedgerow_hsi.csv")
        if "shapefile_zip" in written:
            col_c.download_button("Shapefile (.zip)", Path(written["shapefile_zip"]).read_bytes(), file_name="hedgerow_hsi_shapefile.zip")
        st.download_button("Method statement (.md)", method_md.encode("utf-8"), file_name="HSI_method_statement.md")

        st.subheader("Method statement")
        st.markdown(method_md)


if __name__ == "__main__":
    main()
