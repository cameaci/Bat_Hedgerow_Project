from __future__ import annotations

from typing import Any


NAV_PAGES = [
    "Overview",
    "GIS Enrichment",
    "Data Readiness",
    "GIS-only Bat Screening",
    "Species Strategy",
    "Static Detector Planner",
]


def inject_global_styles(st) -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }
        .hf-hero {
            padding: 1.15rem 1.25rem;
            border: 1px solid rgba(20, 44, 32, 0.10);
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(244,248,241,1) 0%, rgba(233,242,236,1) 100%);
            margin-bottom: 1rem;
        }
        .hf-eyebrow {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #35553d;
            margin-bottom: 0.35rem;
        }
        .hf-title {
            font-size: 2rem;
            font-weight: 700;
            color: #16251a;
            margin: 0 0 0.35rem 0;
        }
        .hf-subtitle {
            font-size: 1rem;
            color: #33443a;
            margin: 0;
            max-width: 78ch;
        }
        .hf-card {
            border: 1px solid rgba(20, 44, 32, 0.10);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(20, 44, 32, 0.05);
            margin-bottom: 0.85rem;
        }
        .hf-card h4 {
            margin: 0 0 0.35rem 0;
            color: #16251a;
            font-size: 1.02rem;
        }
        .hf-card p {
            margin: 0;
            color: #415348;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .hf-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.6rem 0 0.15rem 0;
        }
        .hf-chip {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: #edf4ee;
            color: #2f4b38;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(47, 75, 56, 0.12);
        }
        .hf-chip-active {
            background: #183f2a;
            color: white;
            border-color: #183f2a;
        }
        .hf-chip-warning {
            background: #fff4dd;
            color: #7a5717;
            border-color: rgba(122, 87, 23, 0.16);
        }
        .hf-chip-neutral {
            background: #f3f5f4;
            color: #55645b;
            border-color: rgba(85, 100, 91, 0.10);
        }
        .hf-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #16251a;
            margin: 0 0 0.25rem 0;
        }
        .hf-section-subtitle {
            font-size: 0.95rem;
            color: #4a5c51;
            margin: 0 0 0.9rem 0;
            max-width: 80ch;
        }
        .hf-side-note {
            border-left: 4px solid #7fa787;
            padding: 0.7rem 0.85rem;
            background: #f5f9f5;
            border-radius: 0 12px 12px 0;
            color: #314338;
            font-size: 0.92rem;
        }
        .hf-overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 0.85rem;
            margin: 0.75rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(st, *, workflow_state: dict[str, dict[str, Any]], active_page: str) -> str:
    with st.sidebar:
        st.markdown("### Survey Workflow")
        st.caption("Use the app as a guided sequence: enrich, screen, then plan and evidence.")
        page = st.radio(
            "Navigate",
            NAV_PAGES,
            index=NAV_PAGES.index(active_page) if active_page in NAV_PAGES else 0,
            key="hf_sidebar_nav",
        )
        st.divider()
        st.markdown("### Session Status")
        for label in NAV_PAGES[1:]:
            state = workflow_state.get(label, {})
            st.markdown(f"**{label}**")
            st.caption(str(state.get("status_text", "Not started")))
        st.divider()
        st.caption(
            "Recommended operating pattern: run GIS Enrichment once, review Data Readiness and Species Strategy, then screen and design the detector set in Static Detector Planner."
        )
    return page


def render_page_hero(st, *, eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hf-hero">
          <div class="hf-eyebrow">{eyebrow}</div>
          <div class="hf-title">{title}</div>
          <p class="hf-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_chips(st, *, steps: list[str], current_step: str) -> None:
    chips = []
    for step in steps:
        css_class = "hf-chip hf-chip-active" if step == current_step else "hf-chip hf-chip-neutral"
        chips.append(f'<span class="{css_class}">{step}</span>')
    st.markdown(f'<div class="hf-chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_section_intro(st, *, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hf-section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hf-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_info_card(st, *, title: str, body: str, chips: list[str] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="hf-chip-row">' + "".join(f'<span class="hf-chip">{chip}</span>' for chip in chips) + "</div>"
    st.markdown(
        f"""
        <div class="hf-card">
          <h4>{title}</h4>
          <p>{body}</p>
          {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_side_note(st, text: str) -> None:
    st.markdown(f'<div class="hf-side-note">{text}</div>', unsafe_allow_html=True)
