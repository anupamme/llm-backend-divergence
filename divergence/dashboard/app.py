"""Streamlit dashboard for LLM backend divergence analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from divergence.analysis.logprob_divergence import (
    compute_logprob_divergence,
    visualize_token_divergence,
)
from divergence.analysis.output_divergence import compute_output_divergence
from divergence.dashboard.data_loader import (
    get_available_datasets,
    load_canary_breakdown,
    load_latency_stats,
    load_mmlu_subject_stats,
)

VERDICT_ORDER = {"dispersed": 0, "split": 1, "majority": 2, "unanimous": 3}

SAFE_COLORS = [
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
]


def _parse_db_arg() -> str:
    """Parse --db path from sys.argv."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            path = args[i + 1]
            if Path(path).exists():
                return path
            st.error(f"Database file not found: {path}")
            st.stop()
    st.error("Usage: `streamlit run divergence/dashboard/app.py -- --db <path>`")
    st.stop()
    return ""  # unreachable


def page_overview(db_path: str) -> None:
    """Overview page: latency distributions, error rates, token counts."""
    st.header("Overview")

    stats = load_latency_stats(db_path)
    if not stats:
        st.info("No inference results found in this database.")
        return

    # Metrics row
    cols = st.columns(len(stats))
    for i, s in enumerate(stats):
        with cols[i]:
            error_rate = s.error_count / s.total_count * 100 if s.total_count > 0 else 0
            st.metric(s.backend_name, f"{s.total_tokens:,} tokens")
            st.caption(
                f"{s.total_count} items | {s.error_count} errors ({error_rate:.1f}%)"
            )

    # TTFT box plot
    st.subheader("Time to First Token (ms)")
    fig_ttft = go.Figure()
    for i, s in enumerate(stats):
        if s.ttft_values:
            fig_ttft.add_trace(
                go.Box(
                    y=s.ttft_values,
                    name=s.backend_name,
                    marker_color=SAFE_COLORS[i % len(SAFE_COLORS)],
                )
            )
    fig_ttft.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_ttft, use_container_width=True)

    # ITL box plot
    st.subheader("Inter-Token Latency (ms)")
    fig_itl = go.Figure()
    for i, s in enumerate(stats):
        if s.itl_values:
            fig_itl.add_trace(
                go.Box(
                    y=s.itl_values,
                    name=s.backend_name,
                    marker_color=SAFE_COLORS[i % len(SAFE_COLORS)],
                )
            )
    fig_itl.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_itl, use_container_width=True)

    # Total latency box plot
    st.subheader("Total Latency (ms)")
    fig_total = go.Figure()
    for i, s in enumerate(stats):
        if s.total_latency_values:
            fig_total.add_trace(
                go.Box(
                    y=s.total_latency_values,
                    name=s.backend_name,
                    marker_color=SAFE_COLORS[i % len(SAFE_COLORS)],
                )
            )
    fig_total.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_total, use_container_width=True)


def page_output_divergence(db_path: str) -> None:
    """Output divergence page: verdict table with side-by-side completions."""
    st.header("Output Divergence")

    datasets = get_available_datasets(db_path)
    if not datasets:
        st.info("No datasets found in this database.")
        return

    dataset = st.sidebar.selectbox("Dataset", datasets)
    report = compute_output_divergence(db_path, dataset)

    if not report.prompt_verdicts:
        st.info(f"No output divergence data for dataset '{dataset}'.")
        return

    # Sort by divergence severity
    sorted_verdicts = sorted(
        report.prompt_verdicts,
        key=lambda v: VERDICT_ORDER.get(v.verdict, 99),
    )

    # Summary
    n_backends = len(report.backends)
    st.markdown(f"**{report.n_prompts} prompts** across {n_backends} backends")

    # Table
    table_data = [
        {
            "Item ID": v.item_id,
            "Prompt": v.prompt[:80] + ("..." if len(v.prompt) > 80 else ""),
            "Verdict": v.verdict,
        }
        for v in sorted_verdicts
    ]
    st.dataframe(table_data, use_container_width=True, height=400)

    # Detail view
    st.subheader("Completion Details")
    item_ids = [v.item_id for v in sorted_verdicts]
    selected_id = st.selectbox("Select prompt to inspect", item_ids)

    if selected_id:
        selected = next((v for v in sorted_verdicts if v.item_id == selected_id), None)
        if selected:
            st.markdown(f"**Prompt:** {selected.prompt}")
            st.markdown(f"**Verdict:** {selected.verdict}")
            cols = st.columns(len(selected.completions))
            for i, (backend, completion) in enumerate(
                sorted(selected.completions.items())
            ):
                with cols[i]:
                    st.markdown(f"**{backend}**")
                    st.code(completion, language=None)


def page_logprob_divergence(db_path: str) -> None:
    """Logprob divergence page: top-50 table and per-token heatmap."""
    st.header("Logprob Divergence")

    report = compute_logprob_divergence(db_path)

    if not report.divergences:
        st.info("No logprob divergence data available.")
        return

    st.markdown(
        f"**{report.n_items_analyzed} items analyzed** | "
        f"**{report.n_items_with_alerts} alerts** | "
        f"**{report.n_tokenization_mismatches} tokenization mismatches**"
    )

    # Top-50 table
    st.subheader("Top Divergent Prompts")
    table_data = [
        {
            "Item ID": d.item_id,
            "Backend A": d.backend_a,
            "Backend B": d.backend_b,
            "Tokens": d.n_tokens,
            "Mean KL": f"{d.mean_kl_contribution:.4f}",
            "Max KL": f"{d.max_kl_contribution:.4f}",
            "Alert": "Yes" if d.is_alert else "",
        }
        for d in report.divergences
    ]
    st.dataframe(table_data, use_container_width=True, height=400)

    # Per-token heatmap
    st.subheader("Per-Token KL Heatmap")

    if not report.divergences:
        return

    items_for_viz = [
        f"{d.item_id} ({d.backend_a} vs {d.backend_b})" for d in report.divergences[:20]
    ]
    selected_idx = st.selectbox(
        "Select item/pair",
        range(len(items_for_viz)),
        format_func=lambda i: items_for_viz[i],
    )

    if selected_idx is not None:
        d = report.divergences[selected_idx]
        token_deltas = visualize_token_divergence(
            db_path, d.item_id, d.backend_a, d.backend_b
        )

        if token_deltas:
            kl_values = [td.kl_contribution for td in token_deltas]
            positions = list(range(len(token_deltas)))

            fig = go.Figure(
                data=go.Heatmap(
                    z=[kl_values],
                    x=positions,
                    colorscale="Viridis",
                    colorbar={"title": "KL"},
                )
            )
            fig.update_layout(
                xaxis_title="Token Position",
                yaxis={"visible": False},
                height=200,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Token detail table
            detail_data = [
                {
                    "Pos": td.position,
                    "Token ID": td.token_id,
                    "LP A": f"{td.logprob_a:.4f}",
                    "LP B": f"{td.logprob_b:.4f}",
                    "Delta": f"{td.delta:.4f}",
                    "KL": f"{td.kl_contribution:.4f}",
                }
                for td in token_deltas
            ]
            st.dataframe(detail_data, use_container_width=True)
        else:
            st.warning("Cannot visualize: tokenization mismatch or missing data.")


def page_canary_breakdown(db_path: str) -> None:
    """Canary breakdown page: disagreement rate per precision dimension."""
    st.header("Canary Breakdown")

    breakdown = load_canary_breakdown(db_path)
    if not breakdown:
        st.info("No canary dataset results found.")
        return

    dimensions = [s.dimension for s in breakdown]
    rates = [s.disagreement_rate * 100 for s in breakdown]

    fig = go.Figure(
        data=go.Bar(
            x=dimensions,
            y=rates,
            marker_color=SAFE_COLORS[: len(dimensions)],
        )
    )
    fig.update_layout(
        yaxis_title="Disagreement Rate (%)",
        xaxis_title="Precision Dimension",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    table_data = [
        {
            "Dimension": s.dimension,
            "Total": s.total,
            "Disagreements": s.disagreements,
            "Rate": f"{s.disagreement_rate:.1%}",
        }
        for s in breakdown
    ]
    st.dataframe(table_data, use_container_width=True)


def page_mmlu_subjects(db_path: str) -> None:
    """MMLU subjects page: per-subject disagreement rate."""
    st.header("MMLU Subjects")

    subject_stats = load_mmlu_subject_stats(db_path)
    if not subject_stats:
        st.info("No MMLU results found.")
        return

    subjects = [s.subject for s in subject_stats]
    rates = [s.disagreement_rate * 100 for s in subject_stats]

    fig = go.Figure(
        data=go.Bar(
            x=subjects,
            y=rates,
            marker_color=SAFE_COLORS[: len(subjects)],
        )
    )
    fig.update_layout(
        yaxis_title="Disagreement Rate (%)",
        xaxis_title="Subject",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    table_data = [
        {
            "Subject": s.subject,
            "Total": s.total,
            "Disagreements": s.disagreements,
            "Rate": f"{s.disagreement_rate:.1%}",
        }
        for s in subject_stats
    ]
    st.dataframe(table_data, use_container_width=True)


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    db_path = _parse_db_arg()

    st.set_page_config(
        page_title="LLM Backend Divergence",
        layout="wide",
    )
    st.title("LLM Backend Divergence Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Output Divergence",
            "Logprob Divergence",
            "Canary Breakdown",
            "MMLU Subjects",
        ],
    )

    if page == "Overview":
        page_overview(db_path)
    elif page == "Output Divergence":
        page_output_divergence(db_path)
    elif page == "Logprob Divergence":
        page_logprob_divergence(db_path)
    elif page == "Canary Breakdown":
        page_canary_breakdown(db_path)
    elif page == "MMLU Subjects":
        page_mmlu_subjects(db_path)


def _is_streamlit_running() -> bool:
    """Check if running inside a Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


if _is_streamlit_running():
    main()
