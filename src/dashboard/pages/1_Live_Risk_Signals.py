"""RiskPulse - Live Risk Signals Monitoring Page.

Interactive real-time signal stream, KPI metrics, NLP taxonomy breakdown,
cross-counterparty sentiment, and impact explainability.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure root workspace directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

import pandas as pd
import streamlit as st

from src.dashboard.components import (
    apply_custom_theme,
    fetch_signals_cached,
    fetch_summary_cached,
    get_client,
    plot_event_donut,
    plot_impact_histogram,
    plot_sentiment_by_ticker,
    plot_volume_timeline,
    render_impact_badge,
    render_sentiment_badge,
    setup_autorefresh,
    signals_to_dataframe,
)

st.set_page_config(page_title="Live Risk Signals | RiskPulse", page_icon="📡", layout="wide")
apply_custom_theme()
setup_autorefresh(key="signals_autorefresh")

client = get_client()
is_offline = not client.is_api_online() or os.getenv("OFFLINE_MODE") == "1"

st.title("📡 Live Risk Signals Intelligence")
if is_offline:
    st.markdown(
        '<span class="status-pill pill-amber">⚡ Offline / Replay Mode</span> &nbsp;<span style="font-size:0.82rem; color:#94A3B8;">(Operating on local verified signals storage)</span>',
        unsafe_allow_html=True,
    )
st.caption("Real-time NLP stream scoring financial sentiment, systemic event taxonomies, and multi-factor impact.")

# ---------------------------------------------------------------------------
# Interactive Injection & Demo Scenarios Panel
# ---------------------------------------------------------------------------
if "headline_input_widget" not in st.session_state:
    st.session_state["headline_input_widget"] = ""
if "inject_source" not in st.session_state:
    st.session_state["inject_source"] = "reuters"
if "inject_channel" not in st.session_state:
    st.session_state["inject_channel"] = "news"
if "latest_injected_signal" not in st.session_state:
    st.session_state["latest_injected_signal"] = None


def load_demo_scenario(text: str, source: str, channel: str = "news") -> None:
    st.session_state["headline_input_widget"] = text
    st.session_state["inject_source"] = source
    st.session_state["inject_channel"] = channel
    st.session_state["latest_injected_signal"] = None


def reset_injected_demo() -> None:
    client.delete_injected()
    st.session_state["headline_input_widget"] = ""
    st.session_state["latest_injected_signal"] = None
    if "target_stress_signal" in st.session_state:
        del st.session_state["target_stress_signal"]


expander_open = bool(st.session_state.get("headline_input_widget") or st.session_state.get("latest_injected_signal"))

with st.expander("⚡ Inject Risk Headline & Deterministic Demo Scenarios", expanded=expander_open):
    st.markdown("**Test real-time pipeline inference or trigger live wholesale stress testing:**")

    # 3 One-Click Deterministic Demo Scenarios
    scen_c1, scen_c2, scen_c3 = st.columns(3)

    with scen_c1:
        st.button(
            "🌐 1. Geopolitical Shock",
            on_click=load_demo_scenario,
            args=(
                "Naval clash closes key shipping lane; oil jumps and sanctions threatened against exporters",
                "reuters",
                "news",
            ),
            width="stretch",
            help="Naval clash & sanctions trigger",
        )

    with scen_c2:
        st.button(
            "💳 2. Credit Event Default",
            on_click=load_demo_scenario,
            args=(
                "Regional lender collapses after bond default; regulators warn of contagion across banks",
                "bloomberg",
                "news",
            ),
            width="stretch",
            help="Regional bank collapse & default",
        )

    with scen_c3:
        st.button(
            "🏦 3. Monetary Rate Shock",
            on_click=load_demo_scenario,
            args=(
                "Central bank delivers emergency rate hike as inflation surges past forecasts",
                "reuters",
                "news",
            ),
            width="stretch",
            help="Central bank emergency rate hike",
        )

    input_text = st.text_area(
        "Headline or Financial News Flash",
        key="headline_input_widget",
        placeholder="e.g. Sovereign debt rating downgraded to junk; credit default spreads widen 150 bps",
        height=70,
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        chan_idx = 0 if st.session_state.get("inject_channel", "news") == "news" else 1
        source_mode = st.selectbox("Channel", ["news", "social"], index=chan_idx)
    with ctrl_col2:
        run_inject = st.button("🚀 Run Analysis", type="primary", width="stretch")
    with ctrl_col3:
        if st.button("🧹 Reset Demo (Clear Injected)", width="stretch", on_click=reset_injected_demo):
            st.cache_data.clear()
            st.success("Injected demo signals removed successfully.")
            st.rerun()

    # Process Injection ONLY on 'Run Analysis' button click
    if run_inject:
        headline_to_run = input_text.strip()
        if not headline_to_run:
            st.warning("⚠️ Please select one of the demo scenarios above or enter a headline to analyze.")
        else:
            with st.spinner("Executing NLP pipeline inference..."):
                sig = client.analyze(
                    text=headline_to_run,
                    source_type=source_mode,
                    source=st.session_state.get("inject_source", "reuters"),
                    title=headline_to_run,
                )
            if sig:
                st.cache_data.clear()
                st.session_state["target_stress_signal"] = sig.signal_id
                st.session_state["latest_injected_signal"] = sig
            else:
                st.error("Failed to analyze headline. Ensure API or offline mode is running.")

    # Render results from current or active session
    active_sig = st.session_state.get("latest_injected_signal")
    if active_sig:
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.markdown(
                f"**Sentiment:** {render_sentiment_badge(active_sig.sentiment_label)} `{active_sig.sentiment_score:+.2f}`",
                unsafe_allow_html=True,
            )
        with res_c2:
            st.markdown(f"**Event Category:** `{active_sig.event_type}` ({active_sig.event_confidence:.0%})")
        with res_c3:
            st.markdown(f"**Impact:** {render_impact_badge(active_sig.impact_score)}", unsafe_allow_html=True)
        with res_c4:
            st.markdown(f"**Source:** `{active_sig.source}` ({active_sig.source_type})")

        st.info(f"💡 **Impact Rationale:** {active_sig.impact_rationale}")

        if active_sig.impact_score >= 7.0:
            st.markdown(
                f"""
                <div class="stress-banner-alert">
                    🚨 <strong>HIGH-IMPACT SCENARIO TRIGGERED ({active_sig.impact_score:.1f}/10):</strong>
                    This event meets wholesale stress testing criteria.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.page_link(
                "pages/3_Strategic_Stress_Test.py",
                label="⚡ Open Strategic Stress Test with this Scenario (Module B)",
                icon="⚡",
            )

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("🔍 Signal Filters")

summary = fetch_summary_cached()
available_events = ["All"] + sorted(list(summary.get("by_event_type", {}).keys()))
available_tickers = ["All"] + sorted(list(summary.get("avg_sentiment_per_ticker", {}).keys()))

selected_ticker = st.sidebar.selectbox("Filter by Ticker", options=available_tickers)
selected_event = st.sidebar.selectbox("Filter by Event Type", options=available_events)
selected_source = st.sidebar.selectbox("Source Type", options=["All", "news", "social"])
min_impact = st.sidebar.slider("Minimum Impact Score", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

q_ticker = None if selected_ticker == "All" else selected_ticker
q_event = None if selected_event == "All" else selected_event
q_source = None if selected_source == "All" else selected_source

raw_signals = fetch_signals_cached(
    ticker=q_ticker,
    event_type=q_event,
    min_impact=min_impact,
    source_type=q_source,
    limit=250,
)


df = signals_to_dataframe(raw_signals)

# ---------------------------------------------------------------------------
# KPI Cards Row
# ---------------------------------------------------------------------------
if df.empty:
    st.info("ℹ️ No risk signals match the selected filter criteria. Try adjusting the filters in the sidebar.")
    st.stop()

# Signals today calculation
now = datetime.now(timezone.utc)
today_start = pd.Timestamp(now.date(), tz="UTC")
if "timestamp" in df.columns:
    today_signals = df[pd.to_datetime(df["timestamp"], utc=True) >= today_start]
else:
    today_signals = df
today_count = len(today_signals) if len(today_signals) > 0 else len(df)

avg_sentiment = float(df["sentiment_score"].mean()) if "sentiment_score" in df.columns else 0.0
high_impact_count = int((df["impact_score"] >= 7.0).sum()) if "impact_score" in df.columns else 0
unique_sources = int(df["source"].nunique()) if "source" in df.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Signals in View", f"{len(df):,}", delta=f"{today_count} Today")
with kpi2:
    sent_label = "Positive" if avg_sentiment > 0.05 else ("Negative" if avg_sentiment < -0.05 else "Neutral")
    st.metric("Average Sentiment", f"{avg_sentiment:+.2f}", delta=sent_label)
with kpi3:
    st.metric("High-Impact (≥ 7.0)", f"{high_impact_count}", delta="Requires Stress Test", delta_color="inverse")
with kpi4:
    st.metric("Active Sources", f"{unique_sources} Feeds", delta="Multi-Domain")

st.markdown("---")

# ---------------------------------------------------------------------------
# Visual Analytics (2x2 Grid)
# ---------------------------------------------------------------------------
st.subheader("📊 Visual Risk Analytics")
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(plot_event_donut(df), width="stretch")
with row1_col2:
    st.plotly_chart(plot_impact_histogram(df), width="stretch")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(plot_sentiment_by_ticker(df), width="stretch")
with row2_col2:
    st.plotly_chart(plot_volume_timeline(df), width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Latest Signals Badge Table
# ---------------------------------------------------------------------------
st.subheader("📋 Latest Ingested Signals")
st.caption("Displaying top 30 filtered signals with real-time severity and sentiment badges.")

display_rows = df.head(30).copy()

table_rows = []
for _, row in display_rows.iterrows():
    ts_str = row["timestamp"].strftime("%m-%d %H:%M") if pd.notnull(row.get("timestamp")) else "-"
    impact_html = render_impact_badge(row.get("impact_score", 0.0))
    sent_html = render_sentiment_badge(row.get("sentiment_label", "neutral"))
    headline = row.get("headline", "-")
    headline_clean = (headline[:85] + "...") if len(headline) > 85 else headline
    tickers_list = row.get("tickers", [])
    tickers_str = ", ".join(tickers_list) if isinstance(tickers_list, list) and tickers_list else "-"
    event_str = row.get("event_type", "Other")
    conf_val = float(row.get("event_confidence", 0.0))
    conf_str = f"{conf_val:.0%}"
    source_str = str(row.get("source", "-"))

    table_rows.append(
        f'<tr style="border-bottom: 1px solid #334155; color: #F8FAFC;">'
        f'<td style="padding: 8px 10px; white-space: nowrap;">{ts_str}</td>'
        f'<td style="padding: 8px 10px; white-space: nowrap;">{impact_html}</td>'
        f'<td style="padding: 8px 10px; white-space: nowrap;">{sent_html}</td>'
        f'<td style="padding: 8px 10px; font-weight: 500;">{headline_clean}</td>'
        f'<td style="padding: 8px 10px;"><code style="color: #60A5FA;">{tickers_str}</code></td>'
        f'<td style="padding: 8px 10px;">{event_str}</td>'
        f'<td style="padding: 8px 10px;">{conf_str}</td>'
        f'<td style="padding: 8px 10px; color: #94A3B8;">{source_str}</td>'
        f"</tr>"
    )

table_html = (
    '<div style="overflow-x:auto;">'
    '<table style="width:100%; border-collapse: collapse; font-size: 0.84rem; text-align: left; background-color: #1E293B; border-radius: 8px;">'
    "<thead>"
    '<tr style="border-bottom: 2px solid #334155; color: #94A3B8;">'
    '<th style="padding: 10px;">Time (UTC)</th>'
    '<th style="padding: 10px;">Impact</th>'
    '<th style="padding: 10px;">Sentiment</th>'
    '<th style="padding: 10px;">Headline</th>'
    '<th style="padding: 10px;">Tickers</th>'
    '<th style="padding: 10px;">Event Category</th>'
    '<th style="padding: 10px;">Confidence</th>'
    '<th style="padding: 10px;">Source</th>'
    "</tr>"
    "</thead>"
    f"<tbody>{''.join(table_rows)}</tbody>"
    "</table></div>"
)

if hasattr(st, "html"):
    st.html(table_html)
else:
    st.markdown(table_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Signal Details Expander (Inspection & Explainability)
# ---------------------------------------------------------------------------
st.markdown("###")
with st.expander("🔬 Deep-Dive Signal Details & Explainability", expanded=False):
    st.write("Inspect the exact impact components, reasoning rationale, and raw JSON schema for any signal.")

    # Selection selector
    options_dict = {
        f"[{r.get('signal_id', 'unknown')[:8]}] (Impact: {r.get('impact_score', 0)}) - {r.get('headline', '')[:70]}": r
        for r in raw_signals[:50]
    }

    if options_dict:
        selected_key = st.selectbox("Select Signal to Inspect", options=list(options_dict.keys()))
        inspected = options_dict[selected_key]

        detail_col1, detail_col2 = st.columns([1, 1])

        with detail_col1:
            st.markdown("#### Impact Component Breakdown")
            comp = inspected.get("impact_components", {})
            comp_df = pd.DataFrame(
                [
                    {"Component": "Event Prior Severity", "Score": f"{comp.get('event_prior', 0.0):.2f}"},
                    {"Component": "Sentiment Intensity", "Score": f"{comp.get('sentiment_intensity', 0.0):.2f}"},
                    {"Component": "Keyword Intensity", "Score": f"{comp.get('keyword_intensity', 0.0):.2f}"},
                    {"Component": "Source Credibility", "Score": f"{comp.get('source_credibility', 0.0):.2f}"},
                    {"Component": "Corroboration Factor", "Score": f"{comp.get('corroboration', 0.0):.2f}"},
                    {"Component": "Final Impact Score", "Score": f"⭐ {inspected.get('impact_score', 0.0):.1f} / 10"},
                ]
            )
            st.table(comp_df)
            st.info(f"**Impact Rationale:** {inspected.get('impact_rationale', 'N/A')}")

        with detail_col2:
            st.markdown("#### Raw Signal JSON Payload")
            st.json(inspected)
