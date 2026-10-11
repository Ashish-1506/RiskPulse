"""RiskPulse - Home & System Diagnostics Dashboard.

Entry point: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure root workspace directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import get_config
from src.dashboard.components import (
    apply_custom_theme,
    fetch_summary_cached,
    get_client,
    setup_autorefresh,
)

# Page configuration
st.set_page_config(
    page_title="RiskPulse | Real-Time Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()
setup_autorefresh(key="home_autorefresh")

# Sidebar navigation context
st.sidebar.title("🛡️ RiskPulse")
st.sidebar.markdown(
    """
    **Module Navigation**
    - 📡 [1_Live_Risk_Signals](/Live_Risk_Signals)
    - ⚖️ [2_Tactical_Index_Rebalancer](/Tactical_Index_Rebalancer)
    - ⚡ [3_Strategic_Stress_Test](/Strategic_Stress_Test)
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("S&P Global / CRISIL Hackathon 2026")


# Header Section
st.title("🛡️ RiskPulse: AI-Powered Wholesale Risk Intelligence")

st.markdown(
    """
    **RiskPulse** is an enterprise-grade NLP Risk Engine and Strategic Stress Testing platform designed for Wholesale Banking portfolios.
    By processing global financial news, regulatory updates, and market intelligence in real time, RiskPulse extracts counterparties,
    detects cross-asset risk events, scores sentiment and systemic impact, and simulates immediate balance-sheet and expected credit loss (ECL) shocks across multi-asset loan and trading books.
    """
)

# ---------------------------------------------------------------------------
# System Status Strip
# ---------------------------------------------------------------------------
client = get_client()
cfg = get_config()

# Gather live status metrics
api_online = client.is_api_online()
summary_data = fetch_summary_cached()
total_signals = summary_data.get("total_signals", 0)

# Check latest signal timestamp
latest_signals = client.get_latest(limit=1)
if latest_signals:
    last_update_ts = latest_signals[0].timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    active_backend = (
        latest_signals[0].model_version.split("+")[0]
        if latest_signals[0].model_version
        else cfg.sentiment.backend.upper()
    )
else:
    last_update_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    active_backend = cfg.sentiment.backend.upper()

api_badge = (
    '<span class="status-pill pill-green">● REST API Online</span>'
    if api_online
    else '<span class="status-pill pill-amber">● SQLite Local Fallback</span>'
)
backend_pill = f'<span class="status-pill pill-blue">Backend: {active_backend}</span>'

st.markdown(
    f"""
    <div class="status-strip">
        <div>
            <strong>System Status:</strong> {api_badge} &nbsp; {backend_pill}
        </div>
        <div>
            <span>Signals Indexed: <strong>{total_signals:,}</strong></span> &nbsp;|&nbsp;
            <span>Last Sync: <strong>{last_update_ts}</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Key Quick Metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Signals Processed",
        value=f"{total_signals:,}",
        delta="Live Queue" if api_online else "Offline DB",
    )

with col2:
    events_count = len(summary_data.get("by_event_type", {}))
    st.metric(
        label="Monitored Event Types",
        value=f"{events_count} Taxonomies",
        delta="12 Available",
    )

with col3:
    high_impact_count = len(summary_data.get("highest_impact", []))
    st.metric(
        label="Active High-Impact Signals",
        value=f"{high_impact_count}",
        delta="Severity ≥ 7.0",
        delta_color="inverse",
    )

with col4:
    st.metric(
        label="Portfolio Coverage",
        value="$5.02B Book",
        delta="351 Positions",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Navigation Guidance Cards
# ---------------------------------------------------------------------------
st.subheader("🧭 Module Directory")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown(
        """
        ### 📡 Page 1: Live Risk Signals
        **Real-time market perception & systemic news monitoring.**
        - **KPI Overview**: Ingestion velocity, aggregate sentiment, active sources.
        - **Interactive Filters**: Slice by ticker, source type, event category, or severity threshold.
        - **Taxonomy & Impact Visuals**: Event donut chart, sentiment bars per counterparty, and volume timeline.
        - **Explainability**: Full JSON inspector and 5-factor impact component breakdown.

        👉 *Access via sidebar: **1_Live_Risk_Signals***
        """
    )

with nav_col2:
    st.markdown(
        """
        ### ⚖️ Page 2: Tactical Index Rebalancer (Module A)
        **Dynamic sentiment-driven equity asset allocation.**
        - **Adaptive Weights**: Time-decayed sentiment scores across 15 S&P 100 tickers.
        - **Friction Controls**: Box constraints [1%, 15%], 10% turnover cap, no-trade bands.
        - **Visualizations**: Stacked weight area chart over time, current vs base bars, 'Why' attribution panel.
        - **Backtest Verification**: Historical gross/net equity curve vs equal-weight benchmark.

        👉 *Access via sidebar: **2_Tactical_Index_Rebalancer***
        """
    )

with nav_col3:
    st.markdown(
        """
        ### ⚡ Page 3: Strategic Stress Test (Module B)
        **Macro-financial portfolio vulnerability simulation.**
        - **Automated Triggering**: Signals with impact ≥ 7.0 trigger wholesale portfolio repricing.
        - **Cross-Asset Valuation**: Vectorised shock propagation across Loans, Bonds, Swaps, FX, and CDS.
        - **P&L & ECL Metrics**: Baseline vs stressed balance sheet, asset-class waterfalls, sector & country drawdowns.
        - **Scenario Control**: Custom manual overrides, top-10 counterparty losers, and CSV reporting.

        👉 *Access via sidebar: **3_Strategic_Stress_Test***
        """
    )


st.markdown("---")

# Footer
st.caption("RiskPulse Risk Intelligence Platform • S&P Global & CRISIL FinTech Hackathon • All rights reserved.")
