"""RiskPulse - Strategic Stress Testing (Module B).

Wholesale banking multi-asset stress simulation across Loans, Bonds,
Derivatives (IRS, CDS, FX), and Equities, triggered by high-severity NLP signals.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure root workspace directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.api.schemas import RiskSignal
from src.dashboard.components import (
    apply_custom_theme,
    fetch_portfolio_cached,
    fetch_signals_cached,
    format_usd,
    plot_pnl_waterfall,
    plot_sector_and_country_bars,
    plot_valuation_comparison,
    setup_autorefresh,
)
from src.modules.stress.engine import StressEngine
from src.modules.stress.scenarios import ShockPackage, load_scenarios

st.set_page_config(page_title="Strategic Stress Test | RiskPulse", page_icon="⚡", layout="wide")
apply_custom_theme()
setup_autorefresh(key="stress_autorefresh")

st.title("⚡ Strategic Stress Testing (Module B)")
st.caption("Wholesale Banking Portfolio Valuation & Macro-Financial Shock Engine.")

# ---------------------------------------------------------------------------
# Load Wholesale Portfolio
# ---------------------------------------------------------------------------
portfolio_df = fetch_portfolio_cached()
if portfolio_df is None or portfolio_df.empty:
    st.error("⚠️ Wholesale banking portfolio could not be loaded from data/portfolio.csv.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Stress Test Parameters")

trigger_threshold = st.sidebar.slider(
    "Trigger Threshold",
    min_value=5.0,
    max_value=9.5,
    value=7.0,
    step=0.5,
    help="Signals with impact score >= threshold will trigger wholesale portfolio stress repricing.",
)

mode = st.sidebar.radio(
    "Simulation Mode",
    options=["Auto (from live signals)", "Manual scenario"],
    help="Auto triggers from live NLP signals. Manual allows bespoke parameter stress testing.",
)

configured_scenarios = load_scenarios()
stress_engine = StressEngine()

target_signal = None
custom_shock_pkg = None
is_triggered = False

# ---------------------------------------------------------------------------
# Auto Mode Execution
# ---------------------------------------------------------------------------
if mode == "Auto (from live signals)":
    # Fetch all signals from DB/API
    all_signals_data = fetch_signals_cached(min_impact=trigger_threshold, limit=100)

    # Filter only signals matching configured stress event types
    triggered_signals = [
        RiskSignal.model_validate(s) for s in all_signals_data if s.get("event_type") in configured_scenarios
    ]

    if triggered_signals:
        is_triggered = True
        # Sort by impact score descending
        triggered_signals.sort(key=lambda s: s.impact_score, reverse=True)

        signal_options = {f"[{s.impact_score:.1f}] {s.event_type} - {s.headline[:60]}...": s for s in triggered_signals}

        # Check if session state holds a target signal from the injection panel
        default_idx = 0
        target_sig_id = st.session_state.get("target_stress_signal")
        if target_sig_id:
            for idx, s in enumerate(triggered_signals):
                if s.signal_id == target_sig_id:
                    default_idx = idx
                    break

        selected_label = st.sidebar.selectbox(
            f"Select Triggered Signal ({len(triggered_signals)} active)",
            options=list(signal_options.keys()),
            index=default_idx,
        )
        target_signal = signal_options[selected_label]

    else:
        is_triggered = False

# ---------------------------------------------------------------------------
# Manual Mode Execution
# ---------------------------------------------------------------------------
else:
    is_triggered = True
    event_type = st.sidebar.selectbox("Scenario Event Type", options=list(configured_scenarios.keys()))
    manual_impact = st.sidebar.slider("Impact Score (1-10)", min_value=1.0, max_value=10.0, value=8.0, step=0.5)

    use_custom_shocks = st.sidebar.checkbox("Custom Shock Sliders", value=False)
    base_scen = configured_scenarios[event_type]
    severity = manual_impact / 10.0

    if use_custom_shocks:
        eq_shock = (
            st.sidebar.slider("Equity Shock (%)", -50.0, 10.0, float(base_scen.equity * 100 * severity), 1.0) / 100.0
        )
        rates_bps = st.sidebar.slider("Rates Shift (bps)", -200.0, 400.0, float(base_scen.rates_bps * severity), 10.0)
        spread_bps = st.sidebar.slider(
            "Credit Spread Widening (bps)", 0.0, 500.0, float(base_scen.spread_bps * severity), 10.0
        )
        fx_em = st.sidebar.slider("EM FX Shock (%)", -30.0, 5.0, float(base_scen.fx_em * 100 * severity), 1.0) / 100.0
        pd_mult = st.sidebar.slider(
            "PD Multiplier", 1.0, 4.0, float(1.0 + (base_scen.pd_multiplier - 1.0) * severity), 0.1
        )

        custom_shock_pkg = ShockPackage(
            scenario_name=f"Manual Custom: {event_type}",
            event_type=event_type,
            severity=severity,
            equity_shock=eq_shock,
            rates_bps=rates_bps,
            spread_bps=spread_bps,
            fx_em_shock=fx_em,
            fx_usd_shock=base_scen.fx_usd * severity,
            commodity_shock=base_scen.commodity * severity,
            pd_multiplier=pd_mult,
            affected_sectors=[],
            affected_countries=[],
        )
    else:
        now_utc = datetime.now(timezone.utc)
        target_signal = RiskSignal(
            signal_id=f"manual_{now_utc.strftime('%H%M%S')}",
            timestamp=now_utc,
            source="manual_terminal",
            source_type="manual",
            headline=f"Manual Simulation: {event_type} Severe Event",
            tickers=[],
            countries=[],
            sectors=[],
            sentiment_score=-0.75,
            sentiment_label="negative",
            event_type=event_type,
            event_confidence=1.0,
            impact_score=manual_impact,
        )

# ---------------------------------------------------------------------------
# Main Display & Banners
# ---------------------------------------------------------------------------
if not is_triggered:
    neutral_banner = (
        f'<div class="stress-banner-neutral">'
        f"✅ <strong>Wholesale Portfolio Resilient:</strong> No incoming market signals exceed the trigger threshold (Impact ≥ {trigger_threshold:.1f}). "
        f"Portfolio value: {format_usd(portfolio_df['market_value'].sum())} across {len(portfolio_df)} active positions."
        f"</div>"
    )
    if hasattr(st, "html"):
        st.html(neutral_banner)
    else:
        st.markdown(neutral_banner, unsafe_allow_html=True)
    st.info(
        "💡 You can lower the Trigger Threshold in the sidebar or switch to 'Manual scenario' mode to simulate extreme market events."
    )
    st.stop()

# Run the Stress Test
with st.spinner("Valuing wholesale banking book under dynamic macro shocks..."):
    stress_input = custom_shock_pkg if custom_shock_pkg is not None else target_signal
    result = stress_engine.run(portfolio=portfolio_df, signal_or_scenario=stress_input)

if result is None:
    st.warning("⚠️ Selected scenario could not be mapped to wholesale valuation parameters.")
    st.stop()

# Red Trigger Banner
banner_event = result.scenario_params.get("event_type", "High-Impact Risk")
banner_headline = result.headline or "Manual Macro Shock Scenario"
alert_banner = (
    f'<div class="stress-banner-alert">'
    f"🚨 <strong>STRESS TEST TRIGGERED:</strong> Event Type: <u>{banner_event}</u> &nbsp;|&nbsp; Severity: {result.scenario_params.get('severity', 1.0) * 10:.1f}/10<br>"
    f'<span style="font-size: 0.9rem; font-weight: 400; opacity: 0.95;">Headline: {banner_headline}</span>'
    f"</div>"
)
if hasattr(st, "html"):
    st.html(alert_banner)
else:
    st.markdown(alert_banner, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Primary KPI Metric Cards
# ---------------------------------------------------------------------------
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.metric("Base Portfolio Value", format_usd(result.base_value))

with kpi_c2:
    st.metric("Stressed Value", format_usd(result.stressed_value))

with kpi_c3:
    st.metric(
        "Net Portfolio Loss",
        format_usd(result.total_pnl),
        delta=f"{result.pnl_pct:+.2f}%",
        delta_color="inverse",
    )

with kpi_c4:
    st.metric(
        "Expected Credit Loss (Δ ECL)",
        format_usd(result.total_d_ecl),
        delta=f"+{format_usd(result.total_d_ecl)}",
        delta_color="inverse",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Visual Charts: Valuation, Waterfall, Sector & Country
# ---------------------------------------------------------------------------
row1_left, row1_right = st.columns([1, 1])

with row1_left:
    st.plotly_chart(
        plot_valuation_comparison(result.base_value, result.stressed_value),
        width="stretch",
    )

with row1_right:
    st.plotly_chart(
        plot_pnl_waterfall(result.breakdowns.get("asset_class", [])),
        width="stretch",
    )

st.plotly_chart(
    plot_sector_and_country_bars(
        result.breakdowns.get("sector", []),
        result.breakdowns.get("country", []),
    ),
    width="stretch",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top 10 Losing Positions Table
# ---------------------------------------------------------------------------
st.subheader("📉 Top 10 Most Vulnerable Positions")
top_losers_df = pd.DataFrame(result.top_10_losers)

if not top_losers_df.empty:
    display_losers = top_losers_df.copy()
    display_losers["base_value"] = display_losers["base_value"].apply(format_usd)
    display_losers["stressed_value"] = display_losers["stressed_value"].apply(format_usd)
    display_losers["pnl"] = display_losers["pnl"].apply(format_usd)
    display_losers["d_ecl"] = display_losers["d_ecl"].apply(format_usd)

    st.dataframe(
        display_losers,
        column_config={
            "position_id": "Position ID",
            "counterparty": "Counterparty",
            "asset_class": "Asset Class",
            "sector": "Sector",
            "country": "Country",
            "base_value": "Base Value",
            "stressed_value": "Stressed Value",
            "pnl": "P&L",
            "d_ecl": "Δ ECL",
        },
        width="stretch",
        hide_index=True,
    )

    # CSV Download Button
    csv_bytes = top_losers_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Top Losers CSV",
        data=csv_bytes,
        file_name=f"stress_top_losers_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Expanders: Scenario Parameters & Triggering Signal JSON
# ---------------------------------------------------------------------------
st.markdown("###")
exp_c1, exp_c2 = st.columns(2)

with exp_c1:
    with st.expander("⚙️ Scenario Parameters Used", expanded=False):
        st.json(result.scenario_params)

with exp_c2:
    with st.expander("📰 Triggering Headline & Signal Schema", expanded=False):
        if target_signal is not None:
            st.markdown(f"**Headline:** {target_signal.headline}")
            st.json(target_signal.model_dump())
        else:
            st.write("Manual scenario parameters were used directly.")
