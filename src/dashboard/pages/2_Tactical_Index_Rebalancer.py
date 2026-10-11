"""RiskPulse - Tactical Index Rebalancer (Module A).

Dynamic sentiment-driven equity asset allocation across the 15-ticker universe.
Features interactive lookback/alpha/decay calibration, history replay animation,
stacked allocation area charts, per-counterparty delta analysis, explainability attribution,
and out-of-sample backtest comparison.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure root workspace directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.api.schemas import RiskSignal
from src.dashboard.components import (
    apply_custom_theme,
    fetch_signals_cached,
    load_backtest_results,
    load_or_generate_weights_history,
    plot_backtest_equity_curve,
    plot_current_vs_base_weights,
    plot_weights_stacked_area,
)
from src.modules.rebalancer.weights import IndexRebalancer

st.set_page_config(
    page_title="Tactical Rebalancer | RiskPulse",
    page_icon="⚖️",
    layout="wide",
)
apply_custom_theme()

st.title("⚖️ Tactical Index Rebalancer (Module A)")
st.caption("Sentiment-Tilted Equity Portfolio Allocation with Turn-Over Constraints and No-Look-Ahead Controls.")

# ---------------------------------------------------------------------------
# Sidebar Controls & History Replay
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Rebalancer Hyperparameters")

lookback_hours = st.sidebar.slider(
    "Lookback Window (Hours)",
    min_value=6,
    max_value=168,
    value=72,
    step=6,
    help="Time horizon of historical signals considered for sentiment scoring.",
)

alpha = st.sidebar.slider(
    "Sensitivity Factor (α)",
    min_value=0.1,
    max_value=1.5,
    value=0.5,
    step=0.05,
    help="Scales responsiveness of weight shifts to sentiment scores s_i.",
)

decay_lambda = st.sidebar.slider(
    "Decay Rate (λ per hour)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Exponential time-decay factor for older signals.",
)

max_weight = st.sidebar.slider(
    "Max Position Cap",
    min_value=0.10,
    max_value=0.25,
    value=0.15,
    step=0.01,
    help="Upper box constraint on individual stock allocation.",
)

st.sidebar.markdown("---")
st.sidebar.header("▶️ History Replay Animation")

# Load weights history
history_df = load_or_generate_weights_history()

if not history_df.empty:
    unique_timestamps = sorted(history_df["timestamp"].unique())
    total_steps = len(unique_timestamps)
else:
    unique_timestamps = []
    total_steps = 0

if "replay_playing" not in st.session_state:
    st.session_state["replay_playing"] = False
if "replay_idx" not in st.session_state:
    st.session_state["replay_idx"] = max(0, total_steps - 1)

rep_col1, rep_col2 = st.sidebar.columns(2)
with rep_col1:
    if st.button("▶ Play" if not st.session_state["replay_playing"] else "⏸ Pause", width="stretch"):
        st.session_state["replay_playing"] = not st.session_state["replay_playing"]

with rep_col2:
    if st.button("⏮ Reset", width="stretch"):
        st.session_state["replay_playing"] = False
        st.session_state["replay_idx"] = 0

replay_speed = st.sidebar.slider("Step Interval (s)", 0.2, 2.0, 0.8, 0.2)

# Non-blocking replay step advance via autorefresh
if st.session_state["replay_playing"] and total_steps > 1:
    st.session_state["replay_idx"] = (st.session_state["replay_idx"] + 1) % total_steps
    st_autorefresh(interval=int(replay_speed * 1000), key="rebalancer_replay_timer")
    st.sidebar.caption(f"Replaying step {st.session_state['replay_idx'] + 1} of {total_steps}...")

# Timeline scrubber slider
if total_steps > 0:
    scrub_idx = st.sidebar.slider(
        "Scrub Timeline",
        min_value=0,
        max_value=total_steps - 1,
        value=min(st.session_state["replay_idx"], total_steps - 1),
        key="timeline_scrubber",
    )
    if scrub_idx != st.session_state["replay_idx"] and not st.session_state["replay_playing"]:
        st.session_state["replay_idx"] = scrub_idx
    active_ts = pd.to_datetime(unique_timestamps[st.session_state["replay_idx"]])
else:
    active_ts = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Rebalancer Instance & Active Calculation
# ---------------------------------------------------------------------------
raw_signals = fetch_signals_cached(limit=500)
signals = [RiskSignal.model_validate(s) for s in raw_signals]

reb = IndexRebalancer()
reb.alpha = alpha
reb.decay_lambda = decay_lambda
reb.max_weight = max_weight

# Run rebalance for active timestamp with lookback cutoff
lookback_start = active_ts - timedelta(hours=lookback_hours)
active_signals = [s for s in signals if lookback_start <= s.timestamp <= active_ts]

rebal_res = reb.rebalance(active_signals, now=active_ts)
current_weights = rebal_res.weights
base_weights = reb.base_weights
tickers = reb.tickers

# Calculate deltas in bps
deltas_bps = {t: (current_weights[t] - base_weights[t]) * 10000 for t in tickers}
n_up = sum(1 for d in deltas_bps.values() if d > 5)
n_down = sum(1 for d in deltas_bps.values() if d < -5)
top_gainer = max(deltas_bps.items(), key=lambda x: x[1])
top_loser = min(deltas_bps.items(), key=lambda x: x[1])

# ---------------------------------------------------------------------------
# Top KPI Metric Cards
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Tilted Allocations",
        value=f"{n_up} Up / {n_down} Down",
        delta=f"{len(tickers)} Total Universe",
    )

with kpi2:
    st.metric(
        label="Portfolio Turnover",
        value=f"{rebal_res.turnover:.2%}",
        delta=f"Cap: {reb.turnover_cap:.0%}",
    )

with kpi3:
    st.metric(
        label="Top Gainer (Overweight)",
        value=f"{top_gainer[0]} (+{top_gainer[1]:.0f} bp)",
        delta=f"{current_weights[top_gainer[0]]:.2%}",
    )

with kpi4:
    st.metric(
        label="Top Loser (Underweight)",
        value=f"{top_loser[0]} ({top_loser[1]:.0f} bp)",
        delta=f"{current_weights[top_loser[0]]:.2%}",
        delta_color="inverse",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Main Stacked Area Chart (Weights History)
# ---------------------------------------------------------------------------
st.subheader("📈 Multi-Asset Tactical Weight Dynamics")
st.caption(
    f"Stacked allocation trajectory over time. Currently viewing snapshot at: **{active_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}**"
)

# Vertical spacing to prevent heading text overlap
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
st.plotly_chart(plot_weights_stacked_area(history_df), width="stretch")

# Vertical spacing before current vs base weights chart
st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
st.plotly_chart(plot_current_vs_base_weights(current_weights, base_weights), width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-Ticker Allocations Table & Attribution Panel
# ---------------------------------------------------------------------------
st.subheader("📋 Ticker Breakdown & Signal Drivers")
tab_table, tab_explain = st.tabs(["📊 Allocation Matrix", "🔬 'Why' Attribution Panel"])

with tab_table:
    table_rows = []
    for t in tickers:
        w_cur = current_weights[t]
        w_base = base_weights[t]
        d_bp = deltas_bps[t]
        s_score = rebal_res.scores.get(t, 0.0)
        expls = reb.explain(t, top_n=1)
        top_headline = expls[0]["headline"] if expls else "-"

        table_rows.append(
            {
                "Ticker": t,
                "Sentiment Score (s_i)": f"{s_score:+.3f}",
                "Base Weight": f"{w_base:.2%}",
                "Tactical Weight": f"{w_cur:.2%}",
                "Shift (bps)": f"{d_bp:+.0f} bp",
                "Direction": "▲ Overweight" if d_bp > 5 else ("▼ Underweight" if d_bp < -5 else "— Par"),
                "Primary Driver Headline": (top_headline[:80] + "...") if len(top_headline) > 80 else top_headline,
            }
        )

    st.dataframe(
        pd.DataFrame(table_rows),
        width="stretch",
        hide_index=True,
    )

with tab_explain:
    selected_ticker = st.selectbox(
        "Select Ticker for Root-Cause Analysis:",
        options=tickers,
        index=tickers.index(top_gainer[0]) if top_gainer[0] in tickers else 0,
    )
    explanations = reb.explain(selected_ticker, top_n=3)

    if explanations:
        st.write(f"Top signals driving sentiment and weight tilt for **{selected_ticker}**:")
        for idx, item in enumerate(explanations, 1):
            with st.container():
                st.markdown(
                    f"""
                    **{idx}. {item.get("headline", "Headline")}**
                    *Time:* `{item.get("timestamp", "N/A")}` &nbsp;|&nbsp;
                    *Source:* `{item.get("source_type", "news")}` &nbsp;|&nbsp;
                    *Sentiment:* `{item.get("sentiment_score", 0):+.2f}` &nbsp;|&nbsp;
                    *Impact:* `{item.get("impact_score", 0):.1f}/10` &nbsp;|&nbsp;
                    *Net Contribution:* **`{item.get("contribution", 0):+.4f}`**
                    """
                )
                st.markdown("---")
    else:
        st.info(
            f"No specific NLP news or social signals recorded for {selected_ticker} in the active window. Stock is weighted at base benchmark."
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Backtest Performance Section
# ---------------------------------------------------------------------------
st.subheader("🏁 Out-of-Sample Backtest Results")
st.caption("Strict point-in-time daily rebalancing simulation against equal-weight benchmark index.")

curve_df, metrics_dict = load_backtest_results()

if not curve_df.empty:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([2, 1])

    with b_col1:
        st.plotly_chart(plot_backtest_equity_curve(curve_df), width="stretch")

    with b_col2:
        st.markdown("#### Performance Metrics")
        strat_m = metrics_dict.get("strategy", {})
        net_m = metrics_dict.get("strategy_net_of_cost", {})
        bench_m = metrics_dict.get("benchmark_equal_weight", {})

        metrics_summary = [
            {
                "Metric": "Cumulative Return",
                "Tactical (Gross)": f"{strat_m.get('cumulative_return', 0):+.2%}",
                "Tactical (Net)": f"{net_m.get('cumulative_return', 0):+.2%}",
                "Benchmark": f"{bench_m.get('cumulative_return', 0):+.2%}",
            },
            {
                "Metric": "Annualized Volatility",
                "Tactical (Gross)": f"{strat_m.get('annualized_volatility', 0):.2%}",
                "Tactical (Net)": f"{net_m.get('annualized_volatility', 0):.2%}",
                "Benchmark": f"{bench_m.get('annualized_volatility', 0):.2%}",
            },
            {
                "Metric": "Sharpe Ratio (Rf=0)",
                "Tactical (Gross)": f"{strat_m.get('sharpe_ratio', 0):.2f}",
                "Tactical (Net)": f"{net_m.get('sharpe_ratio', 0):.2f}",
                "Benchmark": f"{bench_m.get('sharpe_ratio', 0):.2f}",
            },
            {
                "Metric": "Max Drawdown",
                "Tactical (Gross)": f"{strat_m.get('max_drawdown', 0):.2%}",
                "Tactical (Net)": f"{net_m.get('max_drawdown', 0):.2%}",
                "Benchmark": f"{bench_m.get('max_drawdown', 0):.2%}",
            },
            {
                "Metric": "Average Turnover",
                "Tactical (Gross)": f"{strat_m.get('average_turnover', 0):.2%}",
                "Tactical (Net)": f"{net_m.get('average_turnover', 0):.2%}",
                "Benchmark": "0.00%",
            },
        ]
        st.table(pd.DataFrame(metrics_summary))

    if "limitations_note" in metrics_dict:
        st.info(f"ℹ️ **Model Limitations:** {metrics_dict['limitations_note']}")
else:
    st.warning("Backtest results not yet found. Run `python -m src.modules.rebalancer.backtest` to generate.")
