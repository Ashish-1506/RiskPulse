"""Reusable UI components, Plotly chart builders, and caching helpers for RiskPulse dashboard."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure root workspace directory is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.api.client import SignalClient
from src.api.schemas import RiskSignal
from src.config import BASE_DIR
from src.modules.stress.portfolio import load_portfolio

# Consistent Dark Palette
DARK_BG = "#0F172A"
DARK_CARD = "#1E293B"
TEXT_COLOR = "#F8FAFC"
ACCENT_BLUE = "#3B82F6"
GREEN_COLOR = "#10B981"
RED_COLOR = "#EF4444"
AMBER_COLOR = "#F59E0B"
SLATE_MUTED = "#64748B"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def apply_custom_theme() -> None:
    """Injects high-end modern CSS styling for metric cards, badges, and headers."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }
        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            padding: 14px 18px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        div[data-testid="stMetric"] label {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            font-weight: 500;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 700;
        }
        /* Custom status strip */
        .status-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 18px;
            margin-bottom: 20px;
            font-size: 0.88rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .pill-green { background-color: #064E3B; color: #34D399; border: 1px solid #059669; }
        .pill-amber { background-color: #78350F; color: #FBBF24; border: 1px solid #D97706; }
        .pill-red { background-color: #7F1D1D; color: #F87171; border: 1px solid #DC2626; }
        .pill-blue { background-color: #1E3A8A; color: #60A5FA; border: 1px solid #2563EB; }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-align: center;
        }
        .badge-positive { background-color: #064E3B; color: #6EE7B7; }
        .badge-negative { background-color: #7F1D1D; color: #FCA5A5; }
        .badge-neutral { background-color: #334155; color: #CBD5E1; }
        .badge-high { background-color: #991B1B; color: #FEE2E2; }
        .badge-med { background-color: #92400E; color: #FEF3C7; }
        .badge-low { background-color: #1E3A8A; color: #DBEAFE; }

        /* Banner alerts */
        .stress-banner-alert {
            background-color: #450A0A;
            border: 1px solid #DC2626;
            color: #FEE2E2;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 600;
        }
        .stress-banner-neutral {
            background-color: #064E3B;
            border: 1px solid #059669;
            color: #D1FAE5;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_autorefresh(key: str = "auto_refresh_widget") -> None:
    """Sidebar toggle and interval slider for live dashboard auto-refresh."""
    st.sidebar.markdown("### 🔄 Live Sync")
    auto_refresh = st.sidebar.toggle("Auto-Refresh", value=False, key=f"{key}_toggle")
    if auto_refresh:
        interval_secs = st.sidebar.slider(
            "Refresh Interval (s)",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
            key=f"{key}_slider",
        )
        st_autorefresh(interval=interval_secs * 1000, key=f"{key}_timer")
        st.sidebar.caption(f"⚡ Syncing every {interval_secs}s")


# ---------------------------------------------------------------------------
# Cached Data Fetchers
# ---------------------------------------------------------------------------


@st.cache_resource
def get_client() -> SignalClient:
    """Returns a singleton instance of SignalClient."""
    return SignalClient()


@st.cache_data(ttl=15)
def fetch_signals_cached(
    ticker: Optional[str] = None,
    event_type: Optional[str] = None,
    min_impact: Optional[float] = None,
    source_type: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Cached retrieval of signals from the SignalClient."""
    client = get_client()
    signals = client.get_signals(
        ticker=ticker,
        event_type=event_type,
        min_impact=min_impact,
        source_type=source_type,
        limit=limit,
    )
    return [s.model_dump() for s in signals]


@st.cache_data(ttl=15)
def fetch_summary_cached() -> Dict[str, Any]:
    """Cached retrieval of summary analytics."""
    client = get_client()
    return client.get_summary()


@st.cache_data(ttl=60)
def fetch_portfolio_cached(path: str = "data/portfolio.csv") -> Optional[pd.DataFrame]:
    """Cached loader of wholesale banking portfolio."""
    full_path = BASE_DIR / path if not Path(path).is_absolute() else Path(path)
    if not full_path.exists():
        return None
    try:
        return load_portfolio(full_path)
    except Exception as exc:
        st.error(f"Failed to load portfolio: {exc}")
        return None


def signals_to_dataframe(signals_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts a list of signal dicts into a structured DataFrame."""
    if not signals_data:
        return pd.DataFrame()
    df = pd.DataFrame(signals_data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Formatting & Badges
# ---------------------------------------------------------------------------


def render_sentiment_badge(label: str) -> str:
    lbl = (label or "neutral").lower()
    if lbl == "positive":
        return '<span class="badge badge-positive">▲ Positive</span>'
    elif lbl == "negative":
        return '<span class="badge badge-negative">▼ Negative</span>'
    return '<span class="badge badge-neutral">■ Neutral</span>'


def render_impact_badge(score: float) -> str:
    val = round(float(score), 1)
    if val >= 7.0:
        return f'<span class="badge badge-high">🔥 {val} High</span>'
    elif val >= 4.0:
        return f'<span class="badge badge-med">⚡ {val} Med</span>'
    return f'<span class="badge badge-low">ℹ {val} Low</span>'


def format_usd(val: float) -> str:
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.1f}K"
    return f"{sign}${abs_val:.2f}"


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------


def plot_event_donut(df: pd.DataFrame) -> go.Figure:
    """Renders a donut chart of event-type distribution."""
    if df.empty or "event_type" not in df.columns:
        return go.Figure()
    counts = df["event_type"].value_counts().reset_index()
    counts.columns = ["event_type", "count"]

    fig = px.pie(
        counts,
        names="event_type",
        values="count",
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#0F172A", width=2)),
    )
    fig.update_layout(
        **CHART_THEME,
        showlegend=False,
        height=320,
        title=dict(text="<b>Event Taxonomy Distribution</b>", font=dict(size=14, color=TEXT_COLOR)),
    )
    return fig


def plot_impact_histogram(df: pd.DataFrame) -> go.Figure:
    """Renders an impact score histogram with the high-severity threshold line."""
    if df.empty or "impact_score" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["impact_score"],
            xbins=dict(start=1.0, end=10.0, size=0.5),
            marker=dict(
                color=df["impact_score"],
                colorscale=[[0, ACCENT_BLUE], [0.65, AMBER_COLOR], [1.0, RED_COLOR]],
                line=dict(color="#0F172A", width=1),
            ),
            hovertemplate="Impact: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=7.0,
        line_width=2,
        line_dash="dash",
        line_color=RED_COLOR,
        annotation_text="Trigger (≥ 7.0)",
        annotation_position="top right",
        annotation_font=dict(color=RED_COLOR, size=11),
    )
    fig.update_layout(
        **CHART_THEME,
        height=320,
        title=dict(text="<b>Impact Score Distribution</b>", font=dict(size=14, color=TEXT_COLOR)),
        xaxis=dict(title="Impact Score (1-10)", gridcolor="#334155", dtick=1),
        yaxis=dict(title="Signals Count", gridcolor="#334155"),
    )
    return fig


def plot_sentiment_by_ticker(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average sentiment score by ticker."""
    if df.empty or "tickers" not in df.columns or "sentiment_score" not in df.columns:
        return go.Figure()

    # Explode tickers list
    rows = []
    for _, row in df.iterrows():
        tickers = row["tickers"] if isinstance(row["tickers"], list) else []
        for t in tickers:
            rows.append({"ticker": t, "sentiment_score": row["sentiment_score"]})

    if not rows:
        return go.Figure()

    ticker_df = pd.DataFrame(rows)
    agg = (
        ticker_df.groupby("ticker")
        .agg(avg_sent=("sentiment_score", "mean"), count=("sentiment_score", "count"))
        .reset_index()
    )
    agg = agg.sort_values(by="avg_sent", ascending=True).tail(15)

    colors = [RED_COLOR if s < -0.05 else (GREEN_COLOR if s > 0.05 else SLATE_MUTED) for s in agg["avg_sent"]]

    fig = go.Figure(
        go.Bar(
            x=agg["avg_sent"],
            y=agg["ticker"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
            text=[f"{v:+.2f} ({c})" for v, c in zip(agg["avg_sent"], agg["count"])],
            textposition="auto",
            hovertemplate="Ticker: %{y}<br>Avg Sentiment: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_THEME,
        height=320,
        title=dict(text="<b>Sentiment by Ticker (Top Tracked)</b>", font=dict(size=14, color=TEXT_COLOR)),
        xaxis=dict(title="Average Sentiment [-1 to +1]", gridcolor="#334155", range=[-1.05, 1.05]),
        yaxis=dict(gridcolor="#334155"),
    )
    return fig


def plot_volume_timeline(df: pd.DataFrame) -> go.Figure:
    """Grouped timeline of news vs social volume."""
    if df.empty or "timestamp" not in df.columns or "source_type" not in df.columns:
        return go.Figure()

    time_df = df.copy()
    time_df["time_bin"] = time_df["timestamp"].dt.floor("h")
    vol = time_df.groupby(["time_bin", "source_type"]).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    if "news" in vol.columns:
        fig.add_trace(
            go.Bar(
                x=vol["time_bin"],
                y=vol["news"],
                name="News Feeds",
                marker_color=ACCENT_BLUE,
            )
        )
    if "social" in vol.columns:
        fig.add_trace(
            go.Bar(
                x=vol["time_bin"],
                y=vol["social"],
                name="Social / Media",
                marker_color=AMBER_COLOR,
            )
        )
    fig.update_layout(
        **CHART_THEME,
        barmode="stack",
        height=320,
        title=dict(text="<b>Signal Ingestion Timeline</b>", font=dict(size=14, color=TEXT_COLOR)),
        xaxis=dict(gridcolor="#334155", title="Time"),
        yaxis=dict(gridcolor="#334155", title="Signal Volume"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Stress Test Visualizations
# ---------------------------------------------------------------------------


def plot_valuation_comparison(base_val: float, stressed_val: float) -> go.Figure:
    """Compares baseline vs stressed total portfolio market value."""
    pnl = stressed_val - base_val
    fig = go.Figure(
        data=[
            go.Bar(
                name="Portfolio Value",
                x=["Baseline Value", "Stressed Value"],
                y=[base_val, stressed_val],
                marker_color=[ACCENT_BLUE, RED_COLOR if pnl < 0 else GREEN_COLOR],
                text=[format_usd(base_val), format_usd(stressed_val)],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        **CHART_THEME,
        height=280,
        title=dict(text="<b>Portfolio Value: Before vs After</b>", font=dict(size=14, color=TEXT_COLOR)),
        yaxis=dict(title="Market Value (USD)", gridcolor="#334155"),
        showlegend=False,
    )
    return fig


def plot_pnl_waterfall(asset_class_breakdown: List[Dict[str, Any]]) -> go.Figure:
    """Waterfall chart showing P&L contributions across asset classes."""
    if not asset_class_breakdown:
        return go.Figure()

    labels = [d.get("asset_class", "Unknown") for d in asset_class_breakdown]
    values = [d.get("pnl", 0.0) for d in asset_class_breakdown]
    total_pnl = sum(values)

    fig = go.Figure(
        go.Waterfall(
            name="P&L",
            orientation="v",
            measure=["relative"] * len(labels) + ["total"],
            x=labels + ["Net Portfolio P&L"],
            textposition="outside",
            text=[format_usd(v) for v in values] + [format_usd(total_pnl)],
            y=values + [total_pnl],
            connector={"line": {"color": "#64748B"}},
            decreasing={"marker": {"color": RED_COLOR}},
            increasing={"marker": {"color": GREEN_COLOR}},
            totals={"marker": {"color": ACCENT_BLUE}},
        )
    )
    fig.update_layout(
        **CHART_THEME,
        height=320,
        title=dict(text="<b>P&L Waterfall by Asset Class</b>", font=dict(size=14, color=TEXT_COLOR)),
        yaxis=dict(title="P&L in USD", gridcolor="#334155"),
    )
    return fig


def plot_sector_and_country_bars(
    sector_breakdown: List[Dict[str, Any]],
    country_breakdown: List[Dict[str, Any]],
) -> go.Figure:
    """Side-by-side horizontal bars for losses by sector and country."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("<b>Loss by Sector (USD)</b>", "<b>Loss by Country (USD)</b>"),
        horizontal_spacing=0.15,
    )

    if sector_breakdown:
        sec_df = pd.DataFrame(sector_breakdown).sort_values(by="pnl", ascending=True).head(8)
        fig.add_trace(
            go.Bar(
                y=sec_df["sector"],
                x=sec_df["pnl"],
                orientation="h",
                marker_color=RED_COLOR,
                text=[format_usd(v) for v in sec_df["pnl"]],
                textposition="auto",
                name="Sector P&L",
            ),
            row=1,
            col=1,
        )

    if country_breakdown:
        cnt_df = pd.DataFrame(country_breakdown).sort_values(by="pnl", ascending=True).head(8)
        fig.add_trace(
            go.Bar(
                y=cnt_df["country"],
                x=cnt_df["pnl"],
                orientation="h",
                marker_color=AMBER_COLOR,
                text=[format_usd(v) for v in cnt_df["pnl"]],
                textposition="auto",
                name="Country P&L",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        **CHART_THEME,
        height=330,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")
    return fig


# ---------------------------------------------------------------------------
# Rebalancer (Module A) Components & Charts
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_or_generate_weights_history() -> pd.DataFrame:
    """Loads weights_history.csv or dynamically replays signals_sample.jsonl to produce it."""
    history_file = BASE_DIR / "data" / "weights_history.csv"
    if history_file.exists() and history_file.stat().st_size > 100:
        try:
            df = pd.read_csv(history_file)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
        except Exception:
            pass

    # Generate if not existing by replaying signals in hourly steps
    import json

    from src.modules.rebalancer.weights import IndexRebalancer
    from src.storage.store import get_signal_store

    store = get_signal_store()
    signals = store.query(limit=1000)
    if not signals:
        sample_path = BASE_DIR / "data" / "signals_sample.jsonl"
        if sample_path.exists():
            signals = []
            with open(sample_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        signals.append(RiskSignal.model_validate(json.loads(line)))

    reb = IndexRebalancer(history_path=history_file)
    reb.reset_to_base()
    if history_file.exists():
        try:
            history_file.unlink()
        except Exception:
            pass

    if signals:
        signals.sort(key=lambda s: s.timestamp)
        t_start = signals[0].timestamp
        t_end = signals[-1].timestamp
        cur = t_start
        while cur <= t_end:
            sub_signals = [s for s in signals if s.timestamp <= cur]
            reb.rebalance(sub_signals, now=cur)
            cur += timedelta(hours=1)

    if history_file.exists():
        df = pd.read_csv(history_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_backtest_results() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads backtest curve CSV and backtest summary JSON."""
    curve_path = BASE_DIR / "data" / "backtest_curve.csv"
    json_path = BASE_DIR / "docs" / "results" / "backtest.json"

    curve_df = pd.DataFrame()
    if curve_path.exists():
        try:
            curve_df = pd.read_csv(curve_path)
        except Exception:
            pass

    metrics_dict: Dict[str, Any] = {}
    if json_path.exists():
        try:
            import json

            with open(json_path, "r", encoding="utf-8") as f:
                metrics_dict = json.load(f)
        except Exception:
            pass

    return curve_df, metrics_dict


def plot_weights_stacked_area(history_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio weights over time with hover showing weight and sentiment."""
    if history_df.empty or "timestamp" not in history_df.columns or "ticker" not in history_df.columns:
        return go.Figure()

    pivot_w = (
        history_df.pivot_table(index="timestamp", columns="ticker", values="weight", aggfunc="last")
        .ffill()
        .fillna(1.0 / 15.0)
    )
    pivot_s = (
        history_df.pivot_table(index="timestamp", columns="ticker", values="score", aggfunc="last").ffill().fillna(0.0)
    )

    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    tickers = sorted(list(pivot_w.columns))

    for idx, ticker in enumerate(tickers):
        color = colors[idx % len(colors)]
        w_series = pivot_w[ticker]
        s_series = pivot_s[ticker] if ticker in pivot_s.columns else [0.0] * len(w_series)

        fig.add_trace(
            go.Scatter(
                x=pivot_w.index,
                y=w_series,
                mode="lines",
                name=ticker,
                stackgroup="one",
                line=dict(width=0.8, color=color),
                customdata=s_series,
                hovertemplate=f"<b>{ticker}</b><br>Time: %{{x|%m-%d %H:%M}}<br>Weight: %{{y:.2%}}<br>Sentiment Score: %{{customdata:+.3f}}<extra></extra>",
            )
        )

    theme = {**CHART_THEME, "margin": dict(l=20, r=20, t=55, b=95)}
    fig.update_layout(
        **theme,
        height=500,
        title=dict(
            text="<b>Portfolio Weight Trajectory Over Time (15 Stocks)</b>",
            font=dict(size=14, color=TEXT_COLOR),
            y=0.97,
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(gridcolor="#334155", title="Time"),
        yaxis=dict(gridcolor="#334155", title="Allocation", tickformat=".0%", range=[0, 1.01]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    return fig


def plot_current_vs_base_weights(
    current_weights: Dict[str, float],
    base_weights: Dict[str, float],
) -> go.Figure:
    """Bar chart comparing current tactical weight vs base equal weight, colored by direction."""
    if not current_weights:
        return go.Figure()

    tickers = sorted(list(current_weights.keys()))
    c_vals = [current_weights[t] * 100 for t in tickers]
    b_val = (base_weights.get(tickers[0], 1.0 / len(tickers))) * 100
    deltas_bps = [(current_weights[t] - base_weights.get(t, 1.0 / len(tickers))) * 10000 for t in tickers]

    bar_colors = [GREEN_COLOR if d > 1e-4 else (RED_COLOR if d < -1e-4 else SLATE_MUTED) for d in deltas_bps]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=c_vals,
            marker_color=bar_colors,
            text=[f"{c:.2f}%\n({d:+.0f}bp)" for c, d in zip(c_vals, deltas_bps)],
            textposition="auto",
            hovertemplate="Ticker: %{x}<br>Current Weight: %{y:.2f}%<br>Shift: %{customdata:+.0f} bps<extra></extra>",
            customdata=deltas_bps,
        )
    )
    fig.add_hline(
        y=b_val,
        line_width=1.5,
        line_dash="dash",
        line_color=ACCENT_BLUE,
        annotation_text=f"Base Weight ({b_val:.2f}%)",
        annotation_position="top right",
        annotation_font=dict(color=ACCENT_BLUE, size=11),
    )

    theme_bar = {**CHART_THEME, "margin": dict(l=20, r=20, t=55, b=30)}
    fig.update_layout(
        **theme_bar,
        height=360,
        title=dict(
            text="<b>Current Tactical Weight vs Base Benchmark</b>",
            font=dict(size=14, color=TEXT_COLOR),
            y=0.97,
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(gridcolor="#334155", title="Ticker Universe"),
        yaxis=dict(gridcolor="#334155", title="Allocation (%)"),
    )
    return fig


def plot_backtest_equity_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Line chart comparing strategy vs benchmark wealth progression."""
    if curve_df.empty or "date" not in curve_df.columns:
        return go.Figure()

    fig = go.Figure()
    if "strategy_equity" in curve_df.columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df["date"],
                y=curve_df["strategy_equity"],
                name="Sentiment Strategy (Gross)",
                line=dict(color=GREEN_COLOR, width=2),
                hovertemplate="Strategy: %{y:.4f}<extra></extra>",
            )
        )
    if "net_strategy_equity" in curve_df.columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df["date"],
                y=curve_df["net_strategy_equity"],
                name="Sentiment Strategy (Net of Cost)",
                line=dict(color=ACCENT_BLUE, width=2, dash="dot"),
                hovertemplate="Net Strategy: %{y:.4f}<extra></extra>",
            )
        )
    if "benchmark_equity" in curve_df.columns:
        fig.add_trace(
            go.Scatter(
                x=curve_df["date"],
                y=curve_df["benchmark_equity"],
                name="Equal-Weight Benchmark",
                line=dict(color=SLATE_MUTED, width=1.5, dash="dash"),
                hovertemplate="Benchmark: %{y:.4f}<extra></extra>",
            )
        )

    theme_bt = {**CHART_THEME, "margin": dict(l=20, r=20, t=55, b=80)}
    fig.update_layout(
        **theme_bt,
        height=430,
        title=dict(
            text="<b>Equity Growth Curve: Tactical Strategy vs Benchmark</b>",
            font=dict(size=14, color=TEXT_COLOR),
            y=0.97,
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(gridcolor="#334155", title="Trading Date"),
        yaxis=dict(gridcolor="#334155", title="Cumulative Wealth ($1.0 Base)"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )
    return fig
