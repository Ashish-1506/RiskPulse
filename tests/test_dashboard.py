"""Unit and smoke tests for Streamlit dashboard components and chart generators."""

import pandas as pd

from src.dashboard.components import (
    fetch_portfolio_cached,
    fetch_signals_cached,
    format_usd,
    plot_event_donut,
    plot_impact_histogram,
    plot_pnl_waterfall,
    plot_sector_and_country_bars,
    plot_sentiment_by_ticker,
    plot_valuation_comparison,
    plot_volume_timeline,
    render_impact_badge,
    render_sentiment_badge,
    signals_to_dataframe,
)
from src.modules.stress.engine import StressEngine
from src.modules.stress.scenarios import load_scenarios


def test_badges_and_formatters():
    pos_b = render_sentiment_badge("positive")
    neg_b = render_sentiment_badge("negative")
    neu_b = render_sentiment_badge("neutral")
    assert "Positive" in pos_b
    assert "Negative" in neg_b
    assert "Neutral" in neu_b

    imp_high = render_impact_badge(8.5)
    imp_med = render_impact_badge(5.0)
    imp_low = render_impact_badge(2.0)
    assert "High" in imp_high
    assert "Med" in imp_med
    assert "Low" in imp_low

    assert format_usd(5.2e9) == "$5.20B"
    assert format_usd(150.5e6) == "$150.5M"
    assert format_usd(-12.4e6) == "-$12.4M"


def test_signals_data_and_charts_with_sample():
    signals = fetch_signals_cached(limit=20)
    assert isinstance(signals, list)

    df = signals_to_dataframe(signals)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        fig_donut = plot_event_donut(df)
        assert fig_donut is not None
        fig_hist = plot_impact_histogram(df)
        assert fig_hist is not None
        fig_sent = plot_sentiment_by_ticker(df)
        assert fig_sent is not None
        fig_vol = plot_volume_timeline(df)
        assert fig_vol is not None


def test_charts_handle_empty_data_gracefully():
    empty_df = pd.DataFrame()
    assert plot_event_donut(empty_df) is not None
    assert plot_impact_histogram(empty_df) is not None
    assert plot_sentiment_by_ticker(empty_df) is not None
    assert plot_volume_timeline(empty_df) is not None
    assert plot_pnl_waterfall([]) is not None
    assert plot_sector_and_country_bars([], []) is not None


def test_stress_test_visualizations():
    portfolio = fetch_portfolio_cached()
    assert portfolio is not None
    assert len(portfolio) > 0

    scenarios = load_scenarios()
    assert "Geopolitical" in scenarios

    engine = StressEngine()
    signals = fetch_signals_cached(min_impact=7.0, limit=5)
    if signals:
        from src.api.schemas import RiskSignal

        sig = RiskSignal.model_validate(signals[0])
        res = engine.run(portfolio=portfolio, signal_or_scenario=sig)
        if res:
            fig_val = plot_valuation_comparison(res.base_value, res.stressed_value)
            assert fig_val is not None
            fig_wf = plot_pnl_waterfall(res.breakdowns.get("asset_class", []))
            assert fig_wf is not None
            fig_bars = plot_sector_and_country_bars(
                res.breakdowns.get("sector", []),
                res.breakdowns.get("country", []),
            )
            assert fig_bars is not None


def test_rebalancer_dashboard_components():
    from src.dashboard.components import (
        load_backtest_results,
        load_or_generate_weights_history,
        plot_backtest_equity_curve,
        plot_current_vs_base_weights,
        plot_weights_stacked_area,
    )

    history_df = load_or_generate_weights_history()
    assert isinstance(history_df, pd.DataFrame)
    if not history_df.empty:
        fig_area = plot_weights_stacked_area(history_df)
        assert fig_area is not None

    current_w = {"AAPL": 0.08, "MSFT": 0.05}
    base_w = {"AAPL": 0.0667, "MSFT": 0.0667}
    fig_bars = plot_current_vs_base_weights(current_w, base_w)
    assert fig_bars is not None

    curve_df, metrics = load_backtest_results()
    assert isinstance(curve_df, pd.DataFrame)
    fig_curve = plot_backtest_equity_curve(curve_df)
    assert fig_curve is not None
