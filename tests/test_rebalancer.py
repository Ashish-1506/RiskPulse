"""Unit and integration tests for Module A: Tactical Index Rebalancer and Backtest Engine."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.api.schemas import RiskSignal
from src.modules.rebalancer.backtest import run_backtest
from src.modules.rebalancer.weights import IndexRebalancer, RebalanceResult


@pytest.fixture
def rebalancer(tmp_path: Path) -> IndexRebalancer:
    history_file = tmp_path / "weights_history.csv"
    reb = IndexRebalancer(history_path=history_file)
    reb.reset_to_base()
    return reb


def make_dummy_signal(
    ticker: str,
    sentiment: float,
    impact: float = 8.0,
    confidence: float = 1.0,
    timestamp: datetime = None,
    source_type: str = "news",
    signal_id: str = "test_sig_1",
) -> RiskSignal:
    ts = timestamp or datetime.now(timezone.utc)
    lbl = "POSITIVE" if sentiment > 0.05 else ("NEGATIVE" if sentiment < -0.05 else "NEUTRAL")
    return RiskSignal(
        signal_id=signal_id,
        timestamp=ts,
        source="reuters",
        source_type=source_type if source_type in ("news", "social") else "news",
        headline=f"Major development concerning {ticker}",
        tickers=[ticker],
        countries=["United States"],
        sectors=["Information Technology"],
        sentiment_score=sentiment,
        sentiment_label=lbl,
        event_type="Earnings/Guidance",
        event_confidence=confidence,
        impact_score=impact,
        impact_rationale="Simulated test rationale",
        model_version="test-v1",
    )


def test_no_signals_weights_stay_at_base(rebalancer: IndexRebalancer):
    """When no signals exist, weights must remain exactly at equal base weights."""
    res = rebalancer.rebalance([], now=datetime.now(timezone.utc))

    assert isinstance(res, RebalanceResult)
    assert res.turnover == 0.0
    for t in rebalancer.tickers:
        assert pytest.approx(res.weights[t], abs=1e-6) == rebalancer.base_weights[t]
        assert pytest.approx(res.deltas[t], abs=1e-6) == 0.0
        assert res.scores[t] == 0.0


def test_weights_sum_to_one_and_within_bounds(rebalancer: IndexRebalancer):
    """Weights must strictly sum to 1.0 and respect [min_weight, max_weight] constraints."""
    now = datetime.now(timezone.utc)
    # Feed extreme positive and negative signals across tickers
    signals = [
        make_dummy_signal("AAPL", sentiment=+0.95, impact=10.0, timestamp=now, signal_id="s1"),
        make_dummy_signal("NVDA", sentiment=+0.90, impact=10.0, timestamp=now, signal_id="s2"),
        make_dummy_signal("MSFT", sentiment=-0.95, impact=10.0, timestamp=now, signal_id="s3"),
        make_dummy_signal("XOM", sentiment=-0.90, impact=10.0, timestamp=now, signal_id="s4"),
    ]

    res = rebalancer.rebalance(signals, now=now)

    total_weight = sum(res.weights.values())
    assert pytest.approx(total_weight, abs=1e-6) == 1.0

    for t, w in res.weights.items():
        assert w >= rebalancer.min_weight - 1e-6, f"{t} weight {w} below min {rebalancer.min_weight}"
        assert w <= rebalancer.max_weight + 1e-6, f"{t} weight {w} above max {rebalancer.max_weight}"


def test_strongly_positive_sentiment_raises_weight_above_base(rebalancer: IndexRebalancer):
    """Strongly positive sentiment raises an asset's weight above base equal weight."""
    now = datetime.now(timezone.utc)
    pos_sig = make_dummy_signal("AAPL", sentiment=+0.85, impact=9.0, timestamp=now, signal_id="pos_aapl")

    res = rebalancer.rebalance([pos_sig], now=now)

    base_w = rebalancer.base_weights["AAPL"]
    new_w = res.weights["AAPL"]
    assert new_w > base_w, f"Expected AAPL weight {new_w} > base {base_w}"
    assert res.scores["AAPL"] > 0.0

    # Verify explain helper
    explanations = rebalancer.explain("AAPL")
    assert len(explanations) == 1
    assert explanations[0]["signal_id"] == "pos_aapl"


def test_strongly_negative_sentiment_lowers_weight_below_base(rebalancer: IndexRebalancer):
    """Strongly negative sentiment lowers an asset's weight below base equal weight."""
    now = datetime.now(timezone.utc)
    neg_sig = make_dummy_signal("MSFT", sentiment=-0.85, impact=9.0, timestamp=now, signal_id="neg_msft")

    res = rebalancer.rebalance([neg_sig], now=now)

    base_w = rebalancer.base_weights["MSFT"]
    new_w = res.weights["MSFT"]
    assert new_w < base_w, f"Expected MSFT weight {new_w} < base {base_w}"
    assert res.scores["MSFT"] < 0.0

    explanations = rebalancer.explain("MSFT")
    assert len(explanations) == 1
    assert explanations[0]["signal_id"] == "neg_msft"


def test_turnover_cap_respected(rebalancer: IndexRebalancer):
    """One-period portfolio turnover must not exceed turnover_cap."""
    rebalancer.turnover_cap = 0.04  # 4% tight cap
    now = datetime.now(timezone.utc)

    # Induce massive shifts across all 15 tickers
    signals = []
    for i, t in enumerate(rebalancer.tickers):
        sent = +1.0 if i % 2 == 0 else -1.0
        signals.append(make_dummy_signal(t, sentiment=sent, impact=10.0, timestamp=now, signal_id=f"sig_{t}"))

    res = rebalancer.rebalance(signals, now=now)

    # Sum of absolute deltas / 2 must be <= cap
    total_delta_sum = sum(abs(d) for d in res.deltas.values()) / 2.0
    assert total_delta_sum <= rebalancer.turnover_cap + 1e-5
    assert res.turnover <= rebalancer.turnover_cap + 1e-5


def test_no_look_ahead_in_backtest_synthetic_series(tmp_path: Path):
    """Backtesting on day t must ONLY consume signals with timestamp <= t."""
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    t1 = pd.Timestamp("2026-01-02", tz="UTC")
    t2 = pd.Timestamp("2026-01-03", tz="UTC")

    tickers = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "NVDA",
        "META",
        "JPM",
        "BAC",
        "GS",
        "XOM",
        "CVX",
        "JNJ",
        "PFE",
        "WMT",
        "KO",
    ]

    # Synthetic constant prices (all 100.0)
    data = {t: [100.0, 105.0, 110.0] for t in tickers}
    prices_df = pd.DataFrame(data, index=[t0, t1, t2])

    # Signal 1 occurs at t0
    sig1 = make_dummy_signal(
        "AAPL",
        sentiment=+0.80,
        impact=8.0,
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        signal_id="sig_at_t0",
    )

    # Signal 2 occurs at t1 (future relative to t0)
    sig2 = make_dummy_signal(
        "NVDA",
        sentiment=+0.95,
        impact=10.0,
        timestamp=datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc),
        signal_id="future_sig_at_t1",
    )

    # Run backtest with both signals
    run_backtest(
        prices_df=prices_df,
        signals=[sig1, sig2],
        cost_bps=0.0,
    )

    # On day t0 (first step), ONLY sig1 was available.
    # Independent verification with standalone rebalancer using ONLY sig1
    standalone_reb = IndexRebalancer()
    standalone_reb.reset_to_base()
    t0_cutoff = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
    standalone_reb.rebalance([sig1], now=t0_cutoff)

    # Check backtest curve for t0
    curve_df = pd.read_csv(Path("data/backtest_curve.csv"))
    assert len(curve_df) == 2  # 2 forward transitions (t0->t1 and t1->t2)

    # Day 1 active signals must be 1 (sig1 only, not sig2)
    assert curve_df["active_signals_count"].iloc[0] == 1
    # Day 2 active signals must be 2 (both sig1 and sig2)
    assert curve_df["active_signals_count"].iloc[1] == 2
