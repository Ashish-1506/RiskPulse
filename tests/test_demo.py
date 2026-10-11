"""Unit and integration tests for demo-proofing, headline injection, and stress triggers."""

import pytest

from src.api.client import SignalClient
from src.config import get_config
from src.modules.stress.engine import StressEngine
from src.modules.stress.portfolio import load_portfolio
from src.modules.stress.scenarios import should_trigger
from src.nlp.engine import RiskEngine
from src.storage.store import get_signal_store


@pytest.fixture
def vader_engine() -> RiskEngine:
    cfg = get_config()
    cfg.sentiment.backend = "vader"
    return RiskEngine(sentiment_backend="vader", use_zero_shot=False)


def test_injected_geopolitical_headline_triggers_stress(vader_engine: RiskEngine):
    """Verifies the injected geopolitical headline returns impact >= 7 and triggers stress test."""
    headline = "Naval clash closes key shipping lane; oil jumps and sanctions threatened against exporters"
    sig = vader_engine.process_single(headline, source="reuters", source_type="news")

    assert sig is not None
    assert sig.event_type == "Geopolitical"
    assert sig.impact_score >= 7.0, f"Expected impact >= 7.0, got {sig.impact_score}"
    assert should_trigger(sig) is True

    # Run stress engine on wholesale banking portfolio
    portfolio_df = load_portfolio("data/portfolio.csv")
    engine = StressEngine()
    result = engine.run(portfolio_df, sig)

    assert result is not None
    assert result.total_pnl < 0, f"Expected portfolio drawdown, got PnL: {result.total_pnl}"
    assert result.scenario_params["event_type"] == "Geopolitical"
    assert len(result.top_10_losers) == 10


def test_all_three_demo_scenarios_reach_high_impact(vader_engine: RiskEngine):
    """Verifies all 3 demo scenarios deterministically achieve impact >= 7.0 and trigger stress testing."""
    scenarios = [
        ("Naval clash closes key shipping lane; oil jumps and sanctions threatened against exporters", "Geopolitical"),
        ("Regional lender collapses after bond default; regulators warn of contagion across banks", "Credit Event"),
        ("Central bank delivers emergency rate hike as inflation surges past forecasts", "Monetary Policy"),
    ]

    for text, expected_event in scenarios:
        sig = vader_engine.process_single(text, source="reuters", source_type="news")
        assert sig is not None
        assert sig.event_type == expected_event, f"Expected {expected_event}, got {sig.event_type}"
        assert sig.impact_score >= 7.0, f"Headline '{text}' got impact {sig.impact_score} < 7.0"
        assert should_trigger(sig) is True


def test_demo_reset_and_delete_injected():
    """Verifies that injecting an ad_hoc signal increases count and delete_injected resets cleanly."""
    store = get_signal_store()
    client = SignalClient()

    # First clean up any prior injected signals
    store.delete_injected(sources=("ad_hoc", "demo_injected"))
    initial_summary = store.summary()
    initial_total = initial_summary["total_signals"]

    # Ensure VADER backend is active for fast execution
    cfg = get_config()
    cfg.sentiment.backend = "vader"

    import time

    unique_text = f"Lender collapses into bankruptcy and default causing contagion across banks {time.time()}"

    # Inject an ad_hoc test signal
    sig = client.analyze(
        text=unique_text,
        source="ad_hoc",
        source_type="news",
    )
    assert sig is not None

    after_inject_total = store.summary()["total_signals"]
    assert after_inject_total >= initial_total + 1

    # Call reset/cleanup
    deleted_count = store.delete_injected(sources=("ad_hoc", "demo_injected"))
    assert deleted_count >= 1

    final_total = store.summary()["total_signals"]
    assert final_total == initial_total


def test_offline_fallback_resilience():
    """Verifies SignalClient seamlessly falls back to local SQLite when API is offline."""
    # Point client to a non-existent port
    offline_client = SignalClient(base_url="http://127.0.0.1:59999", timeout=0.2)
    assert offline_client.is_api_online() is False

    signals = offline_client.get_signals(limit=10)
    assert isinstance(signals, list)
    assert len(signals) > 0

    summary = offline_client.get_summary()
    assert "total_signals" in summary
    assert summary["total_signals"] > 0
