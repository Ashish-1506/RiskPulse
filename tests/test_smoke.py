"""Smoke test validating configuration parsing and ticker universe setup."""

import pytest

from src.config import get_config, get_events_config, get_scenarios_config


def test_smoke_tickers_configuration():
    """Validates that exactly 15 tickers are configured with correct attributes."""
    cfg = get_config()
    expected_tickers = {
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
    }

    loaded_tickers = set(cfg.tickers.keys())
    assert loaded_tickers == expected_tickers, f"Ticker mismatch: {loaded_tickers ^ expected_tickers}"
    assert len(loaded_tickers) == 15

    total_weight = 0.0
    for symbol, meta in cfg.tickers.items():
        assert meta.name, f"Ticker {symbol} missing name"
        assert meta.sector, f"Ticker {symbol} missing sector"
        assert meta.base_weight > 0.0, f"Ticker {symbol} has invalid base weight"
        assert len(meta.aliases) >= 1, f"Ticker {symbol} missing aliases"
        total_weight += meta.base_weight

    # Sum of equal base weights should equal ~1.0
    assert pytest.approx(total_weight, rel=1e-2) == 1.0


def test_smoke_events_configuration():
    """Validates that events taxonomy contains required categorical types."""
    events_cfg = get_events_config()
    assert len(events_cfg.events) >= 12
    required_events = [
        "Geopolitical",
        "Macroeconomic",
        "Monetary Policy",
        "Credit Event",
        "Merger/Acquisition",
        "Earnings/Guidance",
        "Regulatory/Legal",
        "Product Launch",
        "Cyber/Operational",
        "Natural Disaster",
        "Supply Chain/Commodity",
        "Other",
    ]
    for ev in required_events:
        assert ev in events_cfg.events, f"Missing event type: {ev}"
        assert 0.0 <= events_cfg.events[ev].prior_severity <= 1.0


def test_smoke_scenarios_configuration():
    """Validates that stress shock matrices are loaded."""
    scenarios_cfg = get_scenarios_config()
    assert len(scenarios_cfg.scenarios) > 0
    for name, item in scenarios_cfg.scenarios.items():
        assert item.name
        assert item.shocks.equity_pct <= 0.0 or item.shocks.equity_pct > -1.0
        assert item.shocks.pd_multiplier >= 1.0
