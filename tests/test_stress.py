"""Unit and integration tests for Wholesale Banking Stress Testing Engine."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.api.schemas import RiskSignal
from src.modules.stress.engine import StressEngine
from src.modules.stress.portfolio import load_portfolio
from src.modules.stress.scenarios import ShockPackage, should_trigger
from src.modules.stress.valuation import value_portfolio

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "data" / "portfolio.csv"


@pytest.fixture(scope="module")
def portfolio_df():
    """Loads validated portfolio fixture once."""
    assert PORTFOLIO_PATH.is_file(), "data/portfolio.csv must exist before running stress tests"
    return load_portfolio(PORTFOLIO_PATH)


@pytest.fixture(scope="module")
def stress_engine():
    """Initializes StressEngine singleton fixture."""
    return StressEngine()


def make_test_signal(
    event_type: str = "Geopolitical",
    impact_score: float = 8.5,
    sectors: list[str] | None = None,
    countries: list[str] | None = None,
) -> RiskSignal:
    """Helper creating standardized RiskSignal instances for testing."""
    return RiskSignal(
        signal_id="TEST_SIG_001",
        timestamp=datetime.now(timezone.utc),
        source="newsapi",
        source_type="news",
        headline=f"Escalating crisis trigger for {event_type}",
        tickers=[],
        countries=countries or [],
        sectors=sectors or [],
        sentiment_score=-0.75,
        sentiment_label="NEGATIVE",
        event_type=event_type,
        event_confidence=0.90,
        impact_score=impact_score,
        impact_components={"event_prior": 0.8},
        impact_rationale="Severe risk event detected",
        model_version="finbert+rules-v1",
    )


def test_zero_shock_produces_zero_pnl(portfolio_df, stress_engine):
    """Zero shock must produce exactly zero P&L and zero delta ECL."""
    zero_shocks = ShockPackage(
        scenario_name="Zero Shock Baseline",
        event_type="Zero",
        severity=0.0,
        equity_shock=0.0,
        rates_bps=0.0,
        spread_bps=0.0,
        fx_em_shock=0.0,
        fx_usd_shock=0.0,
        commodity_shock=0.0,
        pd_multiplier=1.0,
        affected_sectors=[],
        affected_countries=[],
        exposure_link_bonus=0.0,
        is_custom=True,
    )

    res = stress_engine.run(portfolio_df, zero_shocks)
    assert res is not None
    assert abs(res.total_pnl) < 1e-2, f"Expected 0.0 P&L, got {res.total_pnl}"
    assert abs(res.total_d_ecl) < 1e-2, f"Expected 0.0 dECL, got {res.total_d_ecl}"
    assert res.base_value == res.stressed_value


def test_loss_increases_monotonically_with_severity(portfolio_df, stress_engine):
    """Stress test loss must increase monotonically as severity increases."""
    signal = make_test_signal(event_type="Geopolitical", impact_score=10.0)

    severities = [0.2, 0.5, 0.8, 1.0, 1.3]
    losses = []

    for sev in severities:
        res = stress_engine.run(portfolio_df, signal, severity_override=sev)
        assert res is not None
        # In a severe crisis, total PnL is negative (a loss)
        loss = -res.total_pnl
        losses.append(loss)

    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], (
            f"Loss at severity {severities[i]} ({losses[i]}) was not strictly less than "
            f"loss at {severities[i + 1]} ({losses[i + 1]})"
        )


def test_rates_increase_lowers_bond_value(portfolio_df):
    """+100 bps rate increase must analytically lower fixed-income bond values."""
    rates_up_shock = ShockPackage(
        scenario_name="Pure Rate Hike",
        event_type="Custom",
        severity=1.0,
        equity_shock=0.0,
        rates_bps=100.0,  # +100 bps
        spread_bps=0.0,
        fx_em_shock=0.0,
        fx_usd_shock=0.0,
        commodity_shock=0.0,
        pd_multiplier=1.0,
        is_custom=True,
    )

    valued_df = value_portfolio(portfolio_df, rates_up_shock)
    bonds = valued_df[valued_df["asset_class"] == "Bond"]

    assert len(bonds) > 0
    # Every bond with positive duration must suffer a negative price move
    fixed_bonds = bonds[bonds["mod_duration"] > 0]
    assert (fixed_bonds["pnl"] < 0).all(), "Rate increase did not lower all bond values"
    assert fixed_bonds["pnl"].sum() < 0


def test_spread_widening_raises_cds_protection_value(portfolio_df):
    """Credit spread widening must raise the value of long CDS protection (direction=+1)."""
    spread_blowout = ShockPackage(
        scenario_name="Pure Spread Blowout",
        event_type="Custom",
        severity=1.0,
        equity_shock=0.0,
        rates_bps=0.0,
        spread_bps=200.0,  # +200 bps
        fx_em_shock=0.0,
        fx_usd_shock=0.0,
        commodity_shock=0.0,
        pd_multiplier=1.0,  # No CVA effect for clean spread test
        is_custom=True,
    )

    valued_df = value_portfolio(portfolio_df, spread_blowout)
    cds_buyers = valued_df[(valued_df["instrument_type"] == "CDS") & (valued_df["direction"] == 1)]

    assert len(cds_buyers) > 0
    # When spreads widen, protection buyers gain: P&L > 0
    assert (cds_buyers["pnl"] > 0).all(), "Spread widening did not raise CDS protection value"


def test_stressed_pd_never_exceeds_one(portfolio_df):
    """Stressed probability of default must be strictly capped at 1.0 (100%)."""
    extreme_crisis = ShockPackage(
        scenario_name="Apocalyptic Crisis",
        event_type="Custom",
        severity=1.0,
        equity_shock=-0.50,
        rates_bps=500.0,
        spread_bps=1000.0,
        fx_em_shock=-0.50,
        fx_usd_shock=0.10,
        commodity_shock=1.0,
        pd_multiplier=100.0,  # Extreme 100x multiplier
        is_custom=True,
    )

    valued_df = value_portfolio(portfolio_df, extreme_crisis)
    assert (valued_df["pd_stressed"] <= 1.0).all(), "Stressed PD exceeded 1.0"
    assert (valued_df["pd_stressed"] >= 0.0).all(), "Stressed PD below 0.0"


def test_trigger_rule_respects_threshold_and_events():
    """Trigger rule must trigger only when impact >= threshold and event has a scenario."""
    # 1. Below threshold (impact 6.5 < 7.0) -> False
    sub_sig = make_test_signal(event_type="Geopolitical", impact_score=6.5)
    assert not should_trigger(sub_sig)

    # 2. At or above threshold (impact 7.0) with mapped scenario -> True
    valid_sig = make_test_signal(event_type="Geopolitical", impact_score=7.5)
    assert should_trigger(valid_sig)

    # 3. High impact but 'Other' event type with no stress scenario -> False
    other_sig = make_test_signal(event_type="Other", impact_score=9.5)
    assert not should_trigger(other_sig)


def test_deterministic_output(portfolio_df, stress_engine):
    """Stress tests must be 100% deterministic on identical input."""
    signal = make_test_signal(event_type="Macroeconomic", impact_score=8.0)

    res1 = stress_engine.run(portfolio_df, signal)
    res2 = stress_engine.run(portfolio_df, signal)

    assert res1 is not None and res2 is not None
    assert res1.total_pnl == res2.total_pnl
    assert res1.total_d_ecl == res2.total_d_ecl
    assert res1.stressed_value == res2.stressed_value


def test_reverse_stress_bisection(portfolio_df, stress_engine):
    """Reverse stress testing finds critical severity exceeding target loss limit."""
    target_loss = 2.5  # 2.5% loss limit
    rev_res = stress_engine.reverse_stress(portfolio_df, "Geopolitical", max_loss_pct=target_loss)

    assert rev_res["critical_severity"] > 0.0
    assert rev_res["critical_impact_score"] > 0.0
    assert abs(rev_res["actual_loss_pct"] - target_loss) < 0.10


def test_run_all_triggered(portfolio_df, stress_engine):
    """run_all_triggered processes multiple signals and filters untriggered ones."""
    sig1 = make_test_signal("Geopolitical", impact_score=8.0)  # Triggers
    sig2 = make_test_signal("Credit Event", impact_score=5.0)  # Below threshold -> No trigger
    sig3 = make_test_signal("Macroeconomic", impact_score=8.5)  # Triggers
    sig4 = make_test_signal("Other", impact_score=9.0)  # No scenario -> No trigger

    results = stress_engine.run_all_triggered(portfolio_df, [sig1, sig2, sig3, sig4])
    assert len(results) == 2
    assert {r.scenario_params["event_type"] for r in results} == {"Geopolitical", "Macroeconomic"}
