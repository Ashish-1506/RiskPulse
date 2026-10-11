"""Unit and integration tests for Wholesale Banking Portfolio module."""

import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_portfolio import adapt_external_portfolio, generate_synthetic_portfolio
from src.modules.stress.portfolio import (
    REQUIRED_COLUMNS,
    VALID_ASSET_CLASSES,
    VALID_RATINGS,
    load_portfolio,
    portfolio_summary,
    validate_portfolio,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE_DIR / "data" / "portfolio.csv"


def test_portfolio_file_exists_and_loads():
    """Validates that data/portfolio.csv exists and loads cleanly with load_portfolio."""
    assert PORTFOLIO_PATH.is_file(), f"{PORTFOLIO_PATH} not found"
    df = load_portfolio(PORTFOLIO_PATH)
    assert len(df) > 0


def test_portfolio_schema_and_types():
    """Validates all required columns, absence of negative notionals, and rating validity."""
    df = load_portfolio(PORTFOLIO_PATH)

    # 1. Check all required columns
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"

    # 2. No negative notionals
    assert (df["notional"] >= 0).all(), "Found negative notionals"

    # 3. Valid ratings
    ratings = set(df["rating"].unique())
    assert ratings.issubset(VALID_RATINGS), f"Invalid ratings found: {ratings - VALID_RATINGS}"

    # 4. Valid asset classes
    asset_classes = set(df["asset_class"].unique())
    assert asset_classes.issubset(VALID_ASSET_CLASSES), f"Invalid asset classes: {asset_classes - VALID_ASSET_CLASSES}"

    # 5. Valid directions
    directions = set(df["direction"].unique())
    assert directions.issubset({1, -1, 1.0, -1.0}), f"Invalid directions: {directions}"

    # 6. Valid PD and LGD bounds
    assert (df["pd_1y"] >= 0.0).all() and (df["pd_1y"] <= 1.0).all()
    assert (df["lgd"] >= 0.0).all() and (df["lgd"] <= 1.0).all()

    # 7. Non-negative maturity
    assert (df["maturity_years"] >= 0.0).all()


def test_portfolio_composition_within_tolerance():
    """Validates that the generated portfolio meets composition targets within tolerances."""
    df = load_portfolio(PORTFOLIO_PATH)
    stats = portfolio_summary(df)

    total_mv = stats["total_market_value"]

    # Target ~$5.0 Billion (+/- 5%)
    assert 4_750_000_000 <= total_mv <= 5_250_000_000, f"Total market value {total_mv} outside target range"

    # Check asset class weights
    ac_map = {item["asset_class"]: item["pct_market_value"] for item in stats["by_asset_class"]}

    # Loans ~45% (allow 40% - 50%)
    assert 40.0 <= ac_map.get("Loan", 0.0) <= 50.0, f"Loan weight {ac_map.get('Loan')}% outside tolerance"

    # Bonds ~30% (allow 25% - 35%)
    assert 25.0 <= ac_map.get("Bond", 0.0) <= 35.0, f"Bond weight {ac_map.get('Bond')}% outside tolerance"

    # Derivatives ~18% (allow 13% - 23%)
    assert 13.0 <= ac_map.get("Derivative", 0.0) <= 23.0, (
        f"Derivative weight {ac_map.get('Derivative')}% outside tolerance"
    )

    # Equity ~7% (allow 4% - 10%)
    assert 4.0 <= ac_map.get("Equity", 0.0) <= 10.0, f"Equity weight {ac_map.get('Equity')}% outside tolerance"


def test_portfolio_reproducibility():
    """Validates that generating portfolio with fixed seed (42) is 100% deterministic."""
    df1 = generate_synthetic_portfolio(seed=42)
    df2 = generate_synthetic_portfolio(seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_no_real_company_names():
    """Ensures counterparties are strictly synthetic (e.g. Counterparty_001, Sovereign_US)."""
    df = load_portfolio(PORTFOLIO_PATH)

    counterparty_names = df["counterparty"].tolist()

    # Disallowed real-world company names
    forbidden_real_names = [
        "apple",
        "google",
        "microsoft",
        "tesla",
        "amazon",
        "meta",
        "jpmorgan",
        "goldman",
        "pfizer",
        "exxon",
        "chevron",
        "walmart",
        "citigroup",
        "bank of america",
        "morgan stanley",
        "toyota",
        "aramco",
    ]

    for cp in counterparty_names:
        cp_lower = cp.lower()
        for forbidden in forbidden_real_names:
            assert forbidden not in cp_lower, f"Real company name '{forbidden}' found in counterparty '{cp}'"

        # Verify synthetic naming pattern
        is_synthetic = bool(re.match(r"^Counterparty_\d+$", cp)) or bool(re.match(r"^Sovereign_[A-Za-z_]+$", cp))
        assert is_synthetic, f"Counterparty '{cp}' does not match synthetic pattern"


def test_portfolio_summary_metrics():
    """Validates summary statistics calculations (DV01, CS01, EAD, breakdowns)."""
    df = load_portfolio(PORTFOLIO_PATH)
    stats = portfolio_summary(df)

    assert stats["position_count"] == len(df)
    assert stats["total_notional"] > 0
    assert stats["total_ead"] > 0
    assert 0.0001 <= stats["weighted_avg_pd"] <= 0.10
    assert 0.20 <= stats["weighted_avg_lgd"] <= 0.80

    # DV01 & CS01 are calculated
    assert stats["total_dv01"] != 0
    assert stats["total_cs01"] != 0

    # Sector breakdown covers core sectors
    sectors = {s["sector"] for s in stats["by_sector"]}
    assert "Energy" in sectors
    assert "Financials" in sectors
    assert "Technology" in sectors
    assert "Real Estate" in sectors

    # Country breakdown covers key concentrations
    countries = {c["country"] for c in stats["by_country"]}
    assert "UAE" in countries
    assert "China" in countries
    assert "US" in countries


def test_external_file_adapter():
    """Validates that adapt_external_portfolio maps external columns and imputes missing fields."""
    sample_external_data = pd.DataFrame(
        [
            {
                "deal_id": "TRADE_9991",
                "asset": "Bond",
                "product": "Corporate Bond",
                "borrower": "Counterparty_099",
                "industry": "Energy",
                "nation": "US",
                "ccy": "USD",
                "credit_rating": "BBB",
                "amount": 25_000_000.0,
                "tenor": 5.0,
            },
            {
                "deal_id": "TRADE_9992",
                "asset": "Loan",
                "product": "Term Loan",
                "borrower": "Counterparty_088",
                "industry": "Technology",
                "nation": "Germany",
                "ccy": "EUR",
                "credit_rating": "A",
                "amount": 15_000_000.0,
                "tenor": 3.0,
            },
        ]
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as tmp:
        sample_external_data.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        adapted_df = adapt_external_portfolio(tmp_path)
        assert len(adapted_df) == 2
        assert "position_id" in adapted_df.columns
        assert adapted_df.iloc[0]["position_id"] == "TRADE_9991"
        assert adapted_df.iloc[0]["pd_1y"] == 0.0050  # BBB base PD
        assert adapted_df.iloc[0]["lgd"] == 0.50  # Corporate Bond base LGD
        assert adapted_df.iloc[1]["position_id"] == "TRADE_9992"
        assert adapted_df.iloc[1]["pd_1y"] == 0.0010  # A base PD
        assert adapted_df.iloc[1]["spread_duration"] > 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_validation_errors_on_corrupt_portfolios():
    """Validates that validate_portfolio raises clear ValueErrors on corrupt data."""
    valid_df = load_portfolio(PORTFOLIO_PATH).copy()

    # Negative notional
    bad_notional_df = valid_df.copy()
    bad_notional_df.loc[0, "notional"] = -1000.0
    with pytest.raises(ValueError, match="negative notionals"):
        validate_portfolio(bad_notional_df)

    # Invalid rating
    bad_rating_df = valid_df.copy()
    bad_rating_df.loc[0, "rating"] = "INVALID_RATING"
    with pytest.raises(ValueError, match="Invalid credit ratings"):
        validate_portfolio(bad_rating_df)

    # Invalid asset class
    bad_class_df = valid_df.copy()
    bad_class_df.loc[0, "asset_class"] = "Crypto"
    with pytest.raises(ValueError, match="Invalid asset classes"):
        validate_portfolio(bad_class_df)
