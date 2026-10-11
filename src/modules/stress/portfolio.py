"""Wholesale Banking Portfolio Schema, Loader, Validation, and Analytics.

Defines the multi-asset synthetic wholesale banking book schema spanning
Loans, Bonds, Derivatives, and Equity with regulatory credit and market risk metrics
(PD, LGD, EAD, Duration, Convexity, DV01, CS01).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema Constants & Validation Enums
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = [
    "position_id",
    "asset_class",
    "instrument_type",
    "counterparty",
    "sector",
    "country",
    "region",
    "currency",
    "rating",
    "notional",
    "market_value",
    "maturity_years",
    "mod_duration",
    "convexity",
    "spread_duration",
    "pd_1y",
    "lgd",
    "ead",
    "dv01",
    "cs01",
    "fx_ccy",
    "beta",
    "direction",
]

VALID_ASSET_CLASSES: Set[str] = {"Loan", "Bond", "Derivative", "Equity"}

VALID_INSTRUMENT_TYPES: Set[str] = {
    "Term Loan",
    "Revolver",
    "Trade Finance",
    "Corporate Bond",
    "Sovereign Bond",
    "IRS",
    "FX Forward",
    "CDS",
    "Equity Holding",
}

VALID_RATINGS: Set[str] = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"}

# Regulatory Standard 1-Year Probability of Default (PD) Table
RATING_PD_TABLE: Dict[str, float] = {
    "AAA": 0.0001,  # 0.01%
    "AA": 0.0005,  # 0.05%
    "A": 0.0010,  # 0.10%
    "BBB": 0.0050,  # 0.50%
    "BB": 0.0200,  # 2.00%
    "B": 0.0650,  # 6.50%
    "CCC": 0.2000,  # 20.00%
}

# Standard Basel / Seniority Loss Given Default (LGD) Estimates
INSTRUMENT_LGD_TABLE: Dict[str, float] = {
    "Sovereign Bond": 0.25,
    "Trade Finance": 0.30,
    "Term Loan": 0.40,
    "Revolver": 0.45,
    "Corporate Bond": 0.50,
    "CDS": 0.55,
    "IRS": 0.55,
    "FX Forward": 0.55,
    "Equity Holding": 1.00,
}


# ---------------------------------------------------------------------------
# Portfolio Loading & Validation
# ---------------------------------------------------------------------------


def validate_portfolio(df: pd.DataFrame) -> None:
    """Validates that a DataFrame conforms strictly to the RiskPulse portfolio schema."""
    # 1. Required Columns Check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Portfolio missing required columns: {missing_cols}")

    if len(df) == 0:
        raise ValueError("Portfolio contains zero positions.")

    # 2. No Negative Notionals
    if (df["notional"] < 0).any():
        bad_notionals = df[df["notional"] < 0]["position_id"].tolist()
        raise ValueError(f"Positions with negative notionals detected: {bad_notionals}")

    # 3. Valid Ratings
    invalid_ratings = set(df["rating"].dropna().unique()) - VALID_RATINGS
    if invalid_ratings:
        raise ValueError(f"Invalid credit ratings found: {invalid_ratings}. Must be in {VALID_RATINGS}")

    # 4. Valid Asset Classes
    invalid_classes = set(df["asset_class"].unique()) - VALID_ASSET_CLASSES
    if invalid_classes:
        raise ValueError(f"Invalid asset classes found: {invalid_classes}. Must be in {VALID_ASSET_CLASSES}")

    # 5. Valid Instrument Types
    invalid_instruments = set(df["instrument_type"].unique()) - VALID_INSTRUMENT_TYPES
    if invalid_instruments:
        raise ValueError(f"Invalid instrument types found: {invalid_instruments}. Must be in {VALID_INSTRUMENT_TYPES}")

    # 6. Valid Directions
    invalid_dirs = set(df["direction"].unique()) - {1, -1, 1.0, -1.0}
    if invalid_dirs:
        raise ValueError(f"Invalid directions found: {invalid_dirs}. Must be +1 or -1.")

    # 7. PD & LGD Bounds
    if ((df["pd_1y"] < 0) | (df["pd_1y"] > 1.0)).any():
        raise ValueError("pd_1y values must be between 0.0 and 1.0.")
    if ((df["lgd"] < 0) | (df["lgd"] > 1.0)).any():
        raise ValueError("lgd values must be between 0.0 and 1.0.")


def load_portfolio(path: Path | str) -> pd.DataFrame:
    """Loads and validates the wholesale banking portfolio CSV."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Portfolio file not found at: {file_path}")

    df = pd.read_csv(file_path)
    validate_portfolio(df)
    return df


# ---------------------------------------------------------------------------
# Portfolio Summary Analytics
# ---------------------------------------------------------------------------


def portfolio_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes key totals, risk sensitivities, and exposure breakdowns for the portfolio."""
    validate_portfolio(df)

    total_mv = float(df["market_value"].sum())
    total_notional = float(df["notional"].sum())
    total_ead = float(df["ead"].sum())

    # Weighted Average PD and LGD (weighted by EAD)
    ead_weights = df["ead"].values
    if ead_weights.sum() > 0:
        weighted_pd = float(np.average(df["pd_1y"].values, weights=ead_weights))
        weighted_lgd = float(np.average(df["lgd"].values, weights=ead_weights))
    else:
        weighted_pd = float(df["pd_1y"].mean())
        weighted_lgd = float(df["lgd"].mean())

    total_dv01 = float(df["dv01"].sum())
    total_cs01 = float(df["cs01"].sum())

    # Asset Class Breakdown
    by_asset_class = []
    for ac, group in df.groupby("asset_class"):
        mv = float(group["market_value"].sum())
        notl = float(group["notional"].sum())
        ead_val = float(group["ead"].sum())
        by_asset_class.append(
            {
                "asset_class": ac,
                "count": int(len(group)),
                "market_value": round(mv, 2),
                "pct_market_value": round(mv / max(total_mv, 1e-6) * 100, 2),
                "notional": round(notl, 2),
                "ead": round(ead_val, 2),
            }
        )
    by_asset_class = sorted(by_asset_class, key=lambda x: x["market_value"], reverse=True)

    # Sector Breakdown
    by_sector = []
    for sec, group in df.groupby("sector"):
        mv = float(group["market_value"].sum())
        by_sector.append(
            {
                "sector": sec,
                "count": int(len(group)),
                "market_value": round(mv, 2),
                "pct_market_value": round(mv / max(total_mv, 1e-6) * 100, 2),
            }
        )
    by_sector = sorted(by_sector, key=lambda x: x["market_value"], reverse=True)

    # Country Breakdown
    by_country = []
    for ctry, group in df.groupby("country"):
        mv = float(group["market_value"].sum())
        by_country.append(
            {
                "country": ctry,
                "count": int(len(group)),
                "market_value": round(mv, 2),
                "pct_market_value": round(mv / max(total_mv, 1e-6) * 100, 2),
            }
        )
    by_country = sorted(by_country, key=lambda x: x["market_value"], reverse=True)

    # Rating Breakdown
    rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    by_rating = []
    for r in rating_order:
        group = df[df["rating"] == r]
        if len(group) > 0:
            mv = float(group["market_value"].sum())
            by_rating.append(
                {
                    "rating": r,
                    "count": int(len(group)),
                    "market_value": round(mv, 2),
                    "pct_market_value": round(mv / max(total_mv, 1e-6) * 100, 2),
                    "base_pd": RATING_PD_TABLE.get(r, 0.0),
                }
            )

    return {
        "position_count": len(df),
        "total_notional": round(total_notional, 2),
        "total_market_value": round(total_mv, 2),
        "total_ead": round(total_ead, 2),
        "weighted_avg_pd": round(weighted_pd, 5),
        "weighted_avg_lgd": round(weighted_lgd, 4),
        "total_dv01": round(total_dv01, 2),
        "total_cs01": round(total_cs01, 2),
        "by_asset_class": by_asset_class,
        "by_sector": by_sector,
        "by_country": by_country,
        "by_rating": by_rating,
    }
