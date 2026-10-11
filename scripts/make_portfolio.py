"""Generates synthetic wholesale banking portfolio data/portfolio.csv.

Supports:
  1) Fully synthetic deterministic generation (seed 42) calibrated to:
     - $5.0 Billion total book market value
     - Asset Class composition: Loans ~45%, Bonds ~30%, Derivatives ~18%, Equity ~7%
     - Credit rating PD lookup (AAA 0.01% ... CCC 20.0%)
     - Instrument-level LGD (25% - 60%)
     - Consistent maturity, duration, convexity, DV01, and CS01
     - Real-world concentrations (Energy in UAE, Real Estate in China, Tech in US)
  2) Adapter mode (--input path):
     - Ingests external hackathon transaction CSV, maps columns, and infers risk parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.modules.stress.portfolio import (
    INSTRUMENT_LGD_TABLE,
    RATING_PD_TABLE,
    portfolio_summary,
    validate_portfolio,
)

COUNTRY_REGION_MAP: Dict[str, str] = {
    "US": "North America",
    "UK": "Europe",
    "Germany": "Europe",
    "India": "Asia-Pacific",
    "China": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "Brazil": "Latin America",
    "Mexico": "Latin America",
    "UAE": "Middle East",
    "Turkey": "EMEA",
    "South Africa": "Africa",
}

COUNTRY_CURRENCY_MAP: Dict[str, str] = {
    "US": "USD",
    "UK": "GBP",
    "Germany": "EUR",
    "India": "INR",
    "China": "CNY",
    "Japan": "JPY",
    "Brazil": "BRL",
    "Mexico": "MXN",
    "UAE": "AED",
    "Turkey": "TRY",
    "South Africa": "ZAR",
}


def generate_synthetic_portfolio(seed: int = 42, target_total_mv: float = 5_000_000_000.0) -> pd.DataFrame:
    """Generates a realistic, calibrated synthetic wholesale banking portfolio.

    Target composition by Market Value:
      - Loans: ~45% ($2,250M)
      - Bonds: ~30% ($1,500M)
      - Derivatives: ~18% ($900M)
      - Equity: ~7% ($350M)
      Total Book: ~$5.0 Billion USD.
    """
    rng = np.random.default_rng(seed)

    positions: List[Dict[str, Any]] = []
    pos_counter = 1

    # Pool of 150 synthetic counterparties
    counterparties = [f"Counterparty_{i:03d}" for i in range(1, 151)]

    # Target allocations
    target_loans_mv = target_total_mv * 0.45  # $2,250M
    target_bonds_mv = target_total_mv * 0.30  # $1,500M
    target_derivs_mv = target_total_mv * 0.18  # $900M
    target_equity_mv = target_total_mv * 0.07  # $350M

    # -----------------------------------------------------------------------
    # 1. LOANS (~45% of book, ~180-220 positions)
    # -----------------------------------------------------------------------
    curr_loans_mv = 0.0
    loan_types = ["Term Loan", "Revolver", "Trade Finance"]
    loan_type_weights = [0.60, 0.30, 0.10]
    sectors = [
        "Energy",
        "Financials",
        "Technology",
        "Healthcare",
        "Industrials",
        "Utilities",
        "Consumer",
        "Real Estate",
    ]
    countries = list(COUNTRY_REGION_MAP.keys())

    while curr_loans_mv < target_loans_mv:
        pos_id = f"POS_{pos_counter:04d}"
        pos_counter += 1

        itype = rng.choice(loan_types, p=loan_type_weights)
        cp = rng.choice(counterparties[:80])  # primary borrowing corporate clients

        # Incorporate concentration: 15% probability of UAE Energy or China Real Estate
        r_conc = rng.random()
        if r_conc < 0.12:
            country = "UAE"
            sector = "Energy"
            rating = rng.choice(["BBB", "BB", "A"], p=[0.5, 0.3, 0.2])
        elif r_conc < 0.22:
            country = "China"
            sector = "Real Estate"
            rating = rng.choice(["BB", "B", "CCC"], p=[0.4, 0.4, 0.2])
        else:
            country = rng.choice(countries)
            sector = rng.choice(sectors)
            rating = rng.choice(
                ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"], p=[0.05, 0.12, 0.28, 0.32, 0.15, 0.06, 0.02]
            )

        region = COUNTRY_REGION_MAP[country]
        currency = "USD" if rng.random() < 0.70 else COUNTRY_CURRENCY_MAP[country]

        if itype == "Trade Finance":
            maturity = round(float(rng.uniform(0.25, 1.0)), 2)
            notional = round(float(rng.uniform(3_000_000, 15_000_000)), -4)
        elif itype == "Revolver":
            maturity = round(float(rng.uniform(1.0, 5.0)), 2)
            notional = round(float(rng.uniform(8_000_000, 30_000_000)), -4)
        else:  # Term Loan
            maturity = round(float(rng.uniform(2.0, 7.0)), 2)
            notional = round(float(rng.uniform(10_000_000, 45_000_000)), -4)

        # Market value for loans is typically near par, adjusted slightly for credit quality
        mv_factor = rng.uniform(0.96, 1.01) if rating not in ("B", "CCC") else rng.uniform(0.85, 0.95)
        market_value = round(notional * mv_factor, 2)

        # If adding this would overshoot target significantly, clamp it
        if curr_loans_mv + market_value > target_loans_mv + 20_000_000 and curr_loans_mv > target_loans_mv * 0.95:
            market_value = round(target_loans_mv - curr_loans_mv, 2)
            notional = market_value

        curr_loans_mv += market_value

        # Floating rate loan IR duration is short (3-month reset), but spread duration matches tenor
        mod_duration = round(float(rng.uniform(0.20, 0.30)), 3)
        spread_duration = round(float(maturity * rng.uniform(0.80, 0.95)), 2)
        convexity = 0.01

        pd_1y = RATING_PD_TABLE[rating]
        lgd = INSTRUMENT_LGD_TABLE[itype]

        # EAD: For term loan MV, for revolver drawn + undrawn CCF (~0.75 of notional)
        if itype == "Revolver":
            ead = round(notional * 0.75, 2)
        else:
            ead = market_value

        direction = 1
        dv01 = round(notional * mod_duration * 0.0001 * direction, 2)
        cs01 = round(notional * spread_duration * 0.0001 * direction, 2)
        beta = round(float(rng.uniform(0.8, 1.3)), 2)

        positions.append(
            {
                "position_id": pos_id,
                "asset_class": "Loan",
                "instrument_type": itype,
                "counterparty": cp,
                "sector": sector,
                "country": country,
                "region": region,
                "currency": currency,
                "rating": rating,
                "notional": notional,
                "market_value": market_value,
                "maturity_years": maturity,
                "mod_duration": mod_duration,
                "convexity": convexity,
                "spread_duration": spread_duration,
                "pd_1y": pd_1y,
                "lgd": lgd,
                "ead": ead,
                "dv01": dv01,
                "cs01": cs01,
                "fx_ccy": currency,
                "beta": beta,
                "direction": direction,
            }
        )

    # -----------------------------------------------------------------------
    # 2. BONDS (~30% of book, ~100-140 positions)
    # -----------------------------------------------------------------------
    curr_bonds_mv = 0.0
    bond_types = ["Corporate Bond", "Sovereign Bond"]
    bond_type_weights = [0.70, 0.30]

    while curr_bonds_mv < target_bonds_mv:
        pos_id = f"POS_{pos_counter:04d}"
        pos_counter += 1

        itype = rng.choice(bond_types, p=bond_type_weights)
        if itype == "Sovereign Bond":
            country = rng.choice(countries)
            cp = f"Sovereign_{country.replace(' ', '_')}"
            sector = "Financials"  # Government / Public Finance
            rating = (
                "AAA"
                if country in ("US", "Germany")
                else "AA"
                if country in ("UK", "Japan")
                else "BBB"
                if country in ("India", "UAE", "Mexico")
                else "BB"
            )
        else:
            cp = rng.choice(counterparties[40:120])
            r_conc = rng.random()
            if r_conc < 0.15:
                country = "UAE"
                sector = "Energy"
                rating = rng.choice(["A", "BBB", "BB"])
            elif r_conc < 0.25:
                country = "China"
                sector = "Real Estate"
                rating = rng.choice(["BB", "B", "CCC"])
            elif r_conc < 0.40:
                country = "US"
                sector = "Technology"
                rating = rng.choice(["AA", "A", "BBB"])
            else:
                country = rng.choice(countries)
                sector = rng.choice(sectors)
                rating = rng.choice(
                    ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"], p=[0.08, 0.18, 0.30, 0.26, 0.12, 0.04, 0.02]
                )

        region = COUNTRY_REGION_MAP[country]
        currency = "USD" if rng.random() < 0.75 else COUNTRY_CURRENCY_MAP[country]

        maturity = round(float(rng.uniform(1.5, 12.0)), 2)
        notional = round(float(rng.uniform(8_000_000, 35_000_000)), -4)

        mv_factor = rng.uniform(0.95, 1.04) if rating not in ("B", "CCC") else rng.uniform(0.80, 0.92)
        market_value = round(notional * mv_factor, 2)

        if curr_bonds_mv + market_value > target_bonds_mv + 15_000_000 and curr_bonds_mv > target_bonds_mv * 0.95:
            market_value = round(target_bonds_mv - curr_bonds_mv, 2)
            notional = market_value

        curr_bonds_mv += market_value

        # Fixed rate debt duration
        mod_duration = round(float(maturity * rng.uniform(0.82, 0.92)), 2)
        convexity = round(float(0.12 * (mod_duration**2)), 3)
        spread_duration = mod_duration if itype == "Corporate Bond" else round(mod_duration * 0.6, 2)

        pd_1y = RATING_PD_TABLE[rating]
        lgd = INSTRUMENT_LGD_TABLE[itype]
        ead = market_value

        direction = 1
        dv01 = round(notional * mod_duration * 0.0001 * direction, 2)
        cs01 = round(notional * spread_duration * 0.0001 * direction, 2)
        beta = 0.0 if itype == "Sovereign Bond" else round(float(rng.uniform(0.6, 1.4)), 2)

        positions.append(
            {
                "position_id": pos_id,
                "asset_class": "Bond",
                "instrument_type": itype,
                "counterparty": cp,
                "sector": sector,
                "country": country,
                "region": region,
                "currency": currency,
                "rating": rating,
                "notional": notional,
                "market_value": market_value,
                "maturity_years": maturity,
                "mod_duration": mod_duration,
                "convexity": convexity,
                "spread_duration": spread_duration,
                "pd_1y": pd_1y,
                "lgd": lgd,
                "ead": ead,
                "dv01": dv01,
                "cs01": cs01,
                "fx_ccy": currency,
                "beta": beta,
                "direction": direction,
            }
        )

    # -----------------------------------------------------------------------
    # 3. DERIVATIVES (~18% of book, ~60-80 positions, large notionals, MtM)
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # 3. DERIVATIVES (~18% of book, ~70-90 positions, large notionals, MtM)
    # -----------------------------------------------------------------------
    curr_derivs_mv = 0.0
    deriv_types = ["IRS", "FX Forward", "CDS"]
    deriv_type_weights = [0.55, 0.25, 0.20]

    while True:
        rem_derivs = target_derivs_mv - curr_derivs_mv
        if rem_derivs <= 2_000_000:
            break

        pos_id = f"POS_{pos_counter:04d}"
        pos_counter += 1

        itype = rng.choice(deriv_types, p=deriv_type_weights)
        cp = rng.choice(counterparties[70:150])
        sector = rng.choice(["Financials", "Energy", "Technology", "Industrials"])
        country = rng.choice(["US", "UK", "Germany", "Japan", "UAE", "China"])
        region = COUNTRY_REGION_MAP[country]
        rating = rng.choice(["AA", "A", "BBB"], p=[0.25, 0.50, 0.25])
        currency = COUNTRY_CURRENCY_MAP[country]

        direction = int(rng.choice([1, -1], p=[0.55, 0.45]))
        notional = round(float(rng.uniform(40_000_000, 150_000_000)), -5)

        if itype == "IRS":
            maturity = round(float(rng.uniform(2.0, 10.0)), 2)
            mod_duration = round(float(maturity * rng.uniform(0.65, 0.78)), 2)
            convexity = round(float(0.10 * (mod_duration**2)), 3)
            spread_duration = 0.0
            # Realistic MtM % of notional
            mtm_pct = rng.uniform(0.04, 0.14)
            market_value = round(notional * mtm_pct, 2)
        elif itype == "FX Forward":
            maturity = round(float(rng.uniform(0.25, 2.0)), 2)
            mod_duration = round(float(maturity * 0.90), 2)
            convexity = 0.01
            spread_duration = 0.0
            mtm_pct = rng.uniform(0.03, 0.10)
            market_value = round(notional * mtm_pct, 2)
        else:  # CDS
            maturity = round(float(rng.uniform(1.0, 5.0)), 2)
            mod_duration = 0.10
            convexity = 0.01
            spread_duration = round(float(maturity * 0.95), 2)
            mtm_pct = rng.uniform(0.04, 0.12)
            market_value = round(notional * mtm_pct, 2)

        # Allow ~15% of derivatives to have negative MtM (liability positions)
        if rng.random() < 0.15 and rem_derivs > 50_000_000:
            market_value = -abs(round(market_value * 0.5, 2))

        if market_value > rem_derivs:
            market_value = round(rem_derivs, 2)

        curr_derivs_mv += market_value

        pd_1y = RATING_PD_TABLE[rating]
        lgd = INSTRUMENT_LGD_TABLE[itype]
        ead = round(max(0.0, market_value) + (0.05 * notional), 2)

        dv01 = round(notional * mod_duration * 0.0001 * direction, 2)
        cs01 = round(notional * spread_duration * 0.0001 * direction, 2)
        beta = 0.0

        positions.append(
            {
                "position_id": pos_id,
                "asset_class": "Derivative",
                "instrument_type": itype,
                "counterparty": cp,
                "sector": sector,
                "country": country,
                "region": region,
                "currency": currency,
                "rating": rating,
                "notional": notional,
                "market_value": market_value,
                "maturity_years": maturity,
                "mod_duration": mod_duration,
                "convexity": convexity,
                "spread_duration": spread_duration,
                "pd_1y": pd_1y,
                "lgd": lgd,
                "ead": ead,
                "dv01": dv01,
                "cs01": cs01,
                "fx_ccy": currency,
                "beta": beta,
                "direction": direction,
            }
        )

    # -----------------------------------------------------------------------
    # 4. EQUITY (~7% of book, ~30-40 positions)
    # -----------------------------------------------------------------------
    curr_equity_mv = 0.0

    while True:
        rem_equity = target_equity_mv - curr_equity_mv
        if rem_equity <= 1_000_000:
            break

        pos_id = f"POS_{pos_counter:04d}"
        pos_counter += 1

        cp = rng.choice(counterparties[20:100])
        r_conc = rng.random()
        if r_conc < 0.25:
            country = "US"
            sector = "Technology"
            rating = "A"
            beta = round(float(rng.uniform(1.2, 1.6)), 2)
        elif r_conc < 0.40:
            country = "UAE"
            sector = "Energy"
            rating = "BBB"
            beta = round(float(rng.uniform(0.9, 1.3)), 2)
        else:
            country = rng.choice(countries)
            sector = rng.choice(sectors)
            rating = rng.choice(["AA", "A", "BBB", "BB"])
            beta = round(float(rng.uniform(0.7, 1.4)), 2)

        region = COUNTRY_REGION_MAP[country]
        currency = COUNTRY_CURRENCY_MAP[country]

        market_value = round(float(rng.uniform(6_000_000, 16_000_000)), -4)
        if market_value > rem_equity:
            market_value = round(rem_equity, 2)

        curr_equity_mv += market_value
        notional = market_value

        maturity = 0.0
        mod_duration = 0.0
        convexity = 0.0
        spread_duration = 0.0

        pd_1y = RATING_PD_TABLE[rating]
        lgd = INSTRUMENT_LGD_TABLE["Equity Holding"]
        ead = market_value

        direction = 1
        dv01 = 0.0
        cs01 = 0.0

        positions.append(
            {
                "position_id": pos_id,
                "asset_class": "Equity",
                "instrument_type": "Equity Holding",
                "counterparty": cp,
                "sector": sector,
                "country": country,
                "region": region,
                "currency": currency,
                "rating": rating,
                "notional": notional,
                "market_value": market_value,
                "maturity_years": maturity,
                "mod_duration": mod_duration,
                "convexity": convexity,
                "spread_duration": spread_duration,
                "pd_1y": pd_1y,
                "lgd": lgd,
                "ead": ead,
                "dv01": dv01,
                "cs01": cs01,
                "fx_ccy": currency,
                "beta": beta,
                "direction": direction,
            }
        )

    df = pd.DataFrame(positions)
    validate_portfolio(df)
    return df


# ---------------------------------------------------------------------------
# External File Adapter (Hackathon Ingestion)
# ---------------------------------------------------------------------------


def adapt_external_portfolio(input_path: Path | str) -> pd.DataFrame:
    """Loads an external transaction/portfolio file, maps columns, and infers missing risk parameters."""
    file_path = Path(input_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if file_path.suffix.lower() in (".xlsx", ".xls"):
        raw_df = pd.read_excel(file_path)
    else:
        raw_df = pd.read_csv(file_path)

    # Normalize column names (strip, lower)
    col_map = {c: c.strip().lower() for c in raw_df.columns}
    rev_col_map = {v: k for k, v in col_map.items()}

    def get_val(row: pd.Series, *aliases: str, default: Any = None) -> Any:
        for alias in aliases:
            if alias in rev_col_map:
                val = row[rev_col_map[alias]]
                if pd.notna(val):
                    return val
        return default

    adapted_rows: List[Dict[str, Any]] = []

    for idx, row in raw_df.iterrows():
        pos_id = str(get_val(row, "position_id", "id", "deal_id", "trade_id", default=f"EXT_{idx + 1:04d}"))
        ac_raw = str(get_val(row, "asset_class", "asset", "class", "type", default="Loan")).capitalize()
        asset_class = ac_raw if ac_raw in ("Loan", "Bond", "Derivative", "Equity") else "Loan"

        itype_raw = str(get_val(row, "instrument_type", "instrument", "product", default="Term Loan"))
        instrument_type = itype_raw if itype_raw in INSTRUMENT_LGD_TABLE else "Term Loan"

        cp = str(
            get_val(row, "counterparty", "client", "borrower", "issuer", default=f"Counterparty_{idx % 100 + 1:03d}")
        )
        sector = str(get_val(row, "sector", "industry", default="Financials"))
        country = str(get_val(row, "country", "nation", default="US"))
        region = COUNTRY_REGION_MAP.get(country, "North America")
        currency = str(get_val(row, "currency", "ccy", default="USD"))

        rating_raw = str(get_val(row, "rating", "credit_rating", default="BBB")).upper()
        rating = rating_raw if rating_raw in RATING_PD_TABLE else "BBB"

        notional = float(abs(get_val(row, "notional", "amount", "principal", "par", default=10_000_000.0)))
        mv = float(get_val(row, "market_value", "mv", "value", default=notional))

        maturity = float(abs(get_val(row, "maturity_years", "maturity", "tenor", default=3.0)))

        # Inferred duration & convexity
        if asset_class == "Bond":
            mod_dur = float(get_val(row, "mod_duration", "duration", default=round(maturity * 0.85, 2)))
            cvx = float(get_val(row, "convexity", default=round(0.12 * (mod_dur**2), 3)))
            spr_dur = float(get_val(row, "spread_duration", default=mod_dur))
        elif asset_class == "Loan":
            mod_dur = float(get_val(row, "mod_duration", default=0.25))
            cvx = 0.01
            spr_dur = float(get_val(row, "spread_duration", default=round(maturity * 0.85, 2)))
        elif asset_class == "Derivative":
            mod_dur = float(get_val(row, "mod_duration", default=round(maturity * 0.70, 2)))
            cvx = 0.02
            spr_dur = 0.0 if instrument_type != "CDS" else round(maturity * 0.95, 2)
        else:  # Equity
            mod_dur = 0.0
            cvx = 0.0
            spr_dur = 0.0

        pd_1y = float(get_val(row, "pd_1y", "pd", default=RATING_PD_TABLE[rating]))
        lgd = float(get_val(row, "lgd", default=INSTRUMENT_LGD_TABLE.get(instrument_type, 0.40)))
        ead = float(get_val(row, "ead", default=max(mv, notional * 0.05)))

        direction = int(get_val(row, "direction", "dir", default=1))
        direction = 1 if direction >= 0 else -1

        dv01 = float(get_val(row, "dv01", default=round(notional * mod_dur * 0.0001 * direction, 2)))
        cs01 = float(get_val(row, "cs01", default=round(notional * spr_dur * 0.0001 * direction, 2)))
        fx_ccy = str(get_val(row, "fx_ccy", default=currency))
        beta = float(get_val(row, "beta", default=1.0))

        adapted_rows.append(
            {
                "position_id": pos_id,
                "asset_class": asset_class,
                "instrument_type": instrument_type,
                "counterparty": cp,
                "sector": sector,
                "country": country,
                "region": region,
                "currency": currency,
                "rating": rating,
                "notional": notional,
                "market_value": mv,
                "maturity_years": maturity,
                "mod_duration": mod_dur,
                "convexity": cvx,
                "spread_duration": spr_dur,
                "pd_1y": pd_1y,
                "lgd": lgd,
                "ead": ead,
                "dv01": dv01,
                "cs01": cs01,
                "fx_ccy": fx_ccy,
                "beta": beta,
                "direction": direction,
            }
        )

    adapted_df = pd.DataFrame(adapted_rows)
    validate_portfolio(adapted_df)
    return adapted_df


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or adapt wholesale banking portfolio dataset")
    parser.add_argument("--out", type=str, default="data/portfolio.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--input", type=str, default=None, help="Optional external transaction file to adapt")
    args = parser.parse_args()

    out_path = BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        print(f"Adapting external portfolio file from: {args.input}...")
        df = adapt_external_portfolio(args.input)
    else:
        print(f"Generating synthetic $5.0B wholesale banking portfolio (seed={args.seed})...")
        df = generate_synthetic_portfolio(seed=args.seed)

    df.to_csv(out_path, index=False)
    print(f"Portfolio saved successfully to: {out_path} ({len(df)} positions)")

    # Print summary breakdown
    stats = portfolio_summary(df)
    print("\n" + "=" * 65)
    print("           WHOLESALE BANKING PORTFOLIO SUMMARY")
    print("=" * 65)
    print(f"Total Positions:      {stats['position_count']}")
    print(f"Total Market Value:   ${stats['total_market_value']:,.2f}")
    print(f"Total Notional:       ${stats['total_notional']:,.2f}")
    print(f"Total EAD:            ${stats['total_ead']:,.2f}")
    print(f"Weighted Average PD:  {stats['weighted_avg_pd'] * 100:.3f}%")
    print(f"Weighted Average LGD: {stats['weighted_avg_lgd'] * 100:.2f}%")
    print(f"Total Net DV01:       ${stats['total_dv01']:,.2f}/bp")
    print(f"Total Net CS01:       ${stats['total_cs01']:,.2f}/bp")
    print("-" * 65)
    print("ASSET CLASS COMPOSITION:")
    for ac in stats["by_asset_class"]:
        print(
            f"  - {ac['asset_class']:<12}: ${ac['market_value']:>14,.2f} ({ac['pct_market_value']:>5.1f}%) | {ac['count']} positions"
        )
    print("-" * 65)
    print("TOP SECTOR EXPOSURES:")
    for s in stats["by_sector"][:5]:
        print(f"  - {s['sector']:<14}: ${s['market_value']:>14,.2f} ({s['pct_market_value']:>5.1f}%)")
    print("-" * 65)
    print("TOP COUNTRY EXPOSURES:")
    for c in stats["by_country"][:5]:
        print(f"  - {c['country']:<14}: ${c['market_value']:>14,.2f} ({c['pct_market_value']:>5.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
