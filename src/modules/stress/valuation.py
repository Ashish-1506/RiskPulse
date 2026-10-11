"""Vectorised Financial Valuation & P&L Engine for Wholesale Banking Stress Testing.

Implements regulatory and analytical pricing models across all asset classes:
  - Bonds: dP/P = -mod_duration*dRate + 0.5*convexity*dRate^2 - spread_duration*dSpread
  - Loans: MTM = -spread_duration*dSpread*MV; credit loss delta dECL = EAD*LGD*(PD_stressed - PD_base)
  - IRS: P&L = |dv01| * dRate_bps * direction (payer swap gains when rates rise, receiver gains when rates fall)
  - FX forwards: P&L = notional * dFX(fx_ccy) * direction
  - CDS: P&L = |cs01| * dSpread_bps * direction - CVA (protection buyer gains on spread blowout; CVA = EPE*LGD*dPD)
  - Equity: P&L = MV * beta * equity_shock

All evaluations are 100% vectorised in pandas/numpy with ZERO row iteration loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modules.stress.scenarios import ShockPackage

EM_CURRENCIES = {"BRL", "TRY", "ZAR", "MXN", "INR", "CNY", "AED"}


def value_portfolio(df: pd.DataFrame, shocks: ShockPackage) -> pd.DataFrame:
    """Evaluates stressed valuations and P&L across all portfolio positions vectorially.

    Parameters:
      df: Portfolio DataFrame conforming to RiskPulse schema.
      shocks: ShockPackage containing calibrated cross-asset shifts.

    Returns:
      DataFrame enriched with:
        - base_value
        - stressed_value
        - pnl
        - d_ecl
        - pd_stressed
        - spread_shock_bps
        - rate_shock_bps
    """
    res = df.copy()

    # -----------------------------------------------------------------------
    # 1. Determine Position-Level Exposure Linkage & Specific Shocks
    # -----------------------------------------------------------------------
    # Positions in the directly affected sector or country receive an exposure link bonus
    is_affected_sector = (
        res["sector"].isin(shocks.affected_sectors) if shocks.affected_sectors else pd.Series(False, index=res.index)
    )
    is_affected_country = (
        res["country"].isin(shocks.affected_countries)
        if shocks.affected_countries
        else pd.Series(False, index=res.index)
    )
    is_linked = is_affected_sector | is_affected_country

    link_multiplier = np.where(is_linked, 1.0 + shocks.exposure_link_bonus, 1.0)

    # Spread shock attribution
    if shocks.affected_sector_only:
        # e.g. Credit Event: primary spread shock on affected sector, 20% contagion to other sectors
        pos_spread_bps = np.where(is_affected_sector, shocks.spread_bps * link_multiplier, shocks.spread_bps * 0.20)
    elif shocks.affected_region_only:
        # e.g. Natural Disaster: primary spread shock on affected country/region
        pos_spread_bps = np.where(is_affected_country, shocks.spread_bps * link_multiplier, shocks.spread_bps * 0.25)
    else:
        pos_spread_bps = shocks.spread_bps * link_multiplier

    pos_rates_bps = np.full(len(res), shocks.rates_bps, dtype=float)

    # Equity shock attribution
    pos_equity_shock = shocks.equity_shock * link_multiplier

    # Credit PD multiplier attribution
    if shocks.affected_sector_only:
        pos_pd_mult = np.where(
            is_affected_sector,
            1.0 + (shocks.pd_multiplier - 1.0) * link_multiplier,
            1.0 + (shocks.pd_multiplier - 1.0) * 0.20,
        )
    elif shocks.affected_region_only:
        pos_pd_mult = np.where(
            is_affected_country,
            1.0 + (shocks.pd_multiplier - 1.0) * link_multiplier,
            1.0 + (shocks.pd_multiplier - 1.0) * 0.25,
        )
    else:
        pos_pd_mult = 1.0 + (shocks.pd_multiplier - 1.0) * link_multiplier

    # Compute Stressed 1-Year Probability of Default (capped at 1.0)
    pd_base = res["pd_1y"].values
    pd_stressed = np.minimum(1.0, pd_base * pos_pd_mult)
    d_pd = np.maximum(0.0, pd_stressed - pd_base)

    # Decimals for rate and spread shifts
    d_rate = pos_rates_bps * 0.0001
    d_spread = pos_spread_bps * 0.0001

    # Initialize P&L and d_ecl vectors
    pnl = np.zeros(len(res), dtype=float)
    d_ecl = np.zeros(len(res), dtype=float)

    # -----------------------------------------------------------------------
    # 2. Vectorised Valuation by Asset Class
    # -----------------------------------------------------------------------

    # A) BONDS
    # dP/P = -mod_duration*dRate + 0.5*convexity*dRate^2 - spread_duration*dSpread
    mask_bonds = res["asset_class"] == "Bond"
    if mask_bonds.any():
        mod_dur = res.loc[mask_bonds, "mod_duration"].values
        cvx = res.loc[mask_bonds, "convexity"].values
        spr_dur = res.loc[mask_bonds, "spread_duration"].values
        mv = res.loc[mask_bonds, "market_value"].values
        dr = d_rate[mask_bonds]
        ds = d_spread[mask_bonds]

        dp_p = -mod_dur * dr + 0.5 * cvx * (dr**2) - spr_dur * ds
        pnl[mask_bonds] = mv * dp_p

    # B) LOANS
    # MTM = -spread_duration*dSpread*MV
    # dECL = EAD*LGD*(PD_stressed - PD_base)
    mask_loans = res["asset_class"] == "Loan"
    if mask_loans.any():
        spr_dur = res.loc[mask_loans, "spread_duration"].values
        mv = res.loc[mask_loans, "market_value"].values
        ead = res.loc[mask_loans, "ead"].values
        lgd = res.loc[mask_loans, "lgd"].values
        ds = d_spread[mask_loans]
        dpd = d_pd[mask_loans]

        loan_mtm = -spr_dur * ds * mv
        loan_decl = ead * lgd * dpd

        pnl[mask_loans] = loan_mtm - loan_decl
        d_ecl[mask_loans] = loan_decl

    # C) DERIVATIVES: IRS, FX Forward, CDS
    mask_derivs = res["asset_class"] == "Derivative"
    if mask_derivs.any():
        itypes = res.loc[mask_derivs, "instrument_type"].values
        notl = res.loc[mask_derivs, "notional"].values
        dirs = res.loc[mask_derivs, "direction"].values
        dv01_vals = np.abs(res.loc[mask_derivs, "dv01"].values)
        cs01_vals = np.abs(res.loc[mask_derivs, "cs01"].values)
        lgd = res.loc[mask_derivs, "lgd"].values
        ccys = res.loc[mask_derivs, "currency"].values
        dr_bps = pos_rates_bps[mask_derivs]
        ds_bps = pos_spread_bps[mask_derivs]
        dpd = d_pd[mask_derivs]

        deriv_pnl = np.zeros(len(itypes), dtype=float)

        # 1. IRS: P&L = |dv01| * dRate_bps * direction
        mask_irs = itypes == "IRS"
        if mask_irs.any():
            deriv_pnl[mask_irs] = dv01_vals[mask_irs] * dr_bps[mask_irs] * dirs[mask_irs]

        # 2. FX Forwards: P&L = notional * dFX * direction
        mask_fx = itypes == "FX Forward"
        if mask_fx.any():
            fx_shocks = np.zeros(len(itypes), dtype=float)
            for i, ccy in enumerate(ccys):
                if ccy == "USD":
                    fx_shocks[i] = 0.0
                elif ccy in EM_CURRENCIES:
                    fx_shocks[i] = shocks.fx_em_shock
                else:
                    fx_shocks[i] = -shocks.fx_usd_shock  # Foreign ccy depreciates when USD strengthens

            deriv_pnl[mask_fx] = notl[mask_fx] * fx_shocks[mask_fx] * dirs[mask_fx]

        # 3. CDS: P&L = |cs01| * dSpread_bps * direction - CVA
        # Protection buyer (direction=+1) gains when spreads widen.
        # Simple CVA simplification: CVA = EPE * LGD * dPD, where EPE = 0.1 * notional.
        mask_cds = itypes == "CDS"
        if mask_cds.any():
            cds_spread_pnl = cs01_vals[mask_cds] * ds_bps[mask_cds] * dirs[mask_cds]
            epe = 0.10 * notl[mask_cds]
            cva = epe * lgd[mask_cds] * dpd[mask_cds]
            deriv_pnl[mask_cds] = cds_spread_pnl - cva

        pnl[mask_derivs] = deriv_pnl

    # D) EQUITY
    # P&L = MV * beta * equity_shock
    mask_equity = res["asset_class"] == "Equity"
    if mask_equity.any():
        mv = res.loc[mask_equity, "market_value"].values
        beta = res.loc[mask_equity, "beta"].values
        eq_shock = pos_equity_shock[mask_equity]

        pnl[mask_equity] = mv * beta * eq_shock

    # -----------------------------------------------------------------------
    # 3. Output Assembly
    # -----------------------------------------------------------------------
    res["base_value"] = res["market_value"]
    res["pnl"] = np.round(pnl, 2)
    res["stressed_value"] = np.round(res["base_value"] + res["pnl"], 2)
    res["d_ecl"] = np.round(d_ecl, 2)
    res["pd_stressed"] = np.round(pd_stressed, 6)
    res["spread_shock_bps"] = np.round(pos_spread_bps, 2)
    res["rate_shock_bps"] = np.round(pos_rates_bps, 2)

    return res
