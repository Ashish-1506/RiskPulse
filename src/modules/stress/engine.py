"""Stress Testing Engine for Strategic Wholesale Banking Portfolio Analysis.

Orchestrates scenario selection, exposure link scaling, vectorised asset valuation,
reverse stress testing (bisection on severity), and risk breakdown reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.api.schemas import RiskSignal
from src.config import get_config, logger
from src.modules.stress.scenarios import (
    ShockPackage,
    build_shocks,
    load_scenarios,
    should_trigger,
)
from src.modules.stress.valuation import value_portfolio


@dataclass
class StressResult:
    """Consolidated stress test result output schema."""

    base_value: float
    stressed_value: float
    total_pnl: float
    pnl_pct: float
    total_d_ecl: float
    breakdowns: Dict[str, List[Dict[str, Any]]]
    top_10_losers: List[Dict[str, Any]]
    scenario_params: Dict[str, Any]
    triggering_signal_id: Optional[str] = None
    headline: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StressEngine:
    """Core wholesale banking portfolio stress testing and reverse stress engine."""

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self.cfg = cfg or get_config()
        self.scenarios = load_scenarios()

    def run(
        self,
        portfolio: pd.DataFrame,
        signal_or_scenario: Union[RiskSignal, ShockPackage, Dict[str, Any]],
        severity_override: Optional[float] = None,
    ) -> Optional[StressResult]:
        """Executes stress test on portfolio for a RiskSignal, ShockPackage, or custom parameters."""
        if len(portfolio) == 0:
            raise ValueError("Cannot run stress test on an empty portfolio.")

        # 1. Resolve ShockPackage
        sig_id = None
        headline = None

        if isinstance(signal_or_scenario, ShockPackage):
            shocks = signal_or_scenario
        elif isinstance(signal_or_scenario, RiskSignal):
            sig_id = signal_or_scenario.signal_id
            headline = signal_or_scenario.headline
            shocks = build_shocks(
                signal=signal_or_scenario,
                scenarios=self.scenarios,
                cfg=self.cfg,
                severity_override=severity_override,
            )
            if shocks is None:
                logger.info(f"Signal '{signal_or_scenario.event_type}' has no configured stress scenario. Skipping.")
                return None
        elif isinstance(signal_or_scenario, dict):
            shocks = build_shocks(custom_shocks=signal_or_scenario, cfg=self.cfg)
        else:
            raise TypeError(f"Unsupported signal_or_scenario type: {type(signal_or_scenario)}")

        if shocks is None:
            return None

        # 2. Vectorised Portfolio Valuation
        valued_df = value_portfolio(portfolio, shocks)

        # 3. Aggregate Portfolio-Level Metrics
        base_mv = float(valued_df["base_value"].sum())
        stressed_mv = float(valued_df["stressed_value"].sum())
        total_pnl = float(valued_df["pnl"].sum())
        pnl_pct = round((total_pnl / max(abs(base_mv), 1e-6)) * 100, 2)
        total_d_ecl = float(valued_df["d_ecl"].sum())

        # 4. Exposure & P&L Breakdowns
        breakdowns: Dict[str, List[Dict[str, Any]]] = {}

        for dim in ["asset_class", "sector", "country", "rating"]:
            dim_records = []
            for group_name, grp in valued_df.groupby(dim):
                grp_base = float(grp["base_value"].sum())
                grp_pnl = float(grp["pnl"].sum())
                grp_pct = round((grp_pnl / max(abs(grp_base), 1e-6)) * 100, 2)
                dim_records.append(
                    {
                        dim: group_name,
                        "count": int(len(grp)),
                        "base_value": round(grp_base, 2),
                        "pnl": round(grp_pnl, 2),
                        "pnl_pct": grp_pct,
                        "d_ecl": round(float(grp["d_ecl"].sum()), 2),
                    }
                )
            # Sort worst losses first
            dim_records.sort(key=lambda x: x["pnl"])
            breakdowns[dim] = dim_records

        # 5. Identify Top 10 Losers
        top_losers_df = valued_df.sort_values(by="pnl", ascending=True).head(10)
        top_10_losers = []
        for _, row in top_losers_df.iterrows():
            top_10_losers.append(
                {
                    "position_id": str(row["position_id"]),
                    "counterparty": str(row["counterparty"]),
                    "asset_class": str(row["asset_class"]),
                    "sector": str(row["sector"]),
                    "country": str(row["country"]),
                    "base_value": round(float(row["base_value"]), 2),
                    "stressed_value": round(float(row["stressed_value"]), 2),
                    "pnl": round(float(row["pnl"]), 2),
                    "d_ecl": round(float(row["d_ecl"]), 2),
                }
            )

        scenario_params = {
            "scenario_name": shocks.scenario_name,
            "event_type": shocks.event_type,
            "severity": shocks.severity,
            "equity_shock_pct": round(shocks.equity_shock * 100, 2),
            "rates_bps": shocks.rates_bps,
            "spread_bps": shocks.spread_bps,
            "fx_em_pct": round(shocks.fx_em_shock * 100, 2),
            "pd_multiplier": shocks.pd_multiplier,
            "affected_sectors": shocks.affected_sectors,
            "affected_countries": shocks.affected_countries,
        }

        return StressResult(
            base_value=round(base_mv, 2),
            stressed_value=round(stressed_mv, 2),
            total_pnl=round(total_pnl, 2),
            pnl_pct=pnl_pct,
            total_d_ecl=round(total_d_ecl, 2),
            breakdowns=breakdowns,
            top_10_losers=top_10_losers,
            scenario_params=scenario_params,
            triggering_signal_id=sig_id,
            headline=headline,
        )

    def run_all_triggered(
        self,
        portfolio: pd.DataFrame,
        signals: List[RiskSignal],
    ) -> List[StressResult]:
        """Filters input signals using should_trigger rule and runs stress tests on each."""
        results: List[StressResult] = []
        for sig in signals:
            if should_trigger(sig, cfg=self.cfg):
                res = self.run(portfolio, sig)
                if res is not None:
                    results.append(res)
        return results

    def reverse_stress(
        self,
        portfolio: pd.DataFrame,
        event_type: str,
        max_loss_pct: float,
        max_iterations: int = 25,
        tolerance: float = 0.01,
    ) -> Dict[str, Any]:
        """Calculates critical severity threshold via bisection at which portfolio loss exceeds max_loss_pct.

        Parameters:
          portfolio: Wholesale portfolio DataFrame.
          event_type: Target event category (e.g. 'Geopolitical', 'Macroeconomic', 'Credit Event').
          max_loss_pct: Permissible loss percentage (e.g. 5.0 for 5% loss).

        Returns:
          Dict containing critical_severity, critical_impact_score, estimated_pnl, and limit_reached.
        """
        if event_type not in self.scenarios:
            raise ValueError(f"Event type '{event_type}' has no configured stress scenario.")

        # Create dummy trigger signal with unit impact to scale
        dummy_signal = RiskSignal(
            signal_id="REVERSE_STRESS_PROBE",
            timestamp=pd.Timestamp.now().to_pydatetime(),
            source="reverse_stress",
            source_type="news",
            headline=f"Reverse stress search for {event_type}",
            tickers=[],
            countries=[],
            sectors=[],
            sentiment_score=-1.0,
            sentiment_label="NEGATIVE",
            event_type=event_type,
            event_confidence=1.0,
            impact_score=10.0,
            impact_components={},
            impact_rationale="Reverse stress testing",
            model_version="v1",
        )

        low = 0.0
        high = 1.5

        # Check if high is sufficient
        high_res = self.run(portfolio, dummy_signal, severity_override=high)
        if high_res and abs(high_res.pnl_pct) < max_loss_pct:
            # Expand search range if even 1.5x severity does not exceed max_loss_pct
            high = 3.0

        best_severity = high
        best_res = None

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            res = self.run(portfolio, dummy_signal, severity_override=mid)
            if res is None:
                break

            current_loss_pct = abs(res.pnl_pct)
            best_res = res
            best_severity = mid

            if abs(current_loss_pct - max_loss_pct) < tolerance:
                break

            if current_loss_pct >= max_loss_pct:
                high = mid  # loss too high, search lower severity
            else:
                low = mid  # loss below threshold, search higher severity

        return {
            "event_type": event_type,
            "target_max_loss_pct": round(max_loss_pct, 2),
            "critical_severity": round(best_severity, 4),
            "critical_impact_score": round(min(10.0, best_severity * 10.0), 2),
            "estimated_pnl": best_res.total_pnl if best_res else 0.0,
            "actual_loss_pct": abs(best_res.pnl_pct) if best_res else 0.0,
            "total_d_ecl": best_res.total_d_ecl if best_res else 0.0,
            "scenario_name": self.scenarios[event_type].name,
        }
