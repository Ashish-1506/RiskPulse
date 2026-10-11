"""Scenario Definition, Loading, and Dynamic Shock Builder for Stress Testing.

Translates AI RiskSignals into cross-asset macroeconomic shocks:
  - Rates curve shift (bps)
  - Credit spread widening (bps)
  - Equity market drawdown (%)
  - FX movements vs USD (%)
  - Probability of Default (PD) multiplier
  - Exposure Link Bonus: 50% extra shock for counterparties in directly hit sector/country.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.api.schemas import RiskSignal
from src.config import BASE_DIR, get_config

SCENARIOS_YAML_PATH = BASE_DIR / "config" / "scenarios.yaml"


@dataclass
class Scenario:
    """Baseline stress shock parameters calibrated at maximum impact = 10.0."""

    event_type: str
    name: str
    description: str
    equity: float  # e.g. -0.12 (-12%)
    rates_bps: float  # e.g. -30.0 bps
    spread_bps: float  # e.g. +120.0 bps
    fx_em: float  # e.g. -0.08 (-8% vs USD)
    fx_usd: float  # e.g. +0.02 (+2% USD safe-haven)
    commodity: float  # e.g. +0.15 (+15%)
    pd_multiplier: float  # e.g. 1.5x
    affected_sector_only: bool = False
    affected_region_only: bool = False


@dataclass
class ShockPackage:
    """Dynamic cross-asset shocks ready for vectorised portfolio valuation."""

    scenario_name: str
    event_type: str
    severity: float  # impact_score / 10.0 (or custom severity in [0, 1.5])
    equity_shock: float  # decimal, e.g. -0.084
    rates_bps: float  # e.g. -21.0 bps
    spread_bps: float  # e.g. +84.0 bps
    fx_em_shock: float  # e.g. -0.056
    fx_usd_shock: float  # e.g. +0.014
    commodity_shock: float  # e.g. +0.105
    pd_multiplier: float  # e.g. 1.35x
    affected_sectors: List[str] = field(default_factory=list)
    affected_countries: List[str] = field(default_factory=list)
    exposure_link_bonus: float = 0.50
    affected_sector_only: bool = False
    affected_region_only: bool = False
    is_custom: bool = False


_SCENARIOS_CACHE: Optional[Dict[str, Scenario]] = None


def load_scenarios(yaml_path: Path = SCENARIOS_YAML_PATH) -> Dict[str, Scenario]:
    """Loads all calibrated event scenarios from config/scenarios.yaml."""
    global _SCENARIOS_CACHE
    if _SCENARIOS_CACHE is not None:
        return _SCENARIOS_CACHE

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Scenarios configuration not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenarios_dict = {}
    for event_type, cfg in data.get("scenarios", {}).items():
        scenarios_dict[event_type] = Scenario(
            event_type=event_type,
            name=cfg.get("name", event_type),
            description=cfg.get("description", ""),
            equity=float(cfg.get("equity", 0.0)),
            rates_bps=float(cfg.get("rates_bps", 0.0)),
            spread_bps=float(cfg.get("spread_bps", 0.0)),
            fx_em=float(cfg.get("fx_em", 0.0)),
            fx_usd=float(cfg.get("fx_usd", 0.0)),
            commodity=float(cfg.get("commodity", 0.0)),
            pd_multiplier=float(cfg.get("pd_multiplier", 1.0)),
            affected_sector_only=bool(cfg.get("affected_sector_only", False)),
            affected_region_only=bool(cfg.get("affected_region_only", False)),
        )

    _SCENARIOS_CACHE = scenarios_dict
    return _SCENARIOS_CACHE


def should_trigger(signal: RiskSignal, cfg: Optional[Any] = None) -> bool:
    """Determines whether a RiskSignal triggers a wholesale stress test.

    Trigger condition:
      1. impact_score >= trigger_threshold (from config, default 7.0)
      2. event_type is a defined stress scenario (Other event types return False).
    """
    app_cfg = cfg or get_config()
    threshold = float(getattr(app_cfg.stress, "trigger_threshold", 7.0))
    scenarios = load_scenarios()

    if signal.impact_score < threshold:
        return False

    return signal.event_type in scenarios


def build_shocks(
    signal: Optional[RiskSignal] = None,
    scenarios: Optional[Dict[str, Scenario]] = None,
    cfg: Optional[Any] = None,
    custom_shocks: Optional[Dict[str, Any]] = None,
    severity_override: Optional[float] = None,
) -> Optional[ShockPackage]:
    """Constructs dynamic shock parameters for a signal or custom slider configuration.

    Scaling:
      - severity = severity_override if provided else impact_score / 10.0
      - Every shock is scaled linearly by severity.
      - Positions whose country or sector appears in the signal receive an extra
        exposure_link_bonus (default +50%).
    """
    app_cfg = cfg or get_config()
    bonus = float(getattr(app_cfg.stress, "exposure_link_bonus", 0.50))

    # 1. Manual Custom Scenario (for Streamlit dashboard interactive sliders)
    if custom_shocks is not None:
        return ShockPackage(
            scenario_name=custom_shocks.get("name", "Custom Stress Scenario"),
            event_type="Custom",
            severity=float(custom_shocks.get("severity", 1.0)),
            equity_shock=float(custom_shocks.get("equity", 0.0)),
            rates_bps=float(custom_shocks.get("rates_bps", 0.0)),
            spread_bps=float(custom_shocks.get("spread_bps", 0.0)),
            fx_em_shock=float(custom_shocks.get("fx_em", 0.0)),
            fx_usd_shock=float(custom_shocks.get("fx_usd", 0.0)),
            commodity_shock=float(custom_shocks.get("commodity", 0.0)),
            pd_multiplier=float(custom_shocks.get("pd_multiplier", 1.0)),
            affected_sectors=custom_shocks.get("sectors", []),
            affected_countries=custom_shocks.get("countries", []),
            exposure_link_bonus=bonus,
            affected_sector_only=bool(custom_shocks.get("affected_sector_only", False)),
            affected_region_only=bool(custom_shocks.get("affected_region_only", False)),
            is_custom=True,
        )

    if signal is None:
        return None

    # 2. Look up calibrated scenario
    all_scenarios = scenarios or load_scenarios()
    base_scenario = all_scenarios.get(signal.event_type)
    if base_scenario is None:
        # Other / unmapped event types return None (no stress test)
        return None

    severity = severity_override if severity_override is not None else (signal.impact_score / 10.0)

    # Scale shocks by severity
    scaled_equity = base_scenario.equity * severity
    scaled_rates = base_scenario.rates_bps * severity
    scaled_spread = base_scenario.spread_bps * severity
    scaled_fx_em = base_scenario.fx_em * severity
    scaled_fx_usd = base_scenario.fx_usd * severity
    scaled_commodity = base_scenario.commodity * severity

    # PD multiplier: base is 1.0; delta scales with severity
    # e.g., if pd_multiplier = 1.5, delta is +0.5; scaled multiplier = 1.0 + 0.5 * severity
    scaled_pd_mult = 1.0 + (base_scenario.pd_multiplier - 1.0) * severity

    return ShockPackage(
        scenario_name=base_scenario.name,
        event_type=base_scenario.event_type,
        severity=round(severity, 3),
        equity_shock=round(scaled_equity, 5),
        rates_bps=round(scaled_rates, 2),
        spread_bps=round(scaled_spread, 2),
        fx_em_shock=round(scaled_fx_em, 5),
        fx_usd_shock=round(scaled_fx_usd, 5),
        commodity_shock=round(scaled_commodity, 5),
        pd_multiplier=round(scaled_pd_mult, 4),
        affected_sectors=signal.sectors or [],
        affected_countries=signal.countries or [],
        exposure_link_bonus=bonus,
        affected_sector_only=base_scenario.affected_sector_only,
        affected_region_only=base_scenario.affected_region_only,
        is_custom=False,
    )
