"""Configuration loader, schema validation, and logging initialization for RiskPulse."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Load environment variables from .env if present
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


# ------------------------------------------------------------------------------
# Pydantic Schemas for Configuration
# ------------------------------------------------------------------------------
class AppConfig(BaseModel):
    name: str = "RiskPulse"
    version: str = "0.1.0"
    env: str = "development"
    log_level: str = "INFO"


class TickerMeta(BaseModel):
    name: str
    sector: str
    base_weight: float
    aliases: List[str] = Field(default_factory=list)


class GDELTConfig(BaseModel):
    enabled: bool = True
    query_url: str
    max_records: int = 50
    timespan: str = "24h"
    timeout_sec: int = 10


class RSSFeed(BaseModel):
    name: str
    url: str
    source_type: str = "news"


class RSSConfig(BaseModel):
    enabled: bool = True
    timeout_sec: int = 8
    feeds: List[RSSFeed] = Field(default_factory=list)


class NewsAPIConfig(BaseModel):
    enabled: bool = False
    page_size: int = 25


class SocialConfig(BaseModel):
    replay_file: str = "data/tweets_sample.jsonl"
    replay_speed_multiplier: float = 1.0
    reddit_enabled: bool = False
    stocktwits_enabled: bool = False


class SourcesConfig(BaseModel):
    gdelt: GDELTConfig
    rss: RSSConfig
    newsapi: NewsAPIConfig
    social: SocialConfig


class SentimentConfig(BaseModel):
    backend: str = "finbert"
    model_name: str = "ProsusAI/finbert"
    neutral_band: float = 0.15
    batch_size: int = 16
    max_length: int = 128
    device: str = "cpu"


class ImpactWeights(BaseModel):
    event_prior: float = 0.40
    sentiment: float = 0.25
    keywords: float = 0.15
    source: float = 0.10
    corroboration: float = 0.10


class ImpactConfig(BaseModel):
    weights: ImpactWeights
    source_credibility: Dict[str, float] = Field(default_factory=dict)


class StressConfig(BaseModel):
    trigger_threshold: float = 7.0
    severity_scale: str = "impact/10"
    exposure_link_bonus: float = 0.5


class RebalancerConfig(BaseModel):
    alpha: float = 0.5
    kappa: float = 1.0
    decay_lambda_per_hour: float = 0.05
    min_weight: float = 0.01
    max_weight: float = 0.15
    turnover_cap: float = 0.10
    no_trade_band: float = 0.0025
    social_weight: float = 0.60
    cost_bps: float = 10.0


class RootConfig(BaseModel):
    app: AppConfig
    tickers: Dict[str, TickerMeta]
    sources: SourcesConfig
    sentiment: SentimentConfig
    impact: ImpactConfig
    stress: StressConfig
    rebalancer: RebalancerConfig


class EventItem(BaseModel):
    prior: float = 0.50
    prior_severity: float = 0.50
    description: str
    keywords: List[str] = Field(default_factory=list)
    patterns: Dict[str, float] = Field(default_factory=dict)
    default_shock_scenario: str = "general_volatility"

    def model_post_init(self, __context: Any) -> None:
        if self.patterns and not self.keywords:
            self.keywords = list(self.patterns.keys())
        if self.prior != 0.50 and self.prior_severity == 0.50:
            self.prior_severity = self.prior
        elif self.prior_severity != 0.50 and self.prior == 0.50:
            self.prior = self.prior_severity


class EventsConfig(BaseModel):
    events: Dict[str, EventItem]


class ShockValues(BaseModel):
    equity_pct: float = 0.0
    rates_bps: float = 0.0
    credit_spread_bps: float = 0.0
    fx_pct: float = 0.0
    commodity_pct: float = 0.0
    pd_multiplier: float = 1.0


class ScenarioItem(BaseModel):
    name: str
    description: str
    equity: Optional[float] = None
    rates_bps: Optional[float] = None
    spread_bps: Optional[float] = None
    fx_em: Optional[float] = None
    fx_usd: Optional[float] = None
    commodity: Optional[float] = None
    pd_multiplier: Optional[float] = None
    affected_sector_only: bool = False
    affected_region_only: bool = False
    shocks: Optional[ShockValues] = None

    def model_post_init(self, __context: Any) -> None:
        if self.shocks is None:
            self.shocks = ShockValues(
                equity_pct=self.equity or 0.0,
                rates_bps=self.rates_bps or 0.0,
                credit_spread_bps=self.spread_bps or 0.0,
                fx_pct=self.fx_em or 0.0,
                commodity_pct=self.commodity or 0.0,
                pd_multiplier=self.pd_multiplier or 1.0,
            )


class ScenariosConfig(BaseModel):
    scenarios: Dict[str, ScenarioItem]


# ------------------------------------------------------------------------------
# Logging Initialization
# ------------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configures unified standard logging across all RiskPulse modules."""
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        force=True,
    )
    logger = logging.getLogger("RiskPulse")
    logger.setLevel(numeric_level)
    return logger


# ------------------------------------------------------------------------------
# Config Loader Functions with LRU Caching
# ------------------------------------------------------------------------------
def _read_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


@lru_cache(maxsize=1)
def get_config(config_dir: Optional[str] = None) -> RootConfig:
    """Loads and validates config.yaml into a validated RootConfig model."""
    target_dir = Path(config_dir) if config_dir else BASE_DIR / "config"
    config_file = target_dir / "config.yaml"
    raw_data = _read_yaml_file(config_file)

    # Auto-enable NewsAPI if key present in environment
    if os.getenv("NEWSAPI_KEY"):
        if "sources" in raw_data and "newsapi" in raw_data["sources"]:
            raw_data["sources"]["newsapi"]["enabled"] = True

    # Override sentiment backend if specified via env
    env_backend = os.getenv("SENTIMENT_BACKEND")
    if env_backend and "sentiment" in raw_data:
        raw_data["sentiment"]["backend"] = env_backend.lower()

    return RootConfig.model_validate(raw_data)


@lru_cache(maxsize=1)
def get_events_config(config_dir: Optional[str] = None) -> EventsConfig:
    """Loads and validates events.yaml taxonomy."""
    target_dir = Path(config_dir) if config_dir else BASE_DIR / "config"
    events_file = target_dir / "events.yaml"
    raw_data = _read_yaml_file(events_file)
    return EventsConfig.model_validate(raw_data)


@lru_cache(maxsize=1)
def get_scenarios_config(config_dir: Optional[str] = None) -> ScenariosConfig:
    """Loads and validates scenarios.yaml stress matrices."""
    target_dir = Path(config_dir) if config_dir else BASE_DIR / "config"
    scenarios_file = target_dir / "scenarios.yaml"
    raw_data = _read_yaml_file(scenarios_file)
    return ScenariosConfig.model_validate(raw_data)


# Initialize top-level application logger
logger = setup_logging()
