"""Pydantic schemas for RiskPulse API and signal representations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RiskSignal(BaseModel):
    """Unified machine-readable risk signal emitted per processed event."""

    signal_id: str = Field(description="Deterministic SHA-1 identifier of the signal")
    timestamp: datetime = Field(description="UTC timestamp of the event")
    source: str = Field(description="Originating source name (e.g., reuters, gdelt, tweets)")
    source_type: Literal["news", "social"] = Field(description="Categorical source medium")
    headline: str = Field(description="Extracted and cleaned headline or post title")
    url: Optional[str] = Field(default=None, description="Original article/post URL")
    tickers: List[str] = Field(default_factory=list, description="Matched portfolio or equity tickers")
    countries: List[str] = Field(default_factory=list, description="Geopolitical countries/regions affected")
    sectors: List[str] = Field(default_factory=list, description="GICS market sectors affected")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Normalized sentiment score in [-1.0, 1.0]")
    sentiment_label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"] = Field(description="Categorical polarity label")
    event_type: str = Field(description="1 of 12 taxonomical risk event categories")
    event_confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in event classification")
    impact_score: float = Field(ge=1.0, le=10.0, description="Predicted market severity score in [1.0, 10.0]")
    impact_components: Dict[str, Any] = Field(default_factory=dict, description="Component weight attribution")
    impact_rationale: str = Field(description="Human-readable explanation of impact severity")
    model_version: str = Field(description="Active AI/NLP model backend and pipeline version")


class AnalyzeRequest(BaseModel):
    """Payload for on-demand text risk analysis."""

    text: str = Field(min_length=5, description="Unstructured financial text or social post")
    source_type: Literal["news", "social"] = Field(default="news", description="Source channel type")
    source: str = Field(default="ad_hoc", description="Optional source label")
    title: Optional[str] = Field(default=None, description="Optional headline/title")
    url: Optional[str] = Field(default=None, description="Optional source link")


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    version: str
    model_version: str
    total_signals_stored: int
    timestamp: datetime


class SignalsSummaryResponse(BaseModel):
    """Aggregated risk signals summary statistics."""

    total_signals: int
    by_event_type: Dict[str, int]
    by_source_type: Dict[str, int]
    avg_sentiment_per_ticker: Dict[str, float]
    highest_impact_signals: List[RiskSignal]
