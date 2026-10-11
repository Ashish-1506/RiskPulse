"""FastAPI application serving real-time risk signals and on-demand NLP analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AnalyzeRequest,
    HealthResponse,
    RiskSignal,
    SignalsSummaryResponse,
)
from src.config import get_config
from src.nlp.engine import get_risk_engine
from src.storage.store import get_signal_store

cfg = get_config()

app = FastAPI(
    title="RiskPulse AI/NLP Risk Engine API",
    description=(
        "Production-grade financial NLP risk signal extraction engine for wholesale "
        "banking stress-testing and tactical portfolio rebalancing."
    ),
    version=cfg.app.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable permissive CORS for Streamlit dashboard and local consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check & system status",
)
def get_health() -> HealthResponse:
    """Returns application status, active AI/NLP model backends, and storage metrics."""
    engine = get_risk_engine()
    store = get_signal_store()
    stats = store.summary()
    return HealthResponse(
        status="healthy",
        version=cfg.app.version,
        model_version=engine.model_version,
        total_signals_stored=stats["total_signals"],
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/signals",
    response_model=List[RiskSignal],
    tags=["Signals"],
    summary="Query structured risk signals",
)
def get_signals(
    ticker: Optional[str] = Query(None, description="Filter by equity ticker symbol (e.g. AAPL)"),
    event_type: Optional[str] = Query(None, description="Filter by categorical event type"),
    min_impact: Optional[float] = Query(None, ge=1.0, le=10.0, description="Minimum market severity score"),
    source_type: Optional[str] = Query(
        None, pattern="^(news|social)$", description="Medium filter ('news' or 'social')"
    ),
    since: Optional[datetime] = Query(None, description="Cutoff timestamp (ISO 8601 UTC)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
) -> List[RiskSignal]:
    """Retrieves filtered historical and live-ingested risk signals."""
    store = get_signal_store()
    return store.query(
        ticker=ticker,
        event_type=event_type,
        min_impact=min_impact,
        source_type=source_type,
        since=since,
        limit=limit,
    )


@app.get(
    "/signals/latest",
    response_model=List[RiskSignal],
    tags=["Signals"],
    summary="Get recent risk signals",
)
def get_latest_signals(
    limit: int = Query(20, ge=1, le=200, description="Number of recent signals to return"),
) -> List[RiskSignal]:
    """Retrieves the most recent signals sorted chronologically descending."""
    store = get_signal_store()
    return store.latest(n=limit)


@app.get(
    "/signals/summary",
    response_model=SignalsSummaryResponse,
    tags=["Signals"],
    summary="Aggregated signal statistics",
)
def get_signals_summary() -> SignalsSummaryResponse:
    """Returns distributions across event types, source mediums, ticker sentiment averages, and top severities."""
    store = get_signal_store()
    data = store.summary()
    # Coerce highest impact back to RiskSignal models
    highest_sigs = [RiskSignal.model_validate(item) for item in data.get("highest_impact", [])]
    return SignalsSummaryResponse(
        total_signals=data["total_signals"],
        by_event_type=data["by_event_type"],
        by_source_type=data["by_source_type"],
        avg_sentiment_per_ticker=data["avg_sentiment_per_ticker"],
        highest_impact_signals=highest_sigs,
    )


@app.post(
    "/analyze",
    response_model=RiskSignal,
    status_code=status.HTTP_201_CREATED,
    tags=["Inference"],
    summary="Analyze ad-hoc financial text",
)
def analyze_text(request: AnalyzeRequest) -> RiskSignal:
    """Runs the full AI/NLP pipeline on arbitrary input text, persists the resulting signal, and returns it."""
    engine = get_risk_engine()
    store = get_signal_store()

    signal = engine.process_single(
        text=request.text,
        title=request.title,
        source=request.source,
        source_type=request.source_type,
        url=request.url,
    )

    if signal is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provided text could not be processed (possibly non-English or empty).",
        )

    store.save([signal])
    return signal
