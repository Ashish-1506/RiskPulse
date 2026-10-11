"""Integration tests for FastAPI endpoints and the SignalClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.client import SignalClient
from src.api.main import app
from src.nlp.engine import get_risk_engine
from src.storage.store import SignalStore


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Initializes isolated temporary storage and fast VADER engine for API tests."""
    test_db = tmp_path / "test_api_signals.db"
    test_jsonl = tmp_path / "test_api_signals.jsonl"
    store = SignalStore(db_path=test_db, jsonl_path=test_jsonl)

    monkeypatch.setattr("src.api.main.get_signal_store", lambda: store)
    monkeypatch.setattr("src.storage.store.get_signal_store", lambda: store)
    monkeypatch.setattr("src.api.client.get_signal_store", lambda: store)

    # Initialize fast VADER RiskEngine
    engine = get_risk_engine(sentiment_backend="vader", use_zero_shot=False)
    monkeypatch.setattr("src.api.main.get_risk_engine", lambda: engine)
    monkeypatch.setattr("src.api.client.get_risk_engine", lambda: engine)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_health(client: TestClient):
    """Validates /health returns healthy status and model version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_version" in data
    assert "total_signals_stored" in data


def test_api_post_analyze(client: TestClient):
    """Validates /analyze endpoint performs on-demand NLP inference and returns RiskSignal."""
    payload = {
        "text": "Moody's downgrades Apple supplier to junk status citing debt default fears.",
        "source_type": "news",
        "source": "reuters",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["signal_id"]
    assert data["event_type"] == "Credit Event"
    assert data["sentiment_label"] == "NEGATIVE"
    assert data["impact_score"] >= 6.0
    assert "AAPL" in data["tickers"]


def test_api_query_signals_with_filters(client: TestClient):
    """Validates /signals with filtering by ticker and min_impact."""
    # Seed two signals via /analyze
    client.post(
        "/analyze",
        json={"text": "Apple announces record quarterly profit on strong iPhone demand.", "source_type": "news"},
    )
    client.post(
        "/analyze",
        json={"text": "Catastrophic default and bankruptcy in energy exploration firm.", "source_type": "news"},
    )

    # 1. Query all
    resp_all = client.get("/signals")
    assert resp_all.status_code == 200
    signals = resp_all.json()
    assert len(signals) == 2

    # 2. Query by ticker
    resp_ticker = client.get("/signals?ticker=AAPL")
    assert resp_ticker.status_code == 200
    ticker_sigs = resp_ticker.json()
    assert len(ticker_sigs) == 1
    assert "AAPL" in ticker_sigs[0]["tickers"]

    # 3. Query latest
    resp_latest = client.get("/signals/latest?limit=1")
    assert resp_latest.status_code == 200
    assert len(resp_latest.json()) == 1


def test_api_signals_summary(client: TestClient):
    """Validates /signals/summary aggregation statistics."""
    client.post(
        "/analyze",
        json={"text": "Federal Reserve unexpected rate hike triggers stock plunge.", "source_type": "news"},
    )

    resp = client.get("/signals/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["total_signals"] >= 1
    assert "Monetary Policy" in summary["by_event_type"]
    assert "news" in summary["by_source_type"]


def test_signal_client_fallback_mode():
    """Validates SignalClient direct SQLite fallback when API server is not running."""
    # Points to non-existent port to simulate offline API
    offline_client = SignalClient(base_url="http://127.0.0.1:9999", timeout=0.2)
    assert not offline_client.is_api_online()

    # Client directly analyzes and reads from SQLite
    sig = offline_client.analyze("Microsoft beats earnings estimates with cloud revenue surge.")
    assert sig is not None
    assert "MSFT" in sig.tickers

    signals = offline_client.get_signals(ticker="MSFT")
    assert len(signals) >= 1
    assert signals[0].tickers == ["MSFT"]
