"""Resilient client querying signals via FastAPI HTTP or falling back directly to SQLite."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from src.api.schemas import RiskSignal
from src.config import logger
from src.nlp.engine import get_risk_engine
from src.storage.store import get_signal_store


class SignalClient:
    """Hybrid client that queries FastAPI if running, or reads directly from SQLite for offline autonomy."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 2.0) -> None:
        self.base_url = (base_url or os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout
        self.store = get_signal_store()
        self._is_api_available: Optional[bool] = None

    def is_api_online(self) -> bool:
        """Checks if the FastAPI backend is responsive."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_signals(
        self,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        min_impact: Optional[float] = None,
        source_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RiskSignal]:
        """Fetches signals via HTTP if available, else queries SQLite directly."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_type:
            params["event_type"] = event_type
        if min_impact is not None:
            params["min_impact"] = min_impact
        if source_type:
            params["source_type"] = source_type
        if since:
            params["since"] = since.isoformat()

        try:
            resp = requests.get(f"{self.base_url}/signals", params=params, timeout=self.timeout)
            if resp.status_code == 200:
                raw_items = resp.json()
                return [RiskSignal.model_validate(item) for item in raw_items]
        except Exception as exc:
            logger.debug(f"API request failed ({exc}). Falling back directly to local SQLite storage.")

        # Fallback to direct SQLite read
        return self.store.query(
            ticker=ticker,
            event_type=event_type,
            min_impact=min_impact,
            source_type=source_type,
            since=since,
            limit=limit,
        )

    def get_latest(self, limit: int = 20) -> List[RiskSignal]:
        """Retrieves most recent signals."""
        try:
            resp = requests.get(f"{self.base_url}/signals/latest", params={"limit": limit}, timeout=self.timeout)
            if resp.status_code == 200:
                raw_items = resp.json()
                return [RiskSignal.model_validate(item) for item in raw_items]
        except Exception:
            pass

        return self.store.latest(n=limit)

    def get_summary(self) -> Dict[str, Any]:
        """Retrieves aggregated statistics."""
        try:
            resp = requests.get(f"{self.base_url}/signals/summary", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        return self.store.summary()

    def analyze(
        self,
        text: str,
        source_type: str = "news",
        source: str = "ad_hoc",
        title: Optional[str] = None,
    ) -> Optional[RiskSignal]:
        """Submits ad-hoc text to API or executes local RiskEngine directly."""
        payload = {
            "text": text,
            "source_type": source_type,
            "source": source,
            "title": title,
        }
        try:
            resp = requests.post(f"{self.base_url}/analyze", json=payload, timeout=5.0)
            if resp.status_code in (200, 201):
                return RiskSignal.model_validate(resp.json())
        except Exception:
            pass

        # Local RiskEngine fallback
        engine = get_risk_engine()
        signal = engine.process_single(
            text=text,
            title=title,
            source=source,
            source_type=source_type,
        )
        if signal:
            self.store.save([signal])
        return signal

    def delete_injected(self) -> int:
        """Removes demo-injected signals from local storage."""
        return self.store.delete_injected()
