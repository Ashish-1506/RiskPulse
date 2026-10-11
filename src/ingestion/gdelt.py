"""GDELT DOC 2.0 API ingestion source for global news."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from src.config import get_config, logger
from src.ingestion.base import Source
from src.ingestion.cache import get_cached_response, make_cache_key, set_cached_response
from src.ingestion.normalise import NormalizedDoc

DEFAULT_RISK_KEYWORDS = [
    "sanctions",
    "default",
    "downgrade",
    "merger",
    "rate hike",
    "cyberattack",
    "earthquake",
    "war",
]


class GdeltSource(Source):
    """Fetches and normalizes news articles from GDELT DOC 2.0 API."""

    def __init__(
        self,
        query_url: Optional[str] = None,
        max_records: Optional[int] = None,
        timespan: Optional[str] = None,
        timeout_sec: Optional[int] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        cfg = get_config()
        gdelt_cfg = cfg.sources.gdelt
        self.query_url = query_url or gdelt_cfg.query_url
        self.max_records = max_records or gdelt_cfg.max_records
        self.timespan = timespan or gdelt_cfg.timespan
        self.timeout_sec = timeout_sec or gdelt_cfg.timeout_sec
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.tickers = cfg.tickers

    def _build_query(self) -> str:
        """Constructs a composite query from top company aliases and risk keywords."""
        company_terms = []
        for sym, meta in list(self.tickers.items())[:8]:
            company_terms.append(f'"{meta.name}"')
            company_terms.append(sym)
        company_clause = "(" + " OR ".join(company_terms) + ")"

        keyword_clause = "(" + " OR ".join([f'"{kw}"' if " " in kw else kw for kw in DEFAULT_RISK_KEYWORDS]) + ")"
        return f"{company_clause} {keyword_clause} sourcelang:english"

    def _parse_gdelt_date(self, date_str: Optional[str]) -> datetime:
        """Parses GDELT timestamp (e.g. 20261010T120000Z or 20261010120000) into UTC datetime."""
        if not date_str:
            return datetime.now(timezone.utc)
        clean = date_str.replace("Z", "").replace("T", "")
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
            try:
                dt = datetime.strptime(clean[: len(fmt.replace("%", ""))], fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    def _detect_ticker_hint(self, title: str) -> Optional[str]:
        """Scans title for company aliases to provide an initial ticker hint."""
        title_lower = title.lower()
        for symbol, meta in self.tickers.items():
            if symbol.lower() in title_lower.split():
                return symbol
            for alias in meta.aliases:
                if alias.lower() in title_lower:
                    return symbol
        return None

    def _fetch_from_api(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Executes GET request with exponential backoff retry and cache support."""
        cache_key = make_cache_key("gdelt", params, hour_bucket=True)
        cached = get_cached_response(cache_key)
        if cached and isinstance(cached, dict):
            logger.debug("Using cached GDELT response.")
            return cached

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.query_url, params=params, timeout=self.timeout_sec)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        set_cached_response(cache_key, data)
                        return data
                    except ValueError:
                        logger.warning("GDELT returned non-JSON 200 response (likely text alert).")
                        return None
                elif response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"GDELT returned HTTP {response.status_code}. Retry {attempt}/{self.max_retries}...")
                else:
                    logger.warning(f"GDELT returned unexpected HTTP {response.status_code}.")
                    return None
            except requests.RequestException as exc:
                logger.warning(f"GDELT request failed (attempt {attempt}/{self.max_retries}): {exc}")

            if attempt < self.max_retries:
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        return None

    def fetch(self, since: Optional[datetime] = None, limit: int = 100) -> List[NormalizedDoc]:
        """Fetches and standardizes articles from GDELT."""
        records_to_request = min(limit, self.max_records)
        query = self._build_query()
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "timespan": self.timespan,
            "maxrecords": records_to_request,
            "sort": "DateDesc",
        }

        data = self._fetch_from_api(params)
        if not data or not isinstance(data, dict):
            return []

        articles = data.get("articles", [])
        docs: List[NormalizedDoc] = []

        for item in articles:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            pub_dt = self._parse_gdelt_date(item.get("seendate"))
            if since and pub_dt < since:
                continue

            url = item.get("url")
            ticker = self._detect_ticker_hint(title)

            doc = NormalizedDoc.create(
                source="gdelt",
                source_type="news",
                published_at=pub_dt,
                title=title,
                text=title,  # GDELT ArtList returns headline & snippets
                url=url,
                language="en",
                ticker_hint=ticker,
            )
            docs.append(doc)
            if len(docs) >= limit:
                break

        return docs
