"""NewsAPI v2 ingestion source with graceful key handling and 429 resilience."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import requests

from src.config import get_config, logger
from src.ingestion.base import Source
from src.ingestion.cache import get_cached_response, make_cache_key, set_cached_response
from src.ingestion.normalise import NormalizedDoc

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


class NewsApiSource(Source):
    """Fetches and normalizes news from the NewsAPI service."""

    def __init__(self, api_key: Optional[str] = None, page_size: Optional[int] = None) -> None:
        cfg = get_config()
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.page_size = page_size or cfg.sources.newsapi.page_size
        self.tickers = cfg.tickers

    def _detect_ticker_hint(self, title: str) -> Optional[str]:
        """Matches ticker symbol or company name in title."""
        title_lower = title.lower()
        for symbol, meta in self.tickers.items():
            if symbol.lower() in title_lower.split():
                return symbol
            for alias in meta.aliases:
                if alias.lower() in title_lower:
                    return symbol
        return None

    def _parse_iso_date(self, date_str: Optional[str]) -> datetime:
        """Parses ISO-8601 publishedAt timestamp into UTC datetime."""
        if not date_str:
            return datetime.now(timezone.utc)
        try:
            clean = date_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)

    def fetch(self, since: Optional[datetime] = None, limit: int = 100) -> List[NormalizedDoc]:
        """Fetches articles using NewsAPI everything endpoint."""
        if not self.api_key:
            logger.warning("NewsAPI key not found in environment (NEWSAPI_KEY). Skipping source.")
            return []

        # Build query focused on configured tickers and risk keywords
        query = "(Apple OR Microsoft OR Nvidia OR JPMorgan OR Tesla) AND (default OR downgrade OR sanctions OR revenue)"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, self.page_size),
        }
        if since:
            params["from"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Cache lookup (excluding secret key from cache key)
        cache_key = make_cache_key("newsapi", params, hour_bucket=True)
        cached_data = get_cached_response(cache_key)

        headers = {"X-Api-Key": self.api_key}

        if cached_data and isinstance(cached_data, dict):
            logger.debug("Using cached NewsAPI response.")
            data = cached_data
        else:
            try:
                response = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=10)
                if response.status_code == 429:
                    logger.warning("NewsAPI rate limit reached (HTTP 429). Returning empty list.")
                    return []
                if response.status_code != 200:
                    logger.warning(f"NewsAPI error (HTTP {response.status_code}).")
                    return []
                data = response.json()
                set_cached_response(cache_key, data)
            except requests.RequestException as exc:
                logger.warning(f"NewsAPI network request failed: {exc}")
                return []

        articles = data.get("articles", [])
        docs: List[NormalizedDoc] = []

        for item in articles:
            title = (item.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue

            pub_dt = self._parse_iso_date(item.get("publishedAt"))
            if since and pub_dt < since:
                continue

            description = item.get("description") or ""
            content = item.get("content") or ""
            text = f"{title}. {description} {content}".strip()
            url = item.get("url")
            ticker = self._detect_ticker_hint(title)

            doc = NormalizedDoc.create(
                source="newsapi",
                source_type="news",
                published_at=pub_dt,
                title=title,
                text=text,
                url=url,
                language="en",
                ticker_hint=ticker,
            )
            docs.append(doc)
            if len(docs) >= limit:
                break

        return docs
