"""RSS feed ingestion source using feedparser."""

from __future__ import annotations

import calendar
import time
from datetime import datetime, timezone
from typing import List, Optional

import feedparser

from src.config import get_config, logger
from src.ingestion.base import Source
from src.ingestion.normalise import NormalizedDoc


class RssSource(Source):
    """Fetches articles across configured financial RSS feeds."""

    def __init__(self, timeout_sec: Optional[int] = None) -> None:
        cfg = get_config()
        self.feeds = cfg.sources.rss.feeds
        self.timeout_sec = timeout_sec or cfg.sources.rss.timeout_sec
        self.tickers = cfg.tickers

    def _detect_ticker_hint(self, title: str) -> Optional[str]:
        """Identifies ticker symbol from headline."""
        title_lower = title.lower()
        for symbol, meta in self.tickers.items():
            if symbol.lower() in title_lower.split():
                return symbol
            for alias in meta.aliases:
                if alias.lower() in title_lower:
                    return symbol
        return None

    def _parse_time_struct(self, time_struct: Optional[time.struct_time]) -> datetime:
        """Converts feedparser time_struct to UTC datetime."""
        if not time_struct:
            return datetime.now(timezone.utc)
        timestamp = calendar.timegm(time_struct)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def fetch(self, since: Optional[datetime] = None, limit: int = 100) -> List[NormalizedDoc]:
        """Iterates over configured RSS feeds, gracefully skipping failing endpoints."""
        docs: List[NormalizedDoc] = []

        for feed_cfg in self.feeds:
            try:
                parsed = feedparser.parse(feed_cfg.url)
                if getattr(parsed, "bozo", 0) and not parsed.entries:
                    logger.warning(f"RSS feed parsing error for {feed_cfg.name} ({feed_cfg.url}). Skipping.")
                    continue

                for entry in parsed.entries:
                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue

                    time_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
                    pub_dt = self._parse_time_struct(time_struct)
                    if since and pub_dt < since:
                        continue

                    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                    link = getattr(entry, "link", None)
                    ticker = self._detect_ticker_hint(title)

                    doc = NormalizedDoc.create(
                        source="rss",
                        source_type="news",
                        published_at=pub_dt,
                        title=title,
                        text=f"{title}. {summary}".strip(),
                        url=link,
                        language="en",
                        ticker_hint=ticker,
                    )
                    docs.append(doc)
                    if len(docs) >= limit:
                        return docs
            except Exception as exc:
                logger.warning(f"Failed to fetch RSS feed {feed_cfg.name}: {exc}. Skipping.")
                continue

        return docs
