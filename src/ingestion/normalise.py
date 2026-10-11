"""Data normalization schemas and deduplication logic for ingested documents."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


def generate_doc_id(url: Optional[str], text: str) -> str:
    """Generates a deterministic SHA1 hash for a document using url or text."""
    seed = (url.strip() if url and url.strip() else text.strip()).encode("utf-8")
    return hashlib.sha1(seed).hexdigest()


class NormalizedDoc(BaseModel):
    """Standardized representation of an ingested document across all sources."""

    doc_id: str
    source: str
    source_type: Literal["news", "social"]
    published_at: datetime
    title: str
    text: str
    url: Optional[str] = None
    language: str = "en"
    ticker_hint: Optional[str] = None
    tickers: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    other_tickers: List[str] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        source: str,
        source_type: Literal["news", "social"],
        published_at: datetime,
        title: str,
        text: str,
        url: Optional[str] = None,
        language: str = "en",
        ticker_hint: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
        other_tickers: Optional[List[str]] = None,
    ) -> NormalizedDoc:
        """Helper factory method that auto-computes doc_id and ensures UTC timezone."""
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        else:
            published_at = published_at.astimezone(timezone.utc)

        doc_id = generate_doc_id(url=url, text=text or title)
        return cls(
            doc_id=doc_id,
            source=source,
            source_type=source_type,
            published_at=published_at,
            title=title.strip(),
            text=text.strip(),
            url=url.strip() if url else None,
            language=language,
            ticker_hint=ticker_hint,
            tickers=tickers or [],
            countries=countries or [],
            sectors=sectors or [],
            other_tickers=other_tickers or [],
        )


def _normalize_title(title: str) -> str:
    """Strips punctuation, collapses whitespace, and lowercases title for fuzzy matching."""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(cleaned.split())


def dedupe(docs: List[NormalizedDoc]) -> List[NormalizedDoc]:
    """Drops duplicate documents by exact doc_id and near-duplicate titles.

    Preserves original order, keeping the earliest document encountered.
    """
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    deduped: List[NormalizedDoc] = []

    for doc in docs:
        if doc.doc_id in seen_ids:
            continue

        norm_title = _normalize_title(doc.title) if doc.title else ""
        if norm_title and norm_title in seen_titles:
            continue

        seen_ids.add(doc.doc_id)
        if norm_title:
            seen_titles.add(norm_title)
        deduped.append(doc)

    return deduped
