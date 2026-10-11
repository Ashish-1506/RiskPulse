"""Unit tests for the ingestion subsystem with full network isolation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.cache import get_cached_response, make_cache_key, set_cached_response
from src.ingestion.gdelt import GdeltSource
from src.ingestion.newsapi import NewsApiSource
from src.ingestion.normalise import NormalizedDoc, dedupe, generate_doc_id
from src.ingestion.rss import RssSource
from src.ingestion.tweets_replay import TweetsReplaySource


# ------------------------------------------------------------------------------
# 1. Normalization & Deduplication Tests
# ------------------------------------------------------------------------------
def test_doc_id_generation_deterministic():
    """Validates that doc_id is consistent and deterministic."""
    id1 = generate_doc_id("https://example.com/story1", "Some news text")
    id2 = generate_doc_id("https://example.com/story1", "Different text")
    assert id1 == id2

    id3 = generate_doc_id(None, "Unique text snippet")
    id4 = generate_doc_id("", "Unique text snippet")
    assert id3 == id4
    assert id1 != id3


def test_normalized_doc_utc_coercion():
    """Validates that naive and offset-aware datetimes are converted to UTC."""
    naive_dt = datetime(2026, 10, 10, 15, 30, 0)
    doc = NormalizedDoc.create(
        source="gdelt",
        source_type="news",
        published_at=naive_dt,
        title="Apple announces new M5 chip",
        text="Full text",
        url="https://example.com/m5",
    )
    assert doc.published_at.tzinfo == timezone.utc
    assert doc.doc_id


def test_dedupe_drops_duplicates_and_near_titles():
    """Validates deduplication by doc_id and normalized punctuation/casing."""
    now = datetime.now(timezone.utc)
    doc1 = NormalizedDoc.create("gdelt", "news", now, "Fed Cuts Rates by 25bps!", "Text 1", url="http://a.com/1")
    # Same URL -> same doc_id
    doc2 = NormalizedDoc.create("rss", "news", now, "Fed cuts rates by 25bps!", "Text 2", url="http://a.com/1")
    # Different URL, but near-identical normalized title
    doc3 = NormalizedDoc.create("newsapi", "news", now, "fed cuts rates by 25bps...", "Text 3", url="http://b.com/2")
    # Distinct title
    doc4 = NormalizedDoc.create(
        "gdelt", "news", now, "Oil prices surge on Middle East tensions", "Text 4", url="http://c.com/3"
    )

    raw = [doc1, doc2, doc3, doc4]
    deduped = dedupe(raw)

    assert len(deduped) == 2
    assert deduped[0].doc_id == doc1.doc_id
    assert deduped[1].title == doc4.title


# ------------------------------------------------------------------------------
# 2. GDELT Ingestion Tests (Mocked)
# ------------------------------------------------------------------------------
def test_gdelt_mock_success(monkeypatch: pytest.MonkeyPatch):
    """Mocks GDELT DOC API returning valid JSON articles."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "articles": [
            {
                "url": "https://reuters.com/apple-antitrust",
                "title": "Apple Faces European Commission Antitrust Inquiry",
                "seendate": "20261010T143000Z",
                "domain": "reuters.com",
            },
            {
                "url": "https://bloomberg.com/jpmorgan-guidance",
                "title": "JPMorgan Raises Net Interest Income Forecast",
                "seendate": "20261010T121500Z",
                "domain": "bloomberg.com",
            },
        ]
    }
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    monkeypatch.setattr("src.ingestion.gdelt.get_cached_response", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ingestion.gdelt.set_cached_response", lambda *args, **kwargs: None)

    source = GdeltSource()
    docs = source.fetch(limit=10)

    assert len(docs) == 2
    assert docs[0].source == "gdelt"
    assert docs[0].source_type == "news"
    assert docs[0].ticker_hint == "AAPL"
    assert docs[1].ticker_hint == "JPM"


def test_gdelt_mock_non_json_error(monkeypatch: pytest.MonkeyPatch):
    """Gracefully handles GDELT returning text error message without throwing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("No JSON object could be decoded")
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    monkeypatch.setattr("src.ingestion.gdelt.get_cached_response", lambda *args, **kwargs: None)

    source = GdeltSource()
    docs = source.fetch(limit=5)
    assert docs == []


# ------------------------------------------------------------------------------
# 3. NewsAPI Ingestion Tests (Mocked)
# ------------------------------------------------------------------------------
def test_newsapi_missing_key_behavior(monkeypatch: pytest.MonkeyPatch):
    """Validates that missing NEWSAPI_KEY logs warning and safely returns empty list."""
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    source = NewsApiSource(api_key=None)
    docs = source.fetch(limit=10)
    assert docs == []


def test_newsapi_mock_429_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """Validates that HTTP 429 returns empty list gracefully."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    monkeypatch.setattr("src.ingestion.newsapi.get_cached_response", lambda *args, **kwargs: None)

    source = NewsApiSource(api_key="fake-test-key")
    docs = source.fetch(limit=10)
    assert docs == []


def test_newsapi_mock_success(monkeypatch: pytest.MonkeyPatch):
    """Validates parsing of successful NewsAPI payload."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "articles": [
            {
                "title": "Microsoft and OpenAI Expand Partnership",
                "description": "New data center infrastructure deal announced.",
                "url": "https://cnbc.com/msft-openai",
                "publishedAt": "2026-10-10T10:00:00Z",
            }
        ]
    }
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
    monkeypatch.setattr("src.ingestion.newsapi.get_cached_response", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.ingestion.newsapi.set_cached_response", lambda *args, **kwargs: None)

    source = NewsApiSource(api_key="fake-test-key")
    docs = source.fetch(limit=5)

    assert len(docs) == 1
    assert docs[0].source == "newsapi"
    assert docs[0].ticker_hint == "MSFT"


# ------------------------------------------------------------------------------
# 4. RSS Ingestion Tests (Mocked)
# ------------------------------------------------------------------------------
def test_rss_mock_feed(monkeypatch: pytest.MonkeyPatch):
    """Mocks feedparser returning entries."""
    mock_entry = MagicMock()
    mock_entry.title = "NVIDIA Reports Record Data Center Revenue"
    mock_entry.summary = "Blackwell GPUs drive revenue surge."
    mock_entry.link = "https://finance.yahoo.com/nvda-earnings"
    mock_entry.published_parsed = (2026, 10, 10, 8, 30, 0, 5, 283, 0)

    mock_parsed = MagicMock()
    mock_parsed.bozo = 0
    mock_parsed.entries = [mock_entry]

    monkeypatch.setattr("feedparser.parse", lambda url: mock_parsed)

    source = RssSource()
    docs = source.fetch(limit=5)

    assert len(docs) >= 1
    assert docs[0].source == "rss"
    assert docs[0].source_type == "news"
    assert docs[0].ticker_hint == "NVDA"


# ------------------------------------------------------------------------------
# 5. Tweets Replay Tests
# ------------------------------------------------------------------------------
def test_tweets_replay_sorting_and_schema(tmp_path: Path):
    """Validates CSV reading, column detection, chronological sorting, and social type."""
    csv_file = tmp_path / "custom_tweets.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "text", "ticker"])
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-10-10T12:00:00Z",
                "text": "Second tweet about $TSLA delivery numbers",
                "ticker": "TSLA",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-10-10T08:00:00Z",
                "text": "First tweet about $AAPL product launch",
                "ticker": "AAPL",
            }
        )

    source = TweetsReplaySource(file_path=csv_file)
    docs = source.fetch(limit=10)

    assert len(docs) == 2
    # Verify chronological ascending order
    assert docs[0].published_at < docs[1].published_at
    assert docs[0].source == "tweets"
    assert docs[0].source_type == "social"
    assert docs[0].ticker_hint == "AAPL"
    assert docs[1].ticker_hint == "TSLA"


# ------------------------------------------------------------------------------
# 6. Disk Caching Utility Tests
# ------------------------------------------------------------------------------
def test_disk_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Validates disk cache write and read behavior."""
    monkeypatch.setattr("src.ingestion.cache.CACHE_DIR", tmp_path)

    key = make_cache_key("test", {"query": "sample"}, hour_bucket=False)
    data = {"result": "success", "items": [1, 2, 3]}

    assert get_cached_response(key) is None
    set_cached_response(key, data)
    retrieved = get_cached_response(key)

    assert retrieved == data
