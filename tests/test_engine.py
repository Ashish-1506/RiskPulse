"""Unit tests for the integrated RiskEngine pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api.schemas import RiskSignal
from src.ingestion.normalise import NormalizedDoc
from src.nlp.engine import RiskEngine


@pytest.fixture
def test_engine() -> RiskEngine:
    """Fixture returning a RiskEngine configured with fast VADER and offline rule mode."""
    return RiskEngine(sentiment_backend="vader", use_zero_shot=False)


def test_risk_engine_end_to_end_processing(test_engine: RiskEngine):
    """Processes a batch of raw documents and verifies complete RiskSignal schema attributes."""
    doc1 = NormalizedDoc.create(
        source="newsapi",
        source_type="news",
        published_at=datetime(2026, 10, 10, 10, 0, 0, tzinfo=timezone.utc),
        title="JPMorgan Chase reports record profit and beats revenue expectations",
        text="JPMorgan CEO Jamie Dimon announced strong consumer and corporate banking margins.",
        url="https://cnbc.com/jpm-record",
    )

    doc2 = NormalizedDoc.create(
        source="tweets",
        source_type="social",
        published_at=datetime(2026, 10, 10, 11, 0, 0, tzinfo=timezone.utc),
        title="Severe default risk facing retailer as credit rating downgraded to junk",
        text="Rating agency Moody's slashes outlook citing emergency liquidity crunch for $WMT suppliers.",
    )

    signals = test_engine.process([doc1, doc2])

    assert len(signals) == 2
    for s in signals:
        assert isinstance(s, RiskSignal)
        assert s.signal_id
        assert s.timestamp.tzinfo == timezone.utc
        assert s.source_type in ("news", "social")
        assert -1.0 <= s.sentiment_score <= 1.0
        assert s.sentiment_label in ("POSITIVE", "NEUTRAL", "NEGATIVE")
        assert s.event_type
        assert 0.0 <= s.event_confidence <= 1.0
        assert 1.0 <= s.impact_score <= 10.0
        assert s.impact_rationale
        assert s.model_version

    # First signal should be positive earnings
    assert signals[0].sentiment_label == "POSITIVE"
    assert "JPM" in signals[0].tickers

    # Second signal should be negative credit distress with higher impact
    assert signals[1].sentiment_label == "NEGATIVE"
    assert signals[1].event_type == "Credit Event"
    assert signals[1].impact_score >= 6.0


def test_risk_engine_filters_empty_and_garbage(test_engine: RiskEngine):
    """Validates that empty, sub-minimum length, or non-English news texts are filtered out."""
    d_empty = NormalizedDoc.create("rss", "news", datetime.now(timezone.utc), "", "")
    d_short = NormalizedDoc.create("rss", "news", datetime.now(timezone.utc), "hi", "hi")
    d_foreign = NormalizedDoc.create(
        "newsapi",
        "news",
        datetime.now(timezone.utc),
        "El Banco Central Europeo sube los tipos de interés",
        "Noticia en espanol sobre la subida de tipos de interés.",
    )

    signals = test_engine.process([d_empty, d_short, d_foreign])
    assert len(signals) == 0
