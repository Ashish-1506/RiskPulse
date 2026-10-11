"""Unit tests for the sentiment analysis engine (VADER and FinBERT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.nlp.sentiment import SentimentAnalyzer, SentimentCache

SENTENCE_TEST_CASES = [
    ("Company posts record profit and raises guidance", "positive", lambda s: s > 0.15),
    ("Firm files for bankruptcy after fraud probe", "negative", lambda s: s < -0.15),
    ("Board meeting scheduled for Tuesday", "neutral", lambda s: abs(s) <= 0.20),
    ("Federal Reserve unexpected rate hike triggers stock plunge", "negative", lambda s: s < -0.15),
    ("Tech giant beats earnings estimates with blowout cloud growth", "positive", lambda s: s > 0.15),
    ("Credit rating agency downgrades sovereign debt to junk status", "negative", lambda s: s < -0.15),
    ("Quarterly revenue was unchanged compared to previous quarter", "neutral", lambda s: abs(s) <= 0.25),
    ("FDA approves revolutionary cancer therapy with strong clinical efficacy", "positive", lambda s: s > 0.15),
]


# ------------------------------------------------------------------------------
# 1. VADER Engine Tests (Fast, Always Run)
# ------------------------------------------------------------------------------
@pytest.fixture
def vader_analyzer(tmp_path: Path) -> SentimentAnalyzer:
    """Creates a SentimentAnalyzer configured strictly with VADER and temporary SQLite cache."""
    cache = SentimentCache(db_path=tmp_path / "test_sentiment.db")
    analyzer = SentimentAnalyzer(backend_override="vader")
    analyzer.cache = cache
    return analyzer


@pytest.mark.parametrize("sentence,expected_label,score_condition", SENTENCE_TEST_CASES)
def test_vader_sentiment_direction(
    vader_analyzer: SentimentAnalyzer,
    sentence: str,
    expected_label: str,
    score_condition,
):
    """Validates expected sentiment direction and label for 8 standard financial sentences."""
    results = vader_analyzer.analyze([sentence])
    assert len(results) == 1
    res = results[0]

    assert res.backend in ("vader", "vader_cached")
    assert -1.0 <= res.score <= 1.0
    assert score_condition(res.score), f"Score {res.score} failed condition for '{sentence}'"
    assert res.label == expected_label


def test_vader_caching_mechanism(vader_analyzer: SentimentAnalyzer):
    """Validates that second inference returns cached result."""
    text = "Company posts record profit and raises guidance"
    res1 = vader_analyzer.analyze([text])[0]
    assert "cached" not in res1.backend

    res2 = vader_analyzer.analyze([text])[0]
    assert "cached" in res2.backend
    assert res1.score == res2.score
    assert res1.label == res2.label


# ------------------------------------------------------------------------------
# 2. FinBERT Engine Tests (Marked as Slow)
# ------------------------------------------------------------------------------
@pytest.mark.slow
def test_finbert_sentiment_integration(tmp_path: Path):
    """Integrates FinBERT transformer model (downloads/loads ProsusAI/finbert)."""
    cache = SentimentCache(db_path=tmp_path / "test_finbert.db")
    analyzer = SentimentAnalyzer(backend_override="finbert")
    analyzer.cache = cache

    test_sentences = [
        "Company posts record profit and raises guidance",
        "Firm files for bankruptcy after fraud probe",
    ]
    results = analyzer.analyze(test_sentences)
    assert len(results) == 2

    # First sentence should be positive
    assert results[0].score > 0.0
    assert results[0].label == "positive"
    assert results[0].probs["positive"] > results[0].probs["negative"]

    # Second sentence should be negative
    assert results[1].score < 0.0
    assert results[1].label == "negative"
    assert results[1].probs["negative"] > results[1].probs["positive"]
