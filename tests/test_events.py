"""Unit tests for the event classification taxonomy, rule matcher, and confidence floor."""

from __future__ import annotations

import csv

import pytest

from src.config import BASE_DIR, get_events_config
from src.nlp.events import EventClassifier

TEST_HEADLINES = [
    # 1. Geopolitical
    ("US imposes new sanctions on Russian oil exporters", "Geopolitical"),
    ("Military invasion triggers missile strikes and retaliatory embargo", "Geopolitical"),
    # 2. Credit Event
    ("Moody's downgrades retailer to junk after missed payment", "Credit Event"),
    ("Sovereign debtor enters Chapter 11 bankruptcy restructuring after default", "Credit Event"),
    # 3. Monetary Policy
    ("Fed raises rates by 75 bps", "Monetary Policy"),
    ("ECB announces surprise 50 basis points rate hike", "Monetary Policy"),
    # 4. Merger/Acquisition
    ("Pfizer to acquire biotech firm for 4 billion dollars", "Merger/Acquisition"),
    ("Private equity firm launches hostile bid in multi-billion dollar takeover", "Merger/Acquisition"),
    # 5. Cyber/Operational
    ("Ransomware cyber attack paralyzes banking operations", "Cyber/Operational"),
    # 6. Natural Disaster
    ("Hurricane causes severe flooding and damage across regional refineries", "Natural Disaster"),
    # 7. Macroeconomic
    ("Persistent CPI inflation pushes economy toward stagflation", "Macroeconomic"),
    # 8. Regulatory/Legal
    ("DOJ launches antitrust probe into merger agreement", "Regulatory/Legal"),
    # 9. Earnings/Guidance
    ("Chipmaker issues severe profit warning and slashes FY guidance", "Earnings/Guidance"),
    # 10. Product Launch
    ("Apple unveils new flagship iPhone with breakthrough satellite connectivity", "Product Launch"),
    # 11. Supply Chain/Commodity
    ("Global semiconductor shortage and port strike halt manufacturing lines", "Supply Chain/Commodity"),
    # 12. Other / Low confidence
    ("Quiet trading session with mixed price action across indices", "Other"),
]


@pytest.fixture
def classifier() -> EventClassifier:
    """Fixture returning EventClassifier instance in fast, network-isolated rule mode."""
    return EventClassifier(use_zero_shot=False)


@pytest.mark.parametrize("headline,expected_label", TEST_HEADLINES)
def test_headline_event_classification(
    classifier: EventClassifier,
    headline: str,
    expected_label: str,
):
    """Validates classification across at least 11 categories with expected taxonomy labels."""
    result = classifier.classify(headline)
    assert result.event_type == expected_label, (
        f"Headline: '{headline}' classified as '{result.event_type}', expected '{expected_label}' "
        f"(confidence: {result.confidence}, method: {result.method}, matched: {result.matched_keywords})"
    )
    assert result.confidence >= 0.0


def test_rule_scores_and_keyword_extraction(classifier: EventClassifier):
    """Validates rule_scores returns normalized dict and matched pattern strings."""
    text = "Fed announces rate hike while Moody's warns of corporate default"
    scores, matched_kws = classifier.rule_scores(text)

    assert "Monetary Policy" in scores
    assert "Credit Event" in scores
    assert scores["Monetary Policy"] > 0.5
    assert scores["Credit Event"] > 0.5

    assert "rate hike" in matched_kws["Monetary Policy"]
    assert "default" in matched_kws["Credit Event"]


def test_confidence_floor_defaults_to_other(classifier: EventClassifier):
    """Verifies that headlines with no risk indicators fall below 0.35 and return 'Other'."""
    text = "Sunny skies and pleasant temperatures reported across town today."
    result = classifier.classify(text)
    assert result.event_type == "Other"


def test_labelled_events_csv_exists_and_valid():
    """Validates that starter data/labelled_events.csv exists and has 12 valid labels."""
    csv_path = BASE_DIR / "data" / "labelled_events.csv"
    assert csv_path.is_file(), "data/labelled_events.csv not found"

    events_cfg = get_events_config()
    valid_labels = set(events_cfg.events.keys())

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 12
    found_labels = set()
    for row in rows:
        assert row["text"]
        assert row["label"] in valid_labels, f"Unknown label '{row['label']}' in CSV"
        found_labels.add(row["label"])

    assert len(found_labels) == 12, f"CSV should cover all 12 taxonomy labels (found {len(found_labels)})"
