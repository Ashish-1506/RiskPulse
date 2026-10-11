"""Unit tests for the multi-factor market impact scoring engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.nlp.impact import CorroborationTracker, ImpactScorer


@pytest.fixture
def scorer() -> ImpactScorer:
    """Fixture returning a fresh ImpactScorer with isolated tracker."""
    tracker = CorroborationTracker(window_hours=6.0)
    return ImpactScorer(tracker=tracker)


# ------------------------------------------------------------------------------
# (a) Monotonicity Test
# ------------------------------------------------------------------------------
def test_impact_monotonicity(scorer: ImpactScorer):
    """Validates that a catastrophic credit/default event scores higher than routine product launch."""
    res_severe = scorer.score(
        event_type="Credit Event",
        sentiment_score=-0.88,
        text="Sovereign default and debt collapse triggers emergency bankruptcy for financial institutions.",
        source="reuters",
        domain="reuters.com",
    )

    res_routine = scorer.score(
        event_type="Product Launch",
        sentiment_score=0.25,
        text="Company announces routine product launch featuring upgraded accessories.",
        source="tweets",
    )

    assert res_severe.impact_score > res_routine.impact_score
    # Severe event should typically reach at least 7.0 (trigger threshold)
    assert res_severe.impact_score >= 6.5
    assert res_routine.impact_score <= 4.5


# ------------------------------------------------------------------------------
# (b) Bounds [1.0, 10.0] Test
# ------------------------------------------------------------------------------
def test_impact_score_bounds_and_extremes(scorer: ImpactScorer):
    """Validates impact scores strictly remain within [1.0, 10.0] under extreme values."""
    # Minimum edge case
    res_min = scorer.score(
        event_type="Other",
        sentiment_score=0.0,
        text="The weather was calm and unremarkable.",
        source="unknown",
    )
    assert 1.0 <= res_min.impact_score <= 10.0

    # Maximum edge case
    res_max = scorer.score(
        event_type="Credit Event",
        sentiment_score=-1.0,
        text="Catastrophic trillion dollar default, war, systemic collapse, contagion, bankruptcy, crash!",
        source="bloomberg",
        domain="bloomberg.com",
    )
    assert 1.0 <= res_max.impact_score <= 10.0
    assert res_max.impact_score >= 8.5


# ------------------------------------------------------------------------------
# (c) Corroboration Scaling Test
# ------------------------------------------------------------------------------
def test_corroboration_increases_impact_score():
    """Validates that receiving the same story from multiple distinct sources raises the impact score."""
    tracker = CorroborationTracker(window_hours=6.0)
    scorer = ImpactScorer(tracker=tracker)

    title = "Major cyber attack cripples national payment clearing network"
    text = "Ransomware operation shuts down transaction processing."
    event = "Cyber/Operational"
    now = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)

    # 1st source: Reuters
    res1 = scorer.score(event, -0.7, text, source="reuters", title=title, timestamp=now)
    assert res1.components["corroboration"] == 0.0
    assert res1.components["n_distinct_sources"] == 1

    # 2nd source: Bloomberg
    res2 = scorer.score(event, -0.7, text, source="bloomberg", title=title, timestamp=now)
    assert res2.components["corroboration"] == 0.25
    assert res2.components["n_distinct_sources"] == 2
    assert res2.impact_score >= res1.impact_score

    # 3rd source: CNBC
    res3 = scorer.score(event, -0.7, text, source="cnbc", title=title, timestamp=now)
    assert res3.components["corroboration"] == 0.50
    assert res3.components["n_distinct_sources"] == 3

    # 4th source: Financial Times
    res4 = scorer.score(event, -0.7, text, source="ft", title=title, timestamp=now)
    assert res4.components["corroboration"] == 0.75
    assert res4.components["n_distinct_sources"] == 4

    # 5th source: GDELT
    res5 = scorer.score(event, -0.7, text, source="gdelt", title=title, timestamp=now)
    assert res5.components["corroboration"] == 1.0
    assert res5.components["n_distinct_sources"] == 5

    # Impact score with 5 sources must strictly exceed 1 source
    assert res5.impact_score > res1.impact_score


# ------------------------------------------------------------------------------
# (d) Config Weights Sensitivity Test
# ------------------------------------------------------------------------------
def test_custom_weight_sensitivity():
    """Validates that changing component weights shifts the output predictably."""
    # Scorer A: Heavy weight on sentiment (0.90), negligible on event prior (0.02)
    weights_sentiment_heavy = {
        "event_prior": 0.02,
        "sentiment": 0.90,
        "keywords": 0.02,
        "source": 0.03,
        "corroboration": 0.03,
    }
    scorer_sent = ImpactScorer(custom_weights=weights_sentiment_heavy)

    # Scorer B: Heavy weight on event prior (0.90), negligible on sentiment (0.02)
    weights_prior_heavy = {
        "event_prior": 0.90,
        "sentiment": 0.02,
        "keywords": 0.02,
        "source": 0.03,
        "corroboration": 0.03,
    }
    scorer_prior = ImpactScorer(custom_weights=weights_prior_heavy)

    # High prior (Geopolitical 0.80), but neutral sentiment (0.0)
    text = "Sanctions discussions ongoing regarding regional boundary."
    res_a = scorer_sent.score(event_type="Geopolitical", sentiment_score=0.0, text=text, source="gdelt")
    res_b = scorer_prior.score(event_type="Geopolitical", sentiment_score=0.0, text=text, source="gdelt")

    # Scorer B (prior heavy) must score substantially higher than Scorer A (sentiment heavy with 0 sentiment)
    assert res_b.impact_score > res_a.impact_score
    assert res_b.impact_score - res_a.impact_score >= 3.0


def test_impact_rationale_string(scorer: ImpactScorer):
    """Validates that a human-readable rationale is generated with key metrics."""
    res = scorer.score(
        event_type="Geopolitical",
        sentiment_score=-0.75,
        text="War and missile invasion halts energy infrastructure.",
        source="reuters",
    )
    assert res.rationale
    assert "Geopolitical" in res.rationale
    assert "negative tone" in res.rationale
