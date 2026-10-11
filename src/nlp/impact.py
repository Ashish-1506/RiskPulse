"""Multi-factor market impact severity scoring engine."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.config import get_config, get_events_config

# Canonical high-intensity market keywords
HIGH_INTENSITY_KEYWORDS: Set[str] = {
    "default",
    "war",
    "invasion",
    "collapse",
    "bankruptcy",
    "sanctions",
    "crash",
    "emergency",
    "plunge",
    "record",
    "billion",
    "trillion",
    "systemic",
    "contagion",
    "clash",
    "crisis",
    "hike",
    "inflation",
    "surge",
    "threatened",
    "warn",
}


class ImpactResult(BaseModel):
    """Output schema for market impact severity evaluation."""

    impact_score: float = Field(ge=1.0, le=10.0)
    components: Dict[str, Any]
    rationale: str


class CorroborationTracker:
    """Sliding-window corroboration tracker clustering events reported across distinct sources."""

    def __init__(self, window_hours: float = 6.0, jaccard_threshold: float = 0.50) -> None:
        self.window_hours = window_hours
        self.jaccard_threshold = jaccard_threshold
        # Stores tuples: (timestamp, event_type, token_set, source_identifier, tickers_set, countries_set)
        self.history: List[Tuple[datetime, str, Set[str], str, Set[str], Set[str]]] = []

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
        return set(words)

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def prune(self, current_time: datetime) -> None:
        """Removes records older than the sliding window."""
        cutoff = current_time - timedelta(hours=self.window_hours)
        self.history = [item for item in self.history if item[0] >= cutoff]

    def add_and_compute(
        self,
        event_type: str,
        title: str,
        source: str,
        timestamp: Optional[datetime] = None,
        tickers: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
    ) -> Tuple[float, int]:
        """Registers an event and computes corroboration score based on distinct source count.

        Returns:
            Tuple of (corroboration_score in [0, 1], n_distinct_sources)
        """
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        self.prune(now)

        tokens = self._tokenize(title)
        tickers_set = set(tickers or [])
        countries_set = set(countries or [])
        source_id = source.strip().lower()

        distinct_sources: Set[str] = {source_id}

        for hist_ts, hist_event, hist_tokens, hist_src, hist_tickers, hist_countries in self.history:
            if hist_event != event_type:
                continue

            # Check title token similarity
            similarity = self._jaccard(tokens, hist_tokens)
            shared_entities = bool((tickers_set & hist_tickers) or (countries_set & hist_countries))

            # Match if high text overlap or moderate overlap with shared entities
            if similarity >= self.jaccard_threshold or (similarity >= 0.35 and shared_entities):
                distinct_sources.add(hist_src)

        # Append current event to history
        self.history.append((now, event_type, tokens, source_id, tickers_set, countries_set))

        n_distinct = len(distinct_sources)
        # corroboration = min(1.0, (n_distinct - 1) / 4)
        corroboration = min(1.0, max(0.0, (n_distinct - 1) / 4.0))
        return round(corroboration, 4), n_distinct

    def clear(self) -> None:
        """Clears in-memory sliding window history."""
        self.history.clear()


class ImpactScorer:
    """Calculates multi-factor market impact severity scores in [1.0, 10.0]."""

    def __init__(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
        tracker: Optional[CorroborationTracker] = None,
    ) -> None:
        self.cfg = get_config()
        self.events_cfg = get_events_config()
        self.tracker = tracker or CorroborationTracker()

        # Load weights
        default_weights = {
            "event_prior": self.cfg.impact.weights.event_prior,
            "sentiment": self.cfg.impact.weights.sentiment,
            "keywords": self.cfg.impact.weights.keywords,
            "source": self.cfg.impact.weights.source,
            "corroboration": self.cfg.impact.weights.corroboration,
        }
        self.weights = custom_weights or default_weights
        self.source_credibility_map = self.cfg.impact.source_credibility

    def _get_event_prior(self, event_type: str) -> float:
        """Looks up event prior severity from events taxonomy."""
        event_meta = self.events_cfg.events.get(event_type)
        if event_meta:
            return float(event_meta.prior)
        return 0.30

    def _get_keyword_intensity(self, text: str) -> Tuple[float, int]:
        """Calculates keyword intensity in [0, 1] using 1 - exp(-0.7 * count)."""
        words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        matched_count = len(words & HIGH_INTENSITY_KEYWORDS)
        intensity = 1.0 - math.exp(-0.7 * matched_count)
        return round(min(1.0, max(0.0, intensity)), 4), matched_count

    def _get_source_credibility(self, source: str, domain: Optional[str] = None) -> float:
        """Determines source credibility weight from config map."""
        src_clean = source.strip().lower()
        if domain:
            dom_clean = domain.strip().lower()
            for key, cred in self.source_credibility_map.items():
                if key in dom_clean:
                    return cred

        for key, cred in self.source_credibility_map.items():
            if key in src_clean:
                return cred

        # Fallback values
        if src_clean in ("social", "tweets", "reddit", "stocktwits", "twitter"):
            return 0.40
        return 0.60

    def _generate_rationale(
        self,
        event_type: str,
        event_prior: float,
        sentiment_score: float,
        kw_count: int,
        n_distinct: int,
    ) -> str:
        """Constructs a concise, informative one-sentence impact explanation."""
        sent_desc = (
            f"severe negative tone ({sentiment_score:+.2f})"
            if sentiment_score < -0.40
            else f"strong positive tone ({sentiment_score:+.2f})"
            if sentiment_score > 0.40
            else f"moderate sentiment ({sentiment_score:+.2f})"
        )
        sources_desc = f"{n_distinct} source(s) agree"
        kw_desc = f"{kw_count} high-severity trigger(s)"

        return f"{event_type} prior ({event_prior:.2f}); {sent_desc}; {kw_desc}; {sources_desc}."

    def score(
        self,
        event_type: str,
        sentiment_score: float,
        text: str,
        source: str,
        title: Optional[str] = None,
        domain: Optional[str] = None,
        url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        tickers: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
    ) -> ImpactResult:
        """Computes multi-factor market impact severity score."""
        eval_title = title or (text[:100] if text else "")

        # Auto-extract domain from url if domain was not provided
        if url and not domain:
            from urllib.parse import urlparse

            try:
                domain = urlparse(url).netloc
            except Exception:
                domain = None

        # 1. Event prior
        prior_val = self._get_event_prior(event_type)

        # 2. Sentiment magnitude
        sent_intensity = min(1.0, max(0.0, abs(sentiment_score)))

        # 3. Keyword intensity
        kw_intensity, kw_count = self._get_keyword_intensity(f"{eval_title} {text}")

        # 4. Source credibility
        src_cred = self._get_source_credibility(source=source, domain=domain)

        # 5. Corroboration
        corr_val, n_distinct = self.tracker.add_and_compute(
            event_type=event_type,
            title=eval_title,
            source=source,
            timestamp=timestamp,
            tickers=tickers,
            countries=countries,
        )

        # Weighted raw calculation
        w = self.weights
        raw = (
            w.get("event_prior", 0.40) * prior_val
            + w.get("sentiment", 0.25) * sent_intensity
            + w.get("keywords", 0.15) * kw_intensity
            + w.get("source", 0.10) * src_cred
            + w.get("corroboration", 0.10) * corr_val
        )

        clipped_raw = min(1.0, max(0.0, raw))
        impact_score = round(1.0 + 9.0 * clipped_raw, 1)

        components = {
            "event_prior": round(prior_val, 4),
            "sentiment_intensity": round(sent_intensity, 4),
            "keyword_intensity": round(kw_intensity, 4),
            "source_credibility": round(src_cred, 4),
            "corroboration": round(corr_val, 4),
            "n_distinct_sources": n_distinct,
            "raw_score": round(raw, 4),
            "weights": {k: float(v) for k, v in w.items()},
        }

        rationale = self._generate_rationale(
            event_type=event_type,
            event_prior=prior_val,
            sentiment_score=sentiment_score,
            kw_count=kw_count,
            n_distinct=n_distinct,
        )

        return ImpactResult(
            impact_score=impact_score,
            components=components,
            rationale=rationale,
        )


# Global singleton instance
_IMPACT_SCORER: Optional[ImpactScorer] = None


def get_impact_scorer() -> ImpactScorer:
    """Returns singleton ImpactScorer instance."""
    global _IMPACT_SCORER
    if _IMPACT_SCORER is None:
        _IMPACT_SCORER = ImpactScorer()
    return _IMPACT_SCORER
