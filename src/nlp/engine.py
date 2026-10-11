"""Unified RiskEngine integrating preprocessing, entities, sentiment, events, and impact."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

from src.api.schemas import RiskSignal
from src.config import get_config, logger
from src.ingestion.normalise import NormalizedDoc
from src.nlp.entities import get_extractor
from src.nlp.events import get_event_classifier
from src.nlp.impact import get_impact_scorer
from src.nlp.preprocess import clean_text, is_english
from src.nlp.sentiment import get_sentiment_analyzer


class RiskEngine:
    """Core AI/NLP pipeline transforming normalized raw documents into structured RiskSignals."""

    def __init__(
        self,
        sentiment_backend: Optional[str] = None,
        use_zero_shot: bool = True,
    ) -> None:
        self.cfg = get_config()
        self.sentiment_analyzer = get_sentiment_analyzer(backend=sentiment_backend)
        self.event_classifier = get_event_classifier(use_zero_shot=use_zero_shot)
        self.impact_scorer = get_impact_scorer()
        self.entity_extractor = get_extractor()

    @property
    def model_version(self) -> str:
        """Constructs string identifier reflecting active NLP pipeline backends."""
        sent_be = self.sentiment_analyzer.active_backend
        events_mode = "zeroshot+rules" if self.event_classifier.use_zero_shot else "rules"
        return f"{sent_be}+{events_mode}-v1"

    def process_single(
        self,
        text: str,
        title: Optional[str] = None,
        source: str = "ad_hoc",
        source_type: str = "news",
        url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[RiskSignal]:
        """Convenience method for on-demand single text processing."""
        dt = timestamp or datetime.now(timezone.utc)
        doc = NormalizedDoc.create(
            source=source,
            source_type="social" if source_type == "social" else "news",
            published_at=dt,
            title=title or (text[:80] + "..."),
            text=text,
            url=url,
        )
        signals = self.process([doc])
        return signals[0] if signals else None

    def process(self, docs: List[NormalizedDoc]) -> List[RiskSignal]:
        """Transforms a batch of NormalizedDocs into validated RiskSignals."""
        if not docs:
            return []

        start_time = time.perf_counter()
        valid_docs: List[NormalizedDoc] = []
        cleaned_eval_texts: List[str] = []

        # 1. Clean & English filter
        for doc in docs:
            try:
                clean_title = clean_text(doc.title)
                clean_body = clean_text(doc.text)

                combined_raw = f"{clean_title}. {clean_body}".strip()
                if not combined_raw or len(combined_raw) < 10:
                    continue

                # Check language for news items (social text can be terse/informal)
                if doc.source_type == "news" and not is_english(combined_raw):
                    continue

                # Update doc with cleaned text
                doc.title = clean_title
                doc.text = clean_body
                valid_docs.append(doc)
                cleaned_eval_texts.append(combined_raw)
            except Exception as exc:
                logger.warning(f"Error filtering document {getattr(doc, 'doc_id', 'unknown')}: {exc}. Skipping.")
                continue

        if not valid_docs:
            logger.info("No documents passed text cleaning and language filtering.")
            return []

        # 2. Entity enrichment (Tickers, Countries, Sectors)
        for doc in valid_docs:
            try:
                self.entity_extractor.enrich(doc)
            except Exception as exc:
                logger.warning(f"Error enriching entities for doc {getattr(doc, 'doc_id', 'unknown')}: {exc}.")

        # 3. Batch Sentiment Analysis
        sentiment_results = self.sentiment_analyzer.analyze(cleaned_eval_texts)

        # 4. Event Classification
        event_results = [self.event_classifier.classify(text) for text in cleaned_eval_texts]

        # 5. Impact Scoring & RiskSignal construction
        signals: List[RiskSignal] = []
        model_ver = self.model_version

        for doc, eval_text, sent_res, ev_res in zip(valid_docs, cleaned_eval_texts, sentiment_results, event_results):
            try:
                # Evaluate impact severity
                impact_res = self.impact_scorer.score(
                    event_type=ev_res.event_type,
                    sentiment_score=sent_res.score,
                    text=eval_text,
                    source=doc.source,
                    title=doc.title,
                    url=doc.url,
                    timestamp=doc.published_at,
                    tickers=doc.tickers,
                    countries=doc.countries,
                )

                # Map sentiment label to uppercase schema Literal
                raw_sent_label = sent_res.label.upper()
                sent_label = (
                    "POSITIVE"
                    if raw_sent_label == "POSITIVE"
                    else "NEGATIVE"
                    if raw_sent_label == "NEGATIVE"
                    else "NEUTRAL"
                )

                signal = RiskSignal(
                    signal_id=doc.doc_id,
                    timestamp=doc.published_at,
                    source=doc.source,
                    source_type=doc.source_type,
                    headline=doc.title or eval_text[:100],
                    url=doc.url,
                    tickers=doc.tickers,
                    countries=doc.countries,
                    sectors=doc.sectors,
                    sentiment_score=sent_res.score,
                    sentiment_label=sent_label,
                    event_type=ev_res.event_type,
                    event_confidence=ev_res.confidence,
                    impact_score=impact_res.impact_score,
                    impact_components=impact_res.components,
                    impact_rationale=impact_res.rationale,
                    model_version=model_ver,
                )
                signals.append(signal)
            except Exception as exc:
                logger.warning(
                    f"Error constructing RiskSignal for doc {getattr(doc, 'doc_id', 'unknown')}: {exc}. Skipping."
                )
                continue

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"RiskEngine processed {len(signals)} signals from {len(docs)} input docs "
            f"in {elapsed:.3f}s ({len(signals) / max(elapsed, 1e-6):.1f} signals/sec)."
        )

        return signals


# Global singleton instance
_ENGINE: Optional[RiskEngine] = None


def get_risk_engine(sentiment_backend: Optional[str] = None, use_zero_shot: bool = True) -> RiskEngine:
    """Returns singleton RiskEngine instance."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RiskEngine(sentiment_backend=sentiment_backend, use_zero_shot=use_zero_shot)
    return _ENGINE
