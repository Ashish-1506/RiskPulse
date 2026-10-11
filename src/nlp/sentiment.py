"""Sentiment analysis engine with FinBERT transformer, VADER fallback, and SQLite caching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.config import BASE_DIR, get_config, logger

# SQLite cache database location
CACHE_DB_PATH = BASE_DIR / "data" / "sentiment_cache.db"


class SentimentResult(BaseModel):
    """Standard output schema for sentiment evaluation."""

    text: str
    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["positive", "neutral", "negative"]
    probs: Dict[str, float]
    backend: str


class SentimentCache:
    """SQLite-backed cache for sentiment predictions keyed by SHA-1 hash of input text."""

    def __init__(self, db_path: Path = CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    text_hash TEXT NOT NULL,
                    text_preview TEXT,
                    score REAL NOT NULL,
                    label TEXT NOT NULL,
                    probs_json TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (text_hash, backend)
                )
                """
            )
            conn.commit()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str, backend: Optional[str] = None) -> Optional[SentimentResult]:
        key = self.hash_text(text)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if backend:
                cursor.execute(
                    "SELECT score, label, probs_json, backend FROM sentiment_cache WHERE text_hash = ? AND backend = ?",
                    (key, backend.lower()),
                )
            else:
                cursor.execute(
                    "SELECT score, label, probs_json, backend FROM sentiment_cache WHERE text_hash = ?",
                    (key,),
                )
            row = cursor.fetchone()
            if row:
                score, label, probs_json, be = row
                return SentimentResult(
                    text=text,
                    score=float(score),
                    label=label,
                    probs=json.loads(probs_json),
                    backend=f"{be}_cached",
                )
        return None

    def set(self, result: SentimentResult) -> None:
        key = self.hash_text(result.text)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sentiment_cache
                (text_hash, text_preview, score, label, probs_json, backend, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    result.text[:100],
                    result.score,
                    result.label,
                    json.dumps(result.probs),
                    result.backend.replace("_cached", "").lower(),
                    now_iso,
                ),
            )
            conn.commit()


class SentimentAnalyzer:
    """Unified Sentiment Analyzer supporting FinBERT with automatic VADER fallback."""

    _instance: Optional["SentimentAnalyzer"] = None
    _finbert_pipeline = None
    _vader_analyzer = None

    def __init__(self, backend_override: Optional[str] = None) -> None:
        self.cfg = get_config()
        self.requested_backend = (backend_override or self.cfg.sentiment.backend).lower()
        self.neutral_band = self.cfg.sentiment.neutral_band
        self.batch_size = self.cfg.sentiment.batch_size
        self.cache = SentimentCache()
        self.active_backend = self.requested_backend

        # Device determination (graceful fallback if torch is not installed)
        try:
            import torch

            self.device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            self.device = -1

    def _get_vader_analyzer(self):
        """Initializes VADER with financial lexicon enhancement."""
        if SentimentAnalyzer._vader_analyzer is None:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                SentimentAnalyzer._vader_analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                logger.info("Downloading NLTK vader_lexicon on demand...")
                nltk.download("vader_lexicon", quiet=True)
                SentimentAnalyzer._vader_analyzer = SentimentIntensityAnalyzer()

            # Domain-specific financial lexicon boost
            finance_boost = {
                "default": -3.5,
                "bankruptcy": -3.8,
                "downgrade": -3.0,
                "downgrades": -3.0,
                "downgraded": -3.0,
                "probe": -2.5,
                "plunge": -3.2,
                "plunges": -3.2,
                "plunged": -3.2,
                "slump": -2.8,
                "crash": -3.5,
                "fraud": -3.8,
                "subpoena": -2.5,
                "insolvency": -3.6,
                "beat": 3.0,
                "beats": 3.0,
                "upgrade": 3.0,
                "upgrades": 3.0,
                "upgraded": 3.0,
                "record": 2.5,
                "surge": 3.2,
                "surges": 3.2,
                "surged": 3.2,
                "rally": 2.8,
                "blowout": 3.5,
                "outperform": 3.0,
                "breakthrough": 3.2,
                "approves": 2.8,
                "approval": 2.8,
                "efficacy": 2.5,
                "revolutionary": 2.5,
            }
            SentimentAnalyzer._vader_analyzer.lexicon.update(finance_boost)
        return SentimentAnalyzer._vader_analyzer

    def _get_finbert_pipeline(self):
        """Lazy-loads ProsusAI/finbert pipeline once (singleton)."""
        if SentimentAnalyzer._finbert_pipeline is None:
            start_t = time.perf_counter()
            logger.info("Loading FinBERT model ('ProsusAI/finbert')...")
            try:
                from transformers import pipeline

                pipe = pipeline(
                    task="text-classification",
                    model=self.cfg.sentiment.model_name,
                    top_k=None,
                    truncation=True,
                    max_length=256,
                    device=self.device,
                )
                SentimentAnalyzer._finbert_pipeline = pipe
                elapsed = time.perf_counter() - start_t
                logger.info(f"FinBERT loaded successfully in {elapsed:.2f}s on device: {self.device}.")
            except Exception as exc:
                logger.warning(f"FinBERT failed to load ({exc}). Falling back automatically to VADER.")
                self.active_backend = "vader"
                return None

        return SentimentAnalyzer._finbert_pipeline

    def _classify_score(self, score: float) -> Literal["positive", "neutral", "negative"]:
        """Assigns categorical label based on configured neutral band."""
        if score > self.neutral_band:
            return "positive"
        elif score < -self.neutral_band:
            return "negative"
        return "neutral"

    def _analyze_with_finbert(self, texts: List[str]) -> List[SentimentResult]:
        """Runs FinBERT inference over texts in batches."""
        pipe = self._get_finbert_pipeline()
        if pipe is None:
            return self._analyze_with_vader(texts)

        results: List[SentimentResult] = []
        raw_outputs = pipe(texts, batch_size=self.batch_size)

        for text, item in zip(texts, raw_outputs):
            # item is list of dicts: [{'label': 'positive', 'score': 0.8}, ...]
            probs = {entry["label"].lower(): float(entry["score"]) for entry in item}
            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            p_neu = probs.get("neutral", 0.0)

            score = max(-1.0, min(1.0, p_pos - p_neg))
            label = self._classify_score(score)

            results.append(
                SentimentResult(
                    text=text,
                    score=round(score, 4),
                    label=label,
                    probs={
                        "positive": round(p_pos, 4),
                        "neutral": round(p_neu, 4),
                        "negative": round(p_neg, 4),
                    },
                    backend="finbert",
                )
            )

        return results

    def _analyze_with_vader(self, texts: List[str]) -> List[SentimentResult]:
        """Runs rule-based enhanced VADER inference."""
        sia = self._get_vader_analyzer()
        results: List[SentimentResult] = []

        for text in texts:
            scores = sia.polarity_scores(text)
            compound = float(scores["compound"])
            label = self._classify_score(compound)

            results.append(
                SentimentResult(
                    text=text,
                    score=round(compound, 4),
                    label=label,
                    probs={
                        "positive": round(float(scores["pos"]), 4),
                        "neutral": round(float(scores["neu"]), 4),
                        "negative": round(float(scores["neg"]), 4),
                    },
                    backend="vader",
                )
            )

        return results

    def analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyzes sentiment for a list of texts, checking SQLite cache first.

        Logs throughput at INFO level.
        """
        if not texts:
            return []

        start_time = time.perf_counter()
        final_results: List[Optional[SentimentResult]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # 1. Lookup cache
        for idx, text in enumerate(texts):
            cached_res = self.cache.get(text, backend=self.requested_backend)
            if cached_res is not None:
                final_results[idx] = cached_res
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # 2. Run active backend on uncached items
        if uncached_texts:
            if self.requested_backend == "vader":
                new_results = self._analyze_with_vader(uncached_texts)
            else:
                new_results = self._analyze_with_finbert(uncached_texts)

            for idx, res in zip(uncached_indices, new_results):
                final_results[idx] = res
                self.cache.set(res)

        elapsed = time.perf_counter() - start_time
        throughput = len(texts) / max(elapsed, 1e-6)
        logger.info(
            f"Sentiment evaluation completed: {len(texts)} texts in {elapsed:.3f}s ({throughput:.1f} texts/sec) using {self.active_backend}."
        )

        return [res for res in final_results if res is not None]


# Global singleton instance
_ANALYZER: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer(backend: Optional[str] = None) -> SentimentAnalyzer:
    """Returns or creates the singleton SentimentAnalyzer instance."""
    global _ANALYZER
    if _ANALYZER is None or (backend and _ANALYZER.requested_backend != backend):
        _ANALYZER = SentimentAnalyzer(backend_override=backend)
    return _ANALYZER
