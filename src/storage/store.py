"""Storage subsystem providing dual persistence to SQLite and append-only JSONL."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas import RiskSignal
from src.config import BASE_DIR, logger

DEFAULT_DB_PATH = BASE_DIR / "data" / "signals.db"
DEFAULT_JSONL_PATH = BASE_DIR / "data" / "signals.jsonl"


class SignalStore:
    """Manages SQLite tables and append-only JSONL event log for RiskSignals."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        jsonl_path: Path = DEFAULT_JSONL_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        """Initializes tables and query performance indexes."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    url TEXT,
                    sentiment_score REAL NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_confidence REAL NOT NULL,
                    impact_score REAL NOT NULL,
                    impact_components_json TEXT NOT NULL,
                    impact_rationale TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    tickers_json TEXT NOT NULL,
                    countries_json TEXT NOT NULL,
                    sectors_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_tickers (
                    signal_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    PRIMARY KEY (signal_id, ticker),
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_signals_event_type ON signals (event_type);
                CREATE INDEX IF NOT EXISTS idx_signals_impact_score ON signals (impact_score DESC);
                CREATE INDEX IF NOT EXISTS idx_signals_source_type ON signals (source_type);
                CREATE INDEX IF NOT EXISTS idx_signal_tickers_ticker ON signal_tickers (ticker);
                """
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM signals")
            if self.db_path == DEFAULT_DB_PATH and cursor.fetchone()[0] == 0:
                sample_file = BASE_DIR / "data" / "signals_sample.jsonl"
                if sample_file.exists():
                    self._seed_from_file(conn, sample_file)
            conn.commit()

    def _seed_from_file(self, conn: sqlite3.Connection, sample_file: Path) -> None:
        """Seeds signals and join table from sample JSONL file."""
        signals_to_save: List[RiskSignal] = []
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        signals_to_save.append(RiskSignal.model_validate(json.loads(line)))
            if signals_to_save:
                logger.info(f"Auto-seeding {len(signals_to_save)} baseline signals from {sample_file}...")
                now_iso = datetime.now(timezone.utc).isoformat()
                for s in signals_to_save:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            s.signal_id,
                            s.timestamp.isoformat(),
                            s.source,
                            s.source_type,
                            s.headline,
                            s.url,
                            s.sentiment_score,
                            s.sentiment_label,
                            s.event_type,
                            s.event_confidence,
                            s.impact_score,
                            json.dumps(s.impact_components),
                            s.impact_rationale,
                            s.model_version,
                            json.dumps(s.tickers),
                            json.dumps(s.countries),
                            json.dumps(s.sectors),
                            now_iso,
                        ),
                    )
                    for t in s.tickers:
                        conn.execute("INSERT OR IGNORE INTO signal_tickers VALUES (?, ?)", (s.signal_id, t))
        except Exception as exc:
            logger.warning(f"Could not auto-seed baseline signals: {exc}")

    def _row_to_signal(self, row: sqlite3.Row) -> RiskSignal:
        """Converts SQLite Row back into validated RiskSignal model."""
        ts = datetime.fromisoformat(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return RiskSignal(
            signal_id=row["signal_id"],
            timestamp=ts,
            source=row["source"],
            source_type=row["source_type"],
            headline=row["headline"],
            url=row["url"],
            tickers=json.loads(row["tickers_json"]),
            countries=json.loads(row["countries_json"]),
            sectors=json.loads(row["sectors_json"]),
            sentiment_score=float(row["sentiment_score"]),
            sentiment_label=row["sentiment_label"],
            event_type=row["event_type"],
            event_confidence=float(row["event_confidence"]),
            impact_score=float(row["impact_score"]),
            impact_components=json.loads(row["impact_components_json"]),
            impact_rationale=row["impact_rationale"],
            model_version=row["model_version"],
        )

    def save(self, signals: List[RiskSignal]) -> int:
        """Idempotently saves signals to SQLite and appends new records to JSONL.

        Returns:
            Number of newly inserted unique signals.
        """
        if not signals:
            return 0

        now_iso = datetime.now(timezone.utc).isoformat()
        inserted_count = 0
        newly_saved_signals: List[RiskSignal] = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for s in signals:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO signals (
                        signal_id, timestamp, source, source_type, headline, url,
                        sentiment_score, sentiment_label, event_type, event_confidence,
                        impact_score, impact_components_json, impact_rationale,
                        model_version, tickers_json, countries_json, sectors_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.signal_id,
                        s.timestamp.isoformat(),
                        s.source,
                        s.source_type,
                        s.headline,
                        s.url,
                        s.sentiment_score,
                        s.sentiment_label,
                        s.event_type,
                        s.event_confidence,
                        s.impact_score,
                        json.dumps(s.impact_components),
                        s.impact_rationale,
                        s.model_version,
                        json.dumps(s.tickers),
                        json.dumps(s.countries),
                        json.dumps(s.sectors),
                        now_iso,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted_count += 1
                    newly_saved_signals.append(s)
                    for t in s.tickers:
                        cursor.execute(
                            "INSERT OR IGNORE INTO signal_tickers (signal_id, ticker) VALUES (?, ?)",
                            (s.signal_id, t.upper()),
                        )
            conn.commit()

        # Append newly inserted signals to JSONL
        if newly_saved_signals:
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    for sig in newly_saved_signals:
                        f.write(sig.model_dump_json() + "\n")
            except Exception as exc:
                logger.error(f"Failed to append signals to JSONL file: {exc}")

        return inserted_count

    def query(
        self,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        min_impact: Optional[float] = None,
        source_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RiskSignal]:
        """Queries signals with multi-parameter filtering."""
        query_parts = ["SELECT s.* FROM signals s"]
        params: List[Any] = []

        if ticker:
            query_parts.append("JOIN signal_tickers st ON s.signal_id = st.signal_id")

        where_clauses: List[str] = []

        if ticker:
            where_clauses.append("st.ticker = ?")
            params.append(ticker.upper().strip())

        if event_type:
            where_clauses.append("s.event_type = ?")
            params.append(event_type.strip())

        if min_impact is not None:
            where_clauses.append("s.impact_score >= ?")
            params.append(float(min_impact))

        if source_type:
            where_clauses.append("s.source_type = ?")
            params.append(source_type.strip().lower())

        if since:
            since_iso = since.isoformat()
            where_clauses.append("s.timestamp >= ?")
            params.append(since_iso)

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        query_parts.append("ORDER BY s.timestamp DESC LIMIT ?")
        params.append(max(1, min(1000, limit)))

        sql = " ".join(query_parts)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_signal(r) for r in rows]

    def latest(self, n: int = 20) -> List[RiskSignal]:
        """Returns the n most recent signals."""
        return self.query(limit=n)

    def summary(self) -> Dict[str, Any]:
        """Calculates aggregated metrics across all stored signals."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            # By event type
            cursor.execute("SELECT event_type, COUNT(*) FROM signals GROUP BY event_type ORDER BY COUNT(*) DESC")
            by_event = {row[0]: row[1] for row in cursor.fetchall()}

            # By source type
            cursor.execute("SELECT source_type, COUNT(*) FROM signals GROUP BY source_type")
            by_source = {row[0]: row[1] for row in cursor.fetchall()}

            # Average sentiment per ticker
            cursor.execute(
                """
                SELECT st.ticker, AVG(s.sentiment_score) as avg_sent, COUNT(s.signal_id) as cnt
                FROM signal_tickers st
                JOIN signals s ON st.signal_id = s.signal_id
                GROUP BY st.ticker
                ORDER BY cnt DESC
                """
            )
            avg_sent_ticker = {row[0]: round(float(row[1]), 4) for row in cursor.fetchall()}

        # Top 5 highest impact signals
        highest_impact = self.query(min_impact=7.0, limit=5)
        if not highest_impact:
            # Fallback to top 5 by impact score
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM signals ORDER BY impact_score DESC LIMIT 5")
                highest_impact = [self._row_to_signal(r) for r in cursor.fetchall()]

        return {
            "total_signals": total_signals,
            "by_event_type": by_event,
            "by_source_type": by_source,
            "avg_sentiment_per_ticker": avg_sent_ticker,
            "highest_impact": [s.model_dump() for s in highest_impact],
        }

    def delete_injected(self, sources: Tuple[str, ...] = ("ad_hoc", "demo_injected")) -> int:
        """Deletes ad-hoc injected demo signals from the SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in sources)
            cursor.execute(f"SELECT signal_id FROM signals WHERE source IN ({placeholders})", sources)
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                id_placeholders = ",".join("?" for _ in ids)
                cursor.execute(f"DELETE FROM signal_tickers WHERE signal_id IN ({id_placeholders})", ids)
                cursor.execute(f"DELETE FROM signals WHERE signal_id IN ({id_placeholders})", ids)
                conn.commit()
            return len(ids)


# Global store instance
_STORE: Optional[SignalStore] = None


def get_signal_store(
    db_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> SignalStore:
    """Returns singleton SignalStore instance."""
    global _STORE
    if _STORE is None or db_path is not None:
        _STORE = SignalStore(
            db_path=db_path or DEFAULT_DB_PATH,
            jsonl_path=jsonl_path or DEFAULT_JSONL_PATH,
        )
    return _STORE


def save(signals: List[RiskSignal]) -> int:
    return get_signal_store().save(signals)


def query(
    ticker: Optional[str] = None,
    event_type: Optional[str] = None,
    min_impact: Optional[float] = None,
    source_type: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 100,
) -> List[RiskSignal]:
    return get_signal_store().query(
        ticker=ticker,
        event_type=event_type,
        min_impact=min_impact,
        source_type=source_type,
        since=since,
        limit=limit,
    )


def latest(n: int = 20) -> List[RiskSignal]:
    return get_signal_store().latest(n=n)


def summary() -> Dict[str, Any]:
    return get_signal_store().summary()
