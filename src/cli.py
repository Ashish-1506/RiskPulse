"""Command Line Interface for RiskPulse engine."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from src.config import BASE_DIR, get_config, logger
from src.ingestion.base import Source
from src.ingestion.gdelt import GdeltSource
from src.ingestion.newsapi import NewsApiSource
from src.ingestion.normalise import NormalizedDoc, dedupe
from src.ingestion.rss import RssSource
from src.ingestion.tweets_replay import TweetsReplaySource
from src.nlp.engine import get_risk_engine
from src.storage.store import get_signal_store


def _get_source_instance(source_name: str) -> Source:
    """Factory creating configured Source instance."""
    if source_name == "gdelt":
        return GdeltSource()
    elif source_name == "newsapi":
        return NewsApiSource()
    elif source_name == "rss":
        return RssSource()
    elif source_name == "tweets":
        return TweetsReplaySource()
    else:
        raise ValueError(f"Unknown source: {source_name}")


def handle_ingest(args: argparse.Namespace) -> None:
    """Executes ingestion workflow for specified source."""
    source_name = args.source.lower()
    days = args.days
    limit = args.limit

    since = datetime.now(timezone.utc) - timedelta(days=days) if days > 0 else None
    logger.info(f"Ingesting from source '{source_name}' (since: {since}, limit: {limit})...")

    source = _get_source_instance(source_name)
    raw_docs = source.fetch(since=since, limit=limit)
    docs = dedupe(raw_docs)

    print("\n" + "=" * 60)
    print(f"INGESTION SUMMARY: Source='{source_name}'")
    print(f"Raw Documents Fetched: {len(raw_docs)}")
    print(f"Unique Deduplicated:   {len(docs)}")
    print("=" * 60)

    if not docs:
        print("No documents found matching criteria.")
        return

    sample_size = min(3, len(docs))
    print(f"\nDisplaying {sample_size} sample document(s):\n")
    for i, doc in enumerate(docs[:sample_size], start=1):
        print(f"--- Example #{i} ---")
        print(f"  ID:         {doc.doc_id}")
        print(f"  Timestamp:  {doc.published_at.isoformat()}")
        print(f"  Source:     {doc.source} ({doc.source_type})")
        print(f"  Ticker Hint: {doc.ticker_hint or 'None'}")
        print(f"  Title:      {doc.title}")
        print(f"  Snippet:    {doc.text[:120]}...\n")


def handle_sentiment(args: argparse.Namespace) -> None:
    """Executes sentiment evaluation for text argument."""
    from src.nlp.sentiment import get_sentiment_analyzer

    analyzer = get_sentiment_analyzer(backend=args.backend)
    results = analyzer.analyze([args.text])
    if not results:
        print("No result generated.")
        return

    res = results[0]
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS RESULT")
    print("=" * 60)
    print(f"Text:         {res.text}")
    print(f"Score:        {res.score:+.4f} (range: -1.0 to +1.0)")
    print(f"Label:        {res.label.upper()}")
    print(
        f"Probabilities: Positive: {res.probs.get('positive', 0.0):.4f} | "
        f"Neutral: {res.probs.get('neutral', 0.0):.4f} | "
        f"Negative: {res.probs.get('negative', 0.0):.4f}"
    )
    print(f"Backend:      {res.backend}")
    print("=" * 60 + "\n")


def _load_starter_news_docs() -> List[NormalizedDoc]:
    """Converts labelled events and news headlines into NormalizedDoc batch."""
    docs: List[NormalizedDoc] = []
    events_csv = BASE_DIR / "data" / "labelled_events.csv"
    if events_csv.is_file():
        now = datetime.now(timezone.utc)
        with open(events_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get("text", "").strip()
                if not text:
                    continue
                # Offset timestamps slightly for realistic time series
                ts = now - timedelta(hours=i * 2 + 1)
                docs.append(
                    NormalizedDoc.create(
                        source="newsapi",
                        source_type="news",
                        published_at=ts,
                        title=text,
                        text=text,
                    )
                )
    return docs


def handle_run(args: argparse.Namespace) -> None:
    """Executes pipeline runner in replay or live mode."""
    mode = args.mode.lower()
    limit = args.limit
    interval = args.interval
    backend = getattr(args, "backend", None)

    # In replay mode, default to fast offline VADER and rule matcher if no backend specified
    if mode == "replay" and backend is None:
        engine = get_risk_engine(sentiment_backend="vader", use_zero_shot=False)
    else:
        engine = get_risk_engine(sentiment_backend=backend)

    store = get_signal_store()
    logger.info(
        f"Starting RiskPulse pipeline in '{mode.upper()}' mode (backend: {engine.model_version}, limit: {limit})..."
    )

    def run_cycle() -> int:
        raw_docs: List[NormalizedDoc] = []

        # 1. Fetch social replay
        try:
            tweets_src = TweetsReplaySource()
            social_docs = tweets_src.fetch(limit=limit)
            raw_docs.extend(social_docs)
        except Exception as exc:
            logger.warning(f"Failed to fetch tweets replay: {exc}")

        # 2. Fetch news sources
        try:
            rss_src = RssSource()
            news_docs = rss_src.fetch(limit=max(50, limit // 2))
            raw_docs.extend(news_docs)
        except Exception as exc:
            logger.warning(f"Failed to fetch RSS feeds: {exc}")

        if mode == "live":
            try:
                gdelt_src = GdeltSource()
                gdelt_docs = gdelt_src.fetch(limit=50)
                raw_docs.extend(gdelt_docs)
            except Exception as exc:
                logger.warning(f"Failed to fetch GDELT in live mode: {exc}")

        # Add curated seed headlines
        raw_docs.extend(_load_starter_news_docs())

        # Deduplicate
        unique_docs = dedupe(raw_docs)
        logger.info(
            f"Collected {len(raw_docs)} total docs ({len(unique_docs)} unique). Processing through RiskEngine..."
        )

        # Process through AI/NLP pipeline
        signals = engine.process(unique_docs)
        new_count = store.save(signals)
        logger.info(f"Processed {len(signals)} signals. Persisted {new_count} new unique signals to SQLite & JSONL.")
        return new_count

    if mode == "replay":
        new_saved = run_cycle()
        stats = store.summary()
        print("\n" + "=" * 60)
        print("REPLAY RUN COMPLETE")
        print(f"New Signals Inserted:   {new_saved}")
        print(f"Total Database Signals: {stats['total_signals']}")
        print(f"Event Types Breakdown:  {stats['by_event_type']}")
        print(f"Source Breakdown:       {stats['by_source_type']}")
        print("=" * 60 + "\n")
    else:
        logger.info(f"Live loop initiated. Polling sources every {interval}s (Press Ctrl+C to terminate)...")
        try:
            while True:
                run_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Live polling interrupted cleanly by user.")


def handle_export(args: argparse.Namespace) -> None:
    """Exports latest n stored signals to a target JSONL file."""
    store = get_signal_store()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = args.n

    signals = store.latest(n=n)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in signals:
            f.write(s.model_dump_json() + "\n")

    print(f"Exported {len(signals)} signals successfully to {out_path}")


def main() -> None:
    """Main CLI entrypoint with subcommands."""
    parser = argparse.ArgumentParser(description="RiskPulse AI/NLP Risk Engine CLI")
    parser.add_argument("--check-config", action="store_true", help="Validate YAML configurations and exit")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw text from news or social feeds")
    ingest_parser.add_argument(
        "--source",
        choices=["gdelt", "newsapi", "rss", "tweets"],
        default="gdelt",
        help="Data source identifier",
    )
    ingest_parser.add_argument("--days", type=int, default=2, help="Days of history to look back")
    ingest_parser.add_argument("--limit", type=int, default=100, help="Maximum number of items to fetch")

    # Subcommand: sentiment
    sentiment_parser = subparsers.add_parser("sentiment", help="Evaluate sentiment of input text")
    sentiment_parser.add_argument("text", type=str, help="Text string to evaluate")
    sentiment_parser.add_argument(
        "--backend",
        choices=["finbert", "vader"],
        default=None,
        help="Sentiment backend (finbert or vader)",
    )

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run the full ingestion and risk processing pipeline")
    run_parser.add_argument("--mode", choices=["live", "replay"], default="replay", help="Execution mode")
    run_parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds for live mode")
    run_parser.add_argument("--limit", type=int, default=200, help="Maximum documents to pull per cycle")
    run_parser.add_argument("--backend", choices=["finbert", "vader"], default=None, help="Sentiment backend")

    # Subcommand: export
    export_parser = subparsers.add_parser("export", help="Export stored signals to JSONL")
    export_parser.add_argument("--out", type=str, default="data/signals_sample.jsonl", help="Output file path")
    export_parser.add_argument("--n", type=int, default=400, help="Number of latest signals to export")

    args = parser.parse_args()

    if args.check_config:
        cfg = get_config()
        logger.info(f"Loaded {len(cfg.tickers)} tickers successfully: {list(cfg.tickers.keys())}")
        sys.exit(0)

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "sentiment":
        handle_sentiment(args)
    elif args.command == "run":
        handle_run(args)
    elif args.command == "export":
        handle_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
