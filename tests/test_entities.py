"""Unit tests for entity extraction, ticker disambiguation, and text preprocessing."""

from __future__ import annotations

from datetime import datetime, timezone

from src.ingestion.normalise import NormalizedDoc
from src.nlp.entities import (
    EntityExtractor,
    enrich,
    extract_countries,
    extract_sectors,
    extract_tickers,
)
from src.nlp.preprocess import clean_text, is_english, truncate_for_model


# ------------------------------------------------------------------------------
# 1. Ticker Disambiguation & Extraction Tests
# ------------------------------------------------------------------------------
def test_apple_company_context():
    """Identifies AAPL when 'Apple' appears with financial/market context."""
    text = "Apple shares fall after supplier warning"
    tickers = extract_tickers(text)
    assert "AAPL" in tickers


def test_apple_fruit_no_ticker():
    """Does NOT extract AAPL when 'apple' appears in a purely culinary context."""
    text = "I ate an apple"
    tickers = extract_tickers(text)
    assert "AAPL" not in tickers
    assert tickers == []


def test_unindexed_cashtag_in_other_tickers():
    """Validates that unindexed cashtags ($TSLA) are captured as other_tickers."""
    extractor = EntityExtractor()
    text = "$TSLA delivery numbers miss consensus estimates"
    idx_tickers, other_tickers = extractor.extract_tickers(text)
    assert "TSLA" not in idx_tickers
    assert "TSLA" in other_tickers


def test_cashtag_indexed_ticker():
    """Cashtag for configured ticker ($AAPL) resolves directly to AAPL."""
    text = "Momentum accelerating in $AAPL ahead of product event"
    tickers = extract_tickers(text)
    assert "AAPL" in tickers


def test_microsoft_azure_detection():
    """Matches MSFT from product/cloud alias and inherits Information Technology sector."""
    text = "Microsoft Azure cloud outage impacts enterprise users"
    tickers = extract_tickers(text)
    sectors = extract_sectors(text, tickers=tickers)
    assert "MSFT" in tickers
    assert "Information Technology" in sectors


def test_jpmorgan_chase_detection():
    """Matches JPM from full company name and Financials sector."""
    text = "JPMorgan Chase reports record net interest income"
    tickers = extract_tickers(text)
    sectors = extract_sectors(text, tickers=tickers)
    assert "JPM" in tickers
    assert "Financials" in sectors


def test_exxon_energy_detection():
    """Matches XOM and Energy sector from drilling and crude keywords."""
    text = "ExxonMobil expands offshore crude drilling"
    tickers = extract_tickers(text)
    sectors = extract_sectors(text, tickers=tickers)
    assert "XOM" in tickers
    assert "Energy" in sectors


def test_pfizer_healthcare_detection():
    """Matches PFE and Health Care sector from FDA and drug keywords."""
    text = "Pfizer secures FDA approval for novel oncology drug"
    tickers = extract_tickers(text)
    sectors = extract_sectors(text, tickers=tickers)
    assert "PFE" in tickers
    assert "Health Care" in sectors


def test_walmart_consumer_staples_detection():
    """Matches WMT and Consumer Staples sector from grocery keywords."""
    text = "Walmart raises dividend on strong grocery market share"
    tickers = extract_tickers(text)
    sectors = extract_sectors(text, tickers=tickers)
    assert "WMT" in tickers
    assert "Consumer Staples" in sectors


# ------------------------------------------------------------------------------
# 2. Country & Sector Geopolitical Tests
# ------------------------------------------------------------------------------
def test_russian_sanctions_european_banks():
    """Extracts Russia and European Union in countries, and Financials in sectors."""
    text = "Russian sanctions hit European banks"
    countries = extract_countries(text)
    sectors = extract_sectors(text)
    assert "Russia" in countries
    assert "European Union" in countries
    assert "Financials" in sectors


def test_red_sea_oil_attack():
    """Extracts Middle East in countries and Energy in sectors."""
    text = "Red Sea shipping attack lifts oil"
    countries = extract_countries(text)
    sectors = extract_sectors(text)
    assert "Middle East" in countries
    assert "Energy" in sectors


def test_chinese_semiconductors():
    """Extracts China in countries and Information Technology in sectors."""
    text = "Chinese export restrictions curb semiconductor shipments"
    countries = extract_countries(text)
    sectors = extract_sectors(text)
    assert "China" in countries
    assert "Information Technology" in sectors


# ------------------------------------------------------------------------------
# 3. Document Enrichment Integration Test
# ------------------------------------------------------------------------------
def test_enrich_normalized_doc():
    """Enriches NormalizedDoc and verifies fields are populated."""
    doc = NormalizedDoc.create(
        source="newsapi",
        source_type="news",
        published_at=datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc),
        title="NVIDIA and TSMC announce new chip packaging plant in Japan",
        text="NVDA expands partnership with $TSMC to build high-bandwidth memory servers.",
    )
    enriched = enrich(doc)
    assert "NVDA" in enriched.tickers
    assert "TSMC" in enriched.other_tickers
    assert "Japan" in enriched.countries
    assert "Information Technology" in enriched.sectors


# ------------------------------------------------------------------------------
# 4. Preprocessing Tests
# ------------------------------------------------------------------------------
def test_clean_text_comprehensive():
    """Cleans HTML, URLs, unicode quotes, hashtags, emojis while preserving cashtags."""
    raw = (
        "<p>Breaking: $AAPL announces #iPhone17! 🚀 Check out https://apple.com/live. "
        "“Record margins” expected – says Tim Cook.</p>"
    )
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "🚀" not in cleaned
    assert "$AAPL" in cleaned  # Cashtag preserved
    assert "iPhone17" in cleaned  # Hashtag hash stripped
    assert '"Record margins"' in cleaned  # Unicode quotes normalized
    assert "-" in cleaned  # Dash normalized


def test_is_english():
    """Validates language detection heuristic."""
    assert is_english("Federal Reserve raises interest rates by 25 basis points.")
    assert is_english("Apple stock jumps on blowout earnings report")
    assert not is_english("El Banco Central Europeo sube los tipos de interés.")
    assert not is_english("Die Inflation in Deutschland steigt weiter an.")
    assert not is_english("12345 67890 !!! ???")


def test_truncate_for_model():
    """Truncates long text cleanly at word boundaries."""
    long_text = "word " * 300
    truncated = truncate_for_model(long_text, max_chars=100)
    assert len(truncated) <= 103  # Allows for ellipsis
    assert truncated.endswith("...")
    assert not truncated.endswith(" ...")
