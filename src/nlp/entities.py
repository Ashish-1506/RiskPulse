"""Named entity, ticker, country, and sector extraction pipeline."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.config import get_config
from src.ingestion.normalise import NormalizedDoc

# Try to load spaCy en_core_web_sm if available, otherwise fail gracefully
_SPACY_NLP = None
try:
    import spacy

    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        _SPACY_NLP = None
except ImportError:
    _SPACY_NLP = None


# Corporate / Financial context keywords to disambiguate short common words (e.g. "apple")
FINANCIAL_CONTEXT_KEYWORDS: Set[str] = {
    "share",
    "shares",
    "stock",
    "stocks",
    "earnings",
    "revenue",
    "guidance",
    "profit",
    "loss",
    "margin",
    "margins",
    "nasdaq",
    "nyse",
    "investor",
    "investors",
    "analyst",
    "analysts",
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
    "ceo",
    "cfo",
    "dividend",
    "dividends",
    "supplier",
    "suppliers",
    "rally",
    "slump",
    "fall",
    "falls",
    "fell",
    "jump",
    "jumps",
    "jumped",
    "plunge",
    "plunges",
    "plunged",
    "market",
    "markets",
    "antitrust",
    "sales",
    "corp",
    "corporation",
    "inc",
    "ltd",
    "valuation",
    "buyback",
    "buyout",
    "tech",
}

# Extensive Country & Geopolitical Region Taxonomy (~60 countries/regions with demonyms/aliases)
COUNTRY_MAP: Dict[str, List[str]] = {
    "United States": [
        "united states",
        "u.s.",
        "u.s.a.",
        "usa",
        "us",
        "america",
        "american",
        "washington",
        "wall street",
    ],
    "China": ["china", "chinese", "beijing", "shanghai"],
    "Russia": ["russia", "russian", "moscow", "kremlin"],
    "Ukraine": ["ukraine", "ukrainian", "kyiv", "kiev"],
    "United Kingdom": ["united kingdom", "u.k.", "uk", "britain", "british", "england", "london"],
    "Germany": ["germany", "german", "berlin", "frankfurt"],
    "France": ["france", "french", "paris"],
    "Japan": ["japan", "japanese", "tokyo"],
    "Taiwan": ["taiwan", "taiwanese", "taipei"],
    "India": ["india", "indian", "new delhi", "mumbai"],
    "Israel": ["israel", "israeli", "tel aviv", "jerusalem"],
    "Iran": ["iran", "iranian", "tehran"],
    "Saudi Arabia": ["saudi arabia", "saudi", "riyadh", "aramco"],
    "United Arab Emirates": ["uae", "united arab emirates", "dubai", "abu dhabi"],
    "Canada": ["canada", "canadian", "ottawa", "toronto"],
    "Australia": ["australia", "australian", "sydney", "canberra"],
    "Brazil": ["brazil", "brazilian", "brasilia"],
    "Mexico": ["mexico", "mexican", "mexico city"],
    "South Korea": ["south korea", "korea", "korean", "seoul"],
    "Italy": ["italy", "italian", "rome", "milan"],
    "Spain": ["spain", "spanish", "madrid"],
    "Netherlands": ["netherlands", "dutch", "amsterdam"],
    "Switzerland": ["switzerland", "swiss", "zurich", "geneva"],
    "Poland": ["poland", "polish", "warsaw"],
    "Turkey": ["turkey", "turkish", "ankara", "istanbul"],
    "South Africa": ["south africa", "south african", "johannesburg"],
    "Singapore": ["singapore", "singaporean"],
    "Hong Kong": ["hong kong"],
    "Indonesia": ["indonesia", "indonesian", "jakarta"],
    "Malaysia": ["malaysia", "malaysian", "kuala lumpur"],
    "Vietnam": ["vietnam", "vietnamese", "hanoi"],
    "Thailand": ["thailand", "thai", "bangkok"],
    "Philippines": ["philippines", "philippine", "manila"],
    "Qatar": ["qatar", "qatari", "doha"],
    "Kuwait": ["kuwait", "kuwaiti"],
    "Iraq": ["iraq", "iraqi", "baghdad"],
    "Egypt": ["egypt", "egyptian", "cairo"],
    "Nigeria": ["nigeria", "nigerian", "lagos"],
    "Argentina": ["argentina", "argentine", "buenos aires"],
    "Venezuela": ["venezuela", "venezuelan", "caracas"],
    "Colombia": ["colombia", "colombian", "bogota"],
    "Chile": ["chile", "chilean", "santiago"],
    "Norway": ["norway", "norwegian", "oslo"],
    "Sweden": ["sweden", "swedish", "stockholm"],
    "Finland": ["finland", "finnish", "helsinki"],
    "Denmark": ["denmark", "danish", "copenhagen"],
    "Ireland": ["ireland", "irish", "dublin"],
    "Belgium": ["belgium", "belgian", "brussels"],
    "Greece": ["greece", "greek", "athens"],
    "Portugal": ["portugal", "portuguese", "lisbon"],
    "Austria": ["austria", "austrian", "vienna"],
    "Pakistan": ["pakistan", "pakistani", "islamabad"],
    "Bangladesh": ["bangladesh", "bangladeshi", "dhaka"],
    "New Zealand": ["new zealand"],
    "Middle East": ["middle east", "red sea", "gulf", "strait of hormuz", "persian gulf"],
    "European Union": ["european union", "eu", "eurozone", "europe", "european"],
}

# Sector Keyword Mapping across GICS Sectors
SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "Energy": [
        "oil",
        "crude",
        "refinery",
        "refineries",
        "opec",
        "brent",
        "wti",
        "petroleum",
        "natural gas",
        "gasoline",
        "drilling",
        "pipeline",
        "hydrocarbon",
    ],
    "Financials": [
        "bank",
        "banks",
        "banking",
        "lender",
        "lenders",
        "loan",
        "loans",
        "credit",
        "mortgage",
        "wall street",
        "fed",
        "federal reserve",
        "yield",
        "yields",
        "interest rate",
        "liquidity",
        "bond",
        "bonds",
        "treasury",
        "treasuries",
    ],
    "Information Technology": [
        "chip",
        "chips",
        "semiconductor",
        "semiconductors",
        "software",
        "hardware",
        "cloud",
        "ai",
        "artificial intelligence",
        "gpu",
        "gpus",
        "saas",
        "cybersecurity",
        "server",
        "computing",
        "technology",
    ],
    "Communication Services": [
        "telecom",
        "telecommunications",
        "social media",
        "advertising",
        "streaming",
        "broadband",
        "media",
        "digital advertising",
    ],
    "Consumer Discretionary": [
        "retail",
        "retailer",
        "ecommerce",
        "automotive",
        "automobile",
        "electric vehicle",
        "ev",
        "luxury",
        "consumer spending",
        "discretionary",
    ],
    "Consumer Staples": [
        "grocery",
        "groceries",
        "food",
        "beverage",
        "beverages",
        "drinks",
        "supermarket",
        "staples",
        "household products",
    ],
    "Health Care": [
        "pharma",
        "pharmaceutical",
        "pharmaceuticals",
        "drug",
        "drugs",
        "fda",
        "healthcare",
        "biotech",
        "biotechnology",
        "vaccine",
        "vaccines",
        "clinical trial",
        "hospital",
    ],
    "Industrials": [
        "aerospace",
        "defense",
        "airline",
        "airlines",
        "freight",
        "logistics",
        "shipping",
        "container",
        "cargo",
        "manufacturing",
        "machinery",
    ],
    "Materials": ["chemical", "chemicals", "mining", "metals", "steel", "copper", "lithium", "gold", "rare earth"],
    "Utilities": ["electric utility", "power grid", "water utility", "renewables", "electricity"],
    "Real Estate": ["commercial real estate", "reit", "housing market", "property", "residential"],
}


class EntityExtractor:
    """Extracts indexed tickers, other cashtags, countries, and sectors from text."""

    def __init__(self) -> None:
        self.cfg = get_config()
        self.index_tickers = self.cfg.tickers
        self._build_ticker_matchers()

    def _build_ticker_matchers(self) -> None:
        """Compiles regex matchers for configured tickers and their aliases."""
        self.ticker_regexes: Dict[str, List[re.Pattern]] = {}
        for symbol, meta in self.index_tickers.items():
            patterns = []
            # Exact cashtag match: $AAPL
            patterns.append(re.compile(rf"\${symbol}\b", re.IGNORECASE))
            # Standalone symbol match: AAPL
            patterns.append(re.compile(rf"\b{symbol}\b"))

            for alias in meta.aliases:
                if alias == symbol or alias == f"${symbol}":
                    continue
                # For company names like "Apple Inc", "Microsoft", "Johnson & Johnson"
                escaped = re.escape(alias)
                patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))

            self.ticker_regexes[symbol] = patterns

    def _has_financial_context(self, text: str) -> bool:
        """Checks if text contains financial/corporate keywords for ambiguous nouns."""
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))
        return bool(words & FINANCIAL_CONTEXT_KEYWORDS)

    def extract_tickers(self, text: str) -> Tuple[List[str], List[str]]:
        """Identifies indexed S&P 100 tickers and other discovered cashtags.

        Returns:
            Tuple of (matched_index_tickers, other_discovered_tickers)
        """
        matched_tickers: Set[str] = set()
        text_lower = text.lower()
        has_context = self._has_financial_context(text)

        # 1. Match configured index tickers
        for symbol, patterns in self.ticker_regexes.items():
            self.index_tickers[symbol]
            for pat in patterns:
                match = pat.search(text)
                if not match:
                    continue

                matched_word = match.group(0).lower()

                # Handle ambiguous alias: "apple" (fruit vs company)
                if matched_word in ("apple", "meta", "target"):
                    # Check if exact company indicator (e.g. "Apple Inc", "Apple's", or $AAPL)
                    has_company_indicator = bool(
                        re.search(rf"\b{matched_word}('s| inc| corp| co)\b", text_lower)
                        or f"${symbol.lower()}" in text_lower
                        or symbol in text.split()
                    )
                    if has_company_indicator or has_context:
                        matched_tickers.add(symbol)
                        break
                else:
                    matched_tickers.add(symbol)
                    break

        # 2. Extract all cashtags ($XYZ) to identify other_tickers
        cashtags = re.findall(r"\$([A-Z]{1,5})\b", text)
        other_tickers = [c for c in set(cashtags) if c not in self.index_tickers and c not in matched_tickers]

        return sorted(matched_tickers), sorted(other_tickers)

    def extract_countries(self, text: str) -> List[str]:
        """Detects countries and geopolitical regions from text and optional spaCy GPEs."""
        found_countries: Set[str] = set()
        text_lower = f" {text.lower()} "

        # Dictionary keyword matching
        for country, aliases in COUNTRY_MAP.items():
            for alias in aliases:
                # Require word boundaries for abbreviations like "us", "uk", "eu"
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, text_lower):
                    found_countries.add(country)
                    break

        # spaCy GPE enrichment if model is installed
        if _SPACY_NLP is not None:
            try:
                doc = _SPACY_NLP(text[:1000])
                for ent in doc.ents:
                    if ent.label_ == "GPE":
                        ent_text = ent.text.lower()
                        for country, aliases in COUNTRY_MAP.items():
                            if ent_text in aliases:
                                found_countries.add(country)
            except Exception:
                pass

        return sorted(found_countries)

    def extract_sectors(self, text: str, tickers: Optional[List[str]] = None) -> List[List[str] | str]:
        """Extracts GICS sectors from text keywords and inherits from detected tickers."""
        found_sectors: Set[str] = set()
        text_lower = f" {text.lower()} "

        # 1. Inherit sector of detected tickers from config
        if tickers:
            for symbol in tickers:
                if symbol in self.index_tickers:
                    found_sectors.add(self.index_tickers[symbol].sector)

        # 2. Match sector keywords
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    found_sectors.add(sector)
                    break

        return sorted(found_sectors)

    def enrich(self, doc: NormalizedDoc) -> NormalizedDoc:
        """Enriches NormalizedDoc with extracted tickers, other_tickers, countries, and sectors."""
        combined_text = f"{doc.title}. {doc.text}"
        if doc.ticker_hint and doc.ticker_hint in self.index_tickers:
            combined_text = f"{doc.ticker_hint} {combined_text}"

        tickers, other_tickers = self.extract_tickers(combined_text)
        countries = self.extract_countries(combined_text)
        sectors = self.extract_sectors(combined_text, tickers=tickers)

        doc.tickers = tickers
        doc.other_tickers = other_tickers
        doc.countries = countries
        doc.sectors = sectors
        return doc


# Module-level convenience functions
_EXTRACTOR: Optional[EntityExtractor] = None


def get_extractor() -> EntityExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = EntityExtractor()
    return _EXTRACTOR


def extract_tickers(text: str) -> List[str]:
    """Convenience function returning index tickers."""
    tickers, _ = get_extractor().extract_tickers(text)
    return tickers


def extract_countries(text: str) -> List[str]:
    """Convenience function returning countries and regions."""
    return get_extractor().extract_countries(text)


def extract_sectors(text: str, tickers: Optional[List[str]] = None) -> List[str]:
    """Convenience function returning sectors."""
    return get_extractor().extract_sectors(text, tickers=tickers)


def enrich(doc: NormalizedDoc) -> NormalizedDoc:
    """Enriches a NormalizedDoc in place and returns it."""
    return get_extractor().enrich(doc)
