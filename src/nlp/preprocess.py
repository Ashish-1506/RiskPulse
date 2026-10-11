"""Text cleaning, language validation, and model truncation utilities."""

from __future__ import annotations

import html
import re
from typing import Set

# Common English stopwords for lightweight language detection
ENGLISH_STOPWORDS: Set[str] = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "is",
    "are",
    "was",
    "were",
    "has",
    "had",
    "been",
    "market",
    "shares",
    "stock",
    "growth",
}

# Regex patterns
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_HASHTAG = re.compile(r"#([a-zA-Z0-9_]+)")
# Emojis & miscellaneous pictographic symbols
RE_EMOJIS = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50]")

UNICODE_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "\u00a0": " ",
}


def clean_text(text: str) -> str:
    """Cleans and standardizes raw text while preserving cashtags ($AAPL).

    Operations:
      - Unescapes HTML entities and strips HTML tags.
      - Removes HTTP/HTTPS and www URLs.
      - Normalizes Unicode quotation marks and em/en dashes.
      - Converts hashtags (#inflation -> inflation).
      - Strips emojis and pictographs.
      - Collapses consecutive whitespace characters.
    """
    if not text:
        return ""

    # Unescape HTML entities
    cleaned = html.unescape(text)

    # Strip HTML tags
    cleaned = RE_HTML.sub(" ", cleaned)

    # Strip URLs
    cleaned = RE_URL.sub(" ", cleaned)

    # Normalize unicode quotes and dashes
    for char, rep in UNICODE_REPLACEMENTS.items():
        cleaned = cleaned.replace(char, rep)

    # Convert hashtags to plain words (#inflation -> inflation)
    cleaned = RE_HASHTAG.sub(r"\1", cleaned)

    # Strip emojis
    cleaned = RE_EMOJIS.sub("", cleaned)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


# Non-English high-frequency markers (German, Spanish, French, Italian)
NON_ENGLISH_MARKERS: Set[str] = {
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "einem",
    "einen",
    "eines",
    "und",
    "oder",
    "aber",
    "für",
    "fuer",
    "mit",
    "von",
    "zu",
    "im",
    "am",
    "auf",
    "ist",
    "sind",
    "war",
    "waren",
    "wird",
    "werden",
    "nicht",
    "steigt",
    "sinkt",
    "weiter",
    "deutschland",
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "del",
    "al",
    "por",
    "para",
    "con",
    "sin",
    "sobre",
    "que",
    "como",
    "pero",
    "este",
    "esta",
    "estos",
    "estas",
    "su",
    "sus",
    "es",
    "son",
    "fue",
    "fueron",
    "sube",
    "baja",
    "le",
    "les",
    "du",
    "des",
    "dans",
    "sur",
    "sous",
    "avec",
    "sans",
    "cette",
    "ces",
}

DISTINCTIVE_ENGLISH_WORDS: Set[str] = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "and",
    "to",
    "that",
    "this",
    "these",
    "those",
    "with",
    "for",
    "from",
    "at",
    "by",
    "have",
    "has",
    "had",
    "which",
    "will",
    "would",
    "their",
    "there",
    "what",
    "about",
    "market",
    "shares",
    "stock",
    "growth",
    "jump",
    "jumps",
    "plunge",
    "plunges",
    "fall",
    "falls",
    "rise",
    "raises",
    "rate",
    "rates",
    "cut",
    "cuts",
    "earnings",
    "profit",
}


def is_english(text: str) -> bool:
    """Lightweight heuristic to check if text is English without external dependencies.

    Checks Latin character ratio, absence of distinctive foreign markers,
    and presence of English functional words.
    """
    if not text or not text.strip():
        return False

    clean = text.lower()
    # Check proportion of ASCII/Latin letters
    latin_chars = sum(1 for c in clean if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in clean if c.isalpha())

    if total_letters == 0:
        return False

    latin_ratio = latin_chars / total_letters
    if latin_ratio < 0.70:
        return False

    # Word-level tokenization
    words = re.findall(r"\b[a-z]{2,}\b", clean)
    if not words:
        return False

    # Check for non-English markers
    non_english_count = sum(1 for w in words if w in NON_ENGLISH_MARKERS)
    english_distinct_count = sum(1 for w in words if w in DISTINCTIVE_ENGLISH_WORDS)

    if non_english_count > 0 and non_english_count >= english_distinct_count:
        return False

    if non_english_count >= 2:
        return False

    english_stopword_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)

    # For short headlines (<= 5 words), require at least one English functional/domain word
    if len(words) <= 5:
        return english_stopword_count >= 1 or english_distinct_count >= 1

    # For longer texts, require at least one English stopword and adequate ratio
    return english_stopword_count >= 1


def truncate_for_model(text: str, max_chars: int = 1000) -> str:
    """Truncates text safely at a word boundary up to max_chars."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars - 60:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."
