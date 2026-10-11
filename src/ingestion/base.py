"""Abstract base class for all ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from src.ingestion.normalise import NormalizedDoc


class Source(ABC):
    """Abstract interface for external and replay data sources."""

    @abstractmethod
    def fetch(self, since: Optional[datetime] = None, limit: int = 100) -> List[NormalizedDoc]:
        """Fetches raw documents from the source, normalizes them, and returns a list.

        Args:
            since: Optional cutoff datetime; documents older than this are omitted.
            limit: Maximum number of documents to retrieve.

        Returns:
            List of normalized documents.
        """
        pass
