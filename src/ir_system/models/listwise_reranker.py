"""Scoring contract for a query and an ordered candidate list."""
from abc import abstractmethod
from typing import Sequence

from ir_system.models.base import Model


class ListwiseRerankerModel(Model):
    """Scores may depend on every document and its position in the list."""

    @abstractmethod
    def score_list(self, query_text: str, document_texts: Sequence[str]) -> Sequence[float]:
        """Return one finite score per document, in INPUT order."""
        ...
