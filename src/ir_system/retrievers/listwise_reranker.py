"""Two-stage listwise reranking, returning standard Hit objects."""
from __future__ import annotations

import math
from typing import Sequence

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models.listwise_reranker import ListwiseRerankerModel
from ir_system.retrievers.base import Retriever


class ListwiseRerankerRetriever(Retriever):
    def __init__(self, model: ListwiseRerankerModel, candidate_retriever: Retriever,
                 candidate_top_k: int = 100) -> None:
        if candidate_retriever is None or candidate_top_k <= 0:
            raise ValueError("Listwise reranking needs a candidate retriever and positive candidate_top_k.")
        self._model = model
        # Same composition hook as existing reranking: dense index loading works unchanged.
        self._candidate_retriever = candidate_retriever
        self._candidate_top_k = candidate_top_k
        self._documents: dict[str, str] = {}
        self._built = False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("ListwiseRerankerRetriever.build: corpus must not be empty.")
        self._documents = {doc.doc_id: doc.text for doc in documents}
        self._candidate_retriever.build(documents)
        self._built = True

    def retrieve(self, query: Query, top_k: int) -> Sequence[Hit]:
        if not self._built:
            raise RuntimeError("build() must be called before retrieve().")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        candidates = self._candidate_retriever.retrieve(query, self._candidate_top_k)
        doc_ids = [hit.doc_id for hit in candidates]
        if not doc_ids:
            return []
        if len(set(doc_ids)) != len(doc_ids) or any(d not in self._documents for d in doc_ids):
            raise RuntimeError("Candidate retriever returned duplicate or unknown document IDs.")
        scores = list(self._model.score_list(query.text, [self._documents[d] for d in doc_ids]))
        if len(scores) != len(doc_ids):
            raise RuntimeError("Listwise reranker returned an incorrect number of scores.")
        hits = [Hit(doc_id=d, score=float(s)) for d, s in zip(doc_ids, scores)]
        if any(not math.isfinite(hit.score) for hit in hits):
            raise RuntimeError("Listwise reranker returned non-finite scores.")
        return sorted(hits, key=lambda hit: (-hit.score, hit.doc_id))[:top_k]
