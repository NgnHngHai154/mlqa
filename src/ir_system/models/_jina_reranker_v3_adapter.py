"""Jina v3 native Hugging Face inference; no pairwise conversion or reblocking."""
from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from typing import Sequence

from ir_system.models.listwise_reranker import ListwiseRerankerModel


JINA_V3_ID = "jinaai/jina-reranker-v3"
JINA_V3_REVISION = "10fb694fc21f7a710a563ff1eb977a460f3868e4"


def is_jina_v3(model_id: str) -> bool:
    return model_id.strip().lower() == JINA_V3_ID


class JinaRerankerV3Adapter(ListwiseRerankerModel):
    def __init__(self, model_id: str = JINA_V3_ID, device: str = "cpu",
                 revision: str | None = None) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
            from transformers.models.qwen3 import modeling_qwen3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'Jina v3 requires the [jina-v3] extra. Install pip install -e ".[jina-v3]" '
                'in a separate environment from [all]/[late-interaction].'
            ) from exc

        self._model_id = model_id
        # Local snapshots already identify their files; remote v3 is pinned by default.
        self.revision = revision or (JINA_V3_REVISION if is_jina_v3(model_id) else None)
        if not is_jina_v3(model_id) and not Path(model_id).is_dir():
            raise ValueError("Jina v3 adapter expects jinaai/jina-reranker-v3 or a local snapshot.")
        self._model = AutoModel.from_pretrained(
            model_id, revision=self.revision, trust_remote_code=True, dtype="auto",
        )
        if not callable(getattr(self._model, "rerank", None)):
            raise ValueError("The selected snapshot does not expose Jina's native rerank().")
        # Native _ensure_tokenizer otherwise loads from main, ignoring model revision.
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.revision, trust_remote_code=True,
        )
        self._model._tokenizer = tokenizer
        self._model.to(device)
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def inference_config(self) -> dict:
        return {
            "backend": "jina-v3-native",
            "revision": self.revision,
            "resolved_revision": getattr(self._model.config, "_commit_hash", None),
            "dtype": str(self._model.dtype),
            "rerank_options": "Hugging Face defaults; all candidates in one rerank call",
        }

    def score_list(self, query_text: str, document_texts: Sequence[str]) -> list[float]:
        if not document_texts:
            return []
        # Keep native prompt, truncation, block handling, query aggregation and cosine.
        # top_n=None returns all candidates, allowing deterministic tie-breaking locally.
        results = self._model.rerank(query_text, list(document_texts), top_n=None)
        if len(results) != len(document_texts):
            raise RuntimeError("Jina v3 returned an incorrect number of scores.")
        scores = [0.0] * len(document_texts)
        seen = set()
        for result in results:
            index = result.get("index")
            if (isinstance(index, bool) or not isinstance(index, Integral)
                    or not 0 <= index < len(scores) or index in seen):
                raise RuntimeError(f"Jina v3 returned an invalid/duplicate index: {index!r}")
            score = float(result["relevance_score"])
            if not math.isfinite(score):
                raise RuntimeError(f"Jina v3 returned a non-finite score at index {index}.")
            scores[index] = score
            seen.add(index)
        return scores
