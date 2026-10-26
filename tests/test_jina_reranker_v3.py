"""Offline integration tests; opt-in native checkpoint parity test at the end."""
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ir_system.cli.main import _build_retriever, _find_dense_retrievers, main
from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.query import Query
from ir_system.models import factory
from ir_system.models._jina_reranker_v3_adapter import (
    JINA_V3_ID, JINA_V3_REVISION, JinaRerankerV3Adapter,
)
from ir_system.retrievers.listwise_reranker import ListwiseRerankerRetriever


def adapter_with_results(results):
    adapter = object.__new__(JinaRerankerV3Adapter)
    adapter._model_id = JINA_V3_ID
    adapter.revision = JINA_V3_REVISION
    adapter._model = SimpleNamespace(
        rerank=Mock(return_value=results), dtype="bfloat16",
        config=SimpleNamespace(_commit_hash=JINA_V3_REVISION),
    )
    return adapter


def test_native_load_uses_matching_revision_dtype_and_eval(monkeypatch):
    native = Mock()
    tokenizer = Mock()
    transformers = ModuleType("transformers")
    transformers.AutoModel = SimpleNamespace(from_pretrained=Mock(return_value=native))
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    qwen = ModuleType("transformers.models.qwen3")
    qwen.modeling_qwen3 = object()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3", qwen)
    adapter = JinaRerankerV3Adapter(device="cpu")
    transformers.AutoModel.from_pretrained.assert_called_once_with(
        JINA_V3_ID, revision=JINA_V3_REVISION, trust_remote_code=True, dtype="auto",
    )
    transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
        JINA_V3_ID, revision=JINA_V3_REVISION, trust_remote_code=True,
    )
    assert adapter._model._tokenizer is tokenizer
    native.to.assert_called_once_with("cpu")
    native.eval.assert_called_once_with()


def test_native_list_is_not_split_and_indices_not_text_map_scores():
    docs = ["same text"] * 100
    results = [{"index": np.int64(i), "relevance_score": i / 100} for i in reversed(range(100))]
    adapter = adapter_with_results(results)
    assert adapter.score_list("query", docs) == pytest.approx([i / 100 for i in range(100)])
    adapter._model.rerank.assert_called_once_with("query", docs, top_n=None)


@pytest.mark.parametrize("results", [
    [],
    [{"index": 0, "relevance_score": 1}] * 2,
    [{"index": 0, "relevance_score": 1}, {"index": 2, "relevance_score": 0}],
    [{"index": 0.0, "relevance_score": 1}, {"index": 1, "relevance_score": 0}],
    [{"index": True, "relevance_score": 1}, {"index": 0, "relevance_score": 0}],
    [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0}],
    [{"index": 0, "relevance_score": float("inf")}, {"index": 1, "relevance_score": 0}],
])
def test_bad_native_output_fails_without_partial_rankings(results):
    with pytest.raises(RuntimeError):
        adapter_with_results(results).score_list("q", ["a", "b"])


def test_empty_list_does_not_call_native():
    adapter = adapter_with_results([])
    assert adapter.score_list("q", []) == []
    adapter._model.rerank.assert_not_called()


def test_stage_one_limit_order_ties_and_empty_candidates():
    documents = [Document(str(i), "same") for i in range(101)]
    stage1 = Mock()
    stage1.retrieve.return_value = [Hit(str(i), 0.0) for i in reversed(range(100))]
    adapter = adapter_with_results([{"index": i, "relevance_score": .5} for i in range(100)])
    retriever = ListwiseRerankerRetriever(adapter, stage1)
    with pytest.raises(RuntimeError, match="build"):
        retriever.retrieve(Query("q", "query"), 10)
    retriever.build(documents)
    hits = retriever.retrieve(Query("q", "query"), 10)
    stage1.retrieve.assert_called_once_with(Query("q", "query"), 100)
    assert [h.doc_id for h in hits] == sorted(str(i) for i in range(100))[:10]
    assert all(h.score == .5 for h in hits)
    adapter._model.rerank.assert_called_once()
    stage1.retrieve.return_value = []
    assert retriever.retrieve(Query("q", "query"), 10) == []
    adapter._model.rerank.assert_called_once()


@pytest.mark.parametrize("model_id,expected", [
    (JINA_V3_ID, "listwise-reranker"),
    ("jinaai/jina-reranker-v2-base-multilingual", "cross-encoder"),
    ("Qwen/Qwen3-Reranker-4B", "cross-encoder"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder"),
    ("intfloat/multilingual-e5-small", "single-vector"),
    ("colbert-ir/colbertv2.0", "multi-vector"),
])
def test_factory_routing_is_narrow(model_id, expected):
    assert factory._infer_model_type(model_id) == expected


def test_other_cross_encoders_keep_original_builder(monkeypatch):
    builder = Mock(return_value=object())
    monkeypatch.setattr(factory, "_resolve_device", lambda device: "cpu")
    monkeypatch.setattr(factory, "_build_cross_encoder", builder)
    result = factory.create_model("Qwen/Qwen3-Reranker-4B", batch_size=7)
    assert result is builder.return_value
    builder.assert_called_once_with("Qwen/Qwen3-Reranker-4B", "cpu", 7)


def test_jina_cannot_enter_pairwise_analysis_contract():
    with pytest.raises(ValueError, match="listwise-reranker"):
        factory.create_model(JINA_V3_ID, model_type="cross-encoder")


def test_dense_candidate_index_is_discoverable():
    from ir_system.models.single_vector import SingleVectorEmbeddingModel
    from ir_system.retrievers.dense import DenseRetriever

    class Encoder(SingleVectorEmbeddingModel):
        name = "test"

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2))

    retriever = _build_retriever(
        "listwise-reranker", adapter_with_results([]), 100, 32,
        candidate_retriever_type="dense", candidate_model=Encoder(),
    )
    dense = _find_dense_retrievers(retriever)
    assert len(dense) == 1
    assert isinstance(dense[0], DenseRetriever)


def test_cli_bm25_candidates_metrics_and_saved_config(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text(
        '\n'.join(json.dumps({"doc_id": d, "text": "cat"}) for d in ["a", "b", "c"]),
        encoding="utf-8",
    )
    (dataset / "queries.jsonl").write_text('{"qid":"q","text":"cat"}', encoding="utf-8")
    (dataset / "qrels.tsv").write_text("q\tb\t1\n", encoding="utf-8")
    adapter = adapter_with_results([
        {"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1},
    ])
    import ir_system.models._jina_reranker_v3_adapter as adapter_module
    monkeypatch.setattr(adapter_module, "JinaRerankerV3Adapter", Mock(return_value=adapter))
    monkeypatch.setattr(factory, "_resolve_device", lambda device: "cpu")
    output = tmp_path / "output"
    assert main([
        "--dataset", str(dataset), "--model", JINA_V3_ID,
        "--retriever", "listwise-reranker", "--candidate-top-k", "2",
        "--top-k", "1", "--metrics", "ndcg@1", "mrr@1", "recall@1",
        "--output", str(output),
    ]) == 0
    metrics = json.loads((output / "metrics.json").read_text())
    assert all(value == pytest.approx(1.0) for scores in metrics.values() for value in scores.values())
    config = json.loads((output / "config.json").read_text())
    assert config["candidate_top_k"] == 2
    assert config["listwise_inference"]["revision"] == JINA_V3_REVISION
    adapter._model.rerank.assert_called_once_with("cat", ["cat", "cat"], top_n=None)
    assert '"b"' in (output / "run.jsonl").read_text()


@pytest.mark.skipif(os.environ.get("IR_TEST_JINA_V3") != "1",
                    reason="Set IR_TEST_JINA_V3=1 in the Jina environment to download/run the checkpoint.")
def test_real_checkpoint_native_parity():
    adapter = JinaRerankerV3Adapter(device=os.environ.get("IR_JINA_DEVICE", "cpu"))
    docs = ["Hanoi is the capital of Vietnam.", "Coffee is a drink.", "Hà Nội là thủ đô Việt Nam."]
    query = "What is the capital of Vietnam?"
    native = adapter._model.rerank(query, docs)
    expected = [0.0] * len(docs)
    for item in native:
        expected[item["index"]] = float(item["relevance_score"])
    assert adapter.score_list(query, docs) == pytest.approx(expected, abs=1e-6)
