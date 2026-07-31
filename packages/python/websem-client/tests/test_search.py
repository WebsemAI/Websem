import math

from websem_client import (
    HybridSearchOptions,
    LoadedArtifacts,
    SearchOptions,
    WebsemClient,
    embed_query,
    weighted_reciprocal_rank_fusion,
)
from websem_types import Manifest


def _artifacts() -> LoadedArtifacts:
    manifest: Manifest = {
        "websem_version": "1.0",
        "model_id": "test/model",
        "dims": 2,
        "vocab_size": 3,
        "n_chunks": 3,
        "chunk_size": 600,
        "chunk_overlap": 120,
        "title_prefix": True,
        "chunker_version": "1",
        "doc_scale": 127,
        "files": {
            "chunks": "chunks.json",
            "docs": "docs.bin",
            "tokens": "tokens.bin",
            "scales": "scales.bin",
            "vocab": "vocab.json",
        },
    }
    return LoadedArtifacts(
        manifest=manifest,
        chunks=(
            {
                "document": "a",
                "title": "Alpha guide",
                "href": "/a",
                "snippet": "alpha basics",
            },
            {
                "document": "a",
                "title": "Alpha guide",
                "href": "/a#advanced",
                "snippet": "advanced alpha",
            },
            {
                "document": "b",
                "title": "Beta reference",
                "href": "/b",
                "snippet": "beta alpha a the",
            },
        ),
        docs=bytes((127, 0, 64, 64, 0, 127)),
        tokens=bytes((0, 0, 127, 0, 0, 127)),
        scales=(1.0, 1.0, 1.0),
        vocabulary=("[UNK]", "alpha", "beta"),
    )


def test_embedding_means_scaled_rows_then_normalizes() -> None:
    vector = embed_query([0, 1], bytes((2, 0, 0, 4)), (0.5, 0.25), 2)
    expected = 1 / math.sqrt(2)
    assert vector == (expected, expected)
    assert embed_query([], bytes((1, 2)), (1.0,), 2) == (0.0, 0.0)


def test_semantic_search_rolls_up_best_chunk_and_is_deterministic() -> None:
    client = WebsemClient(_artifacts())

    results = client.semantic_search("alpha", SearchOptions(limit=5, min_score=0.1))

    assert [result["document"] for result in results] == ["a"]
    assert results[0]["href"] == "/a"
    assert results[0]["score"] == 1.0
    assert client.semantic_search("unknown") == []


def test_specific_term_heuristic_is_case_insensitive_and_optional() -> None:
    client = WebsemClient(_artifacts())

    assert client.semantic_search("GUIDE")[0]["document"] == "a"
    assert client.semantic_search("GUIDE", SearchOptions(specific_term_heuristic=False)) == []
    assert (
        client.hybrid_search(
            "GUIDE",
            HybridSearchOptions(semantic_weight=0, keyword_weight=0),
        )[0]["document"]
        == "a"
    )
    assert client.semantic_search("A") == []
    assert client.semantic_search("THE") == []


def test_keyword_search_ranks_metadata_and_rolls_up_documents() -> None:
    results = WebsemClient(_artifacts()).keyword_search("beta alpha")

    assert [result["document"] for result in results] == ["b", "a"]


def test_weighted_rrf_and_hybrid_weights() -> None:
    fused = weighted_reciprocal_rank_fusion((("a", "b"), ("b", "a")), weights=(2.0, 1.0))
    assert [document for document, _ in fused] == ["a", "b"]
    assert weighted_reciprocal_rank_fusion((("a",), ("b",)), weights=(0.0, 1.0)) == [("b", 1 / 61)]

    results = WebsemClient(_artifacts()).hybrid_search(
        "beta", HybridSearchOptions(limit=2, semantic_weight=2.0, keyword_weight=1.0)
    )
    assert [result["document"] for result in results] == ["b", "a"]

    semantic_only = WebsemClient(_artifacts()).hybrid_search(
        "alpha",
        HybridSearchOptions(limit=2, semantic_weight=1.0, keyword_weight=0.0),
    )
    assert [result["document"] for result in semantic_only] == ["a", "b"]
