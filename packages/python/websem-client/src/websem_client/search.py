import math
from collections.abc import Sequence
from dataclasses import dataclass

from websem_types import ChunkRecord, SearchResult

from .artifacts import LoadedArtifacts
from .tokenizer import WordPieceTokenizer, normalize, pre_tokenize

RRF_K = 60
SPECIFIC_TERM_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


@dataclass(frozen=True, slots=True)
class SearchOptions:
    limit: int = 10
    min_score: float = 0.0
    specific_term_heuristic: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if isinstance(self.min_score, bool) or not math.isfinite(self.min_score):
            raise ValueError("min_score must be finite")
        if not isinstance(self.specific_term_heuristic, bool):
            raise ValueError("specific_term_heuristic must be a boolean")


@dataclass(frozen=True, slots=True)
class HybridSearchOptions(SearchOptions):
    rrf_k: int = RRF_K
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0

    def __post_init__(self) -> None:
        SearchOptions.__post_init__(self)
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if (
            isinstance(self.semantic_weight, bool)
            or not math.isfinite(self.semantic_weight)
            or self.semantic_weight < 0
        ):
            raise ValueError("semantic_weight must be finite and non-negative")
        if (
            isinstance(self.keyword_weight, bool)
            or not math.isfinite(self.keyword_weight)
            or self.keyword_weight < 0
        ):
            raise ValueError("keyword_weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class _RankedChunk:
    document: str
    chunk_index: int
    score: float


def _int8(value: int) -> int:
    return value if value < 128 else value - 256


def embed_query(
    token_ids: Sequence[int],
    token_table: bytes,
    scales: Sequence[float],
    dims: int,
) -> tuple[float, ...]:
    if dims < 1:
        raise ValueError("dims must be positive")
    if len(token_table) != len(scales) * dims:
        raise ValueError("token table length does not match scales and dimensions")
    if any(token_id < 0 or token_id >= len(scales) for token_id in token_ids):
        raise ValueError("token id is outside the token table")
    if not token_ids:
        return (0.0,) * dims

    pooled = [0.0] * dims
    for token_id in token_ids:
        scale = scales[token_id]
        offset = token_id * dims
        for dimension in range(dims):
            pooled[dimension] += _int8(token_table[offset + dimension]) * scale
    count = len(token_ids)
    pooled = [value / count for value in pooled]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm == 0:
        return tuple(pooled)
    return tuple(value / norm for value in pooled)


def cosine_scores(query: Sequence[float], docs: bytes, n_chunks: int, dims: int) -> list[float]:
    if len(query) != dims:
        raise ValueError("query length does not match dimensions")
    if len(docs) != n_chunks * dims:
        raise ValueError("docs length does not match chunks and dimensions")
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        return [0.0] * n_chunks

    scores: list[float] = []
    for row in range(n_chunks):
        offset = row * dims
        dot = 0.0
        row_norm_squared = 0
        for dimension in range(dims):
            value = _int8(docs[offset + dimension])
            dot += query[dimension] * value
            row_norm_squared += value * value
        row_norm = math.sqrt(row_norm_squared)
        scores.append(0.0 if row_norm == 0 else dot / (query_norm * row_norm))
    return scores


def weighted_reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    actual_weights = tuple(weights) if weights is not None else (1.0,) * len(rankings)
    if len(actual_weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    if any(not math.isfinite(weight) or weight < 0 for weight in actual_weights):
        raise ValueError("weights must be finite and non-negative")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, actual_weights, strict=True):
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, document in enumerate(ranking, start=1):
            if document in seen:
                continue
            seen.add(document)
            scores[document] = scores.get(document, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _rollup(chunks: Sequence[ChunkRecord], scores: Sequence[float]) -> list[_RankedChunk]:
    if len(chunks) != len(scores):
        raise ValueError("chunk and score lengths differ")
    best: dict[str, _RankedChunk] = {}
    for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
        document = chunk["document"]
        previous = best.get(document)
        if previous is None or score > previous.score:
            best[document] = _RankedChunk(document, index, score)
    return sorted(best.values(), key=lambda item: (-item.score, item.document))


def _search_result(chunk: ChunkRecord, score: float) -> SearchResult:
    result: SearchResult = {
        "document": chunk["document"],
        "title": chunk["title"],
        "href": chunk["href"],
        "snippet": chunk["snippet"],
        "score": score,
    }
    if "heading" in chunk:
        result["heading"] = chunk["heading"]
    if "anchor" in chunk:
        result["anchor"] = chunk["anchor"]
    return result


def _keyword_scores(query: str, chunks: Sequence[ChunkRecord]) -> list[float]:
    query_terms = tuple(
        dict.fromkeys(
            term
            for term in pre_tokenize(normalize(query))
            if any(character.isalnum() for character in term)
        )
    )
    if not query_terms:
        return []

    chunk_terms = [
        [
            term
            for term in pre_tokenize(
                normalize(f"{chunk['title']} {chunk.get('heading', '')} {chunk['snippet']}")
            )
            if any(character.isalnum() for character in term)
        ]
        for chunk in chunks
    ]
    document_frequency = {
        term: sum(term in set(terms) for terms in chunk_terms) for term in query_terms
    }
    average_length = max(1.0, sum(map(len, chunk_terms)) / max(1, len(chunk_terms)))
    scores: list[float] = []
    for terms in chunk_terms:
        score = 0.0
        for term in query_terms:
            frequency = terms.count(term)
            frequency_weight = (frequency * 2.2) / (
                frequency + 1.2 * (0.25 + 0.75 * (len(terms) / average_length))
            )
            frequency_in_documents = document_frequency[term]
            inverse_frequency = math.log(
                1
                + (len(chunk_terms) - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
            )
            score += inverse_frequency * frequency_weight
        scores.append(score)
    return scores


def _lexical_terms(text: str) -> list[str]:
    return [
        term
        for term in pre_tokenize(normalize(text))
        if any(character.isalnum() for character in term)
    ]


class WebsemClient:
    def __init__(self, artifacts: LoadedArtifacts) -> None:
        self.artifacts = artifacts
        self.tokenizer = WordPieceTokenizer(artifacts.vocabulary)

    def _semantic_ranking(self, query: str, min_score: float) -> list[_RankedChunk]:
        token_ids = self.tokenizer.encode(query)
        if not token_ids:
            return []
        dims = self.artifacts.manifest["dims"]
        vector = embed_query(token_ids, self.artifacts.tokens, self.artifacts.scales, dims)
        scores = cosine_scores(
            vector,
            self.artifacts.docs,
            self.artifacts.manifest["n_chunks"],
            dims,
        )
        return [item for item in _rollup(self.artifacts.chunks, scores) if item.score >= min_score]

    def _keyword_ranking(self, query: str) -> list[_RankedChunk]:
        scores = _keyword_scores(query, self.artifacts.chunks)
        if not scores:
            return []
        return [item for item in _rollup(self.artifacts.chunks, scores) if item.score > 0]

    def _specific_term_ranking(self, query: str) -> list[_RankedChunk]:
        embedded_terms = {
            token.casefold() for token in self.artifacts.vocabulary if not token.startswith("##")
        }
        terms = tuple(
            dict.fromkeys(
                term
                for term in _lexical_terms(query)
                if term.casefold() not in embedded_terms
                and len(term) > 1
                and term.casefold() not in SPECIFIC_TERM_STOP_WORDS
            )
        )
        if not terms:
            return []

        scores: list[float] = []
        for chunk in self.artifacts.chunks:
            title = set(_lexical_terms(chunk["title"]))
            heading = set(_lexical_terms(chunk.get("heading", "")))
            snippet = set(_lexical_terms(chunk["snippet"]))
            scores.append(
                float(
                    sum(
                        4 * (term in title) + 2 * (term in heading) + (term in snippet)
                        for term in terms
                    )
                )
            )
        return [item for item in _rollup(self.artifacts.chunks, scores) if item.score > 0]

    @staticmethod
    def _merge_rankings(*rankings: Sequence[_RankedChunk]) -> list[_RankedChunk]:
        merged: dict[str, _RankedChunk] = {}
        for ranking in rankings:
            for item in ranking:
                merged.setdefault(item.document, item)
        return list(merged.values())

    def semantic_search(
        self, query: str, options: SearchOptions | None = None
    ) -> list[SearchResult]:
        resolved = options or SearchOptions()
        specific = self._specific_term_ranking(query) if resolved.specific_term_heuristic else []
        ranking = self._merge_rankings(specific, self._semantic_ranking(query, resolved.min_score))[
            : resolved.limit
        ]
        return [
            _search_result(self.artifacts.chunks[item.chunk_index], item.score) for item in ranking
        ]

    def keyword_search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        return [
            _search_result(self.artifacts.chunks[item.chunk_index], item.score)
            for item in self._keyword_ranking(query)[:limit]
        ]

    def hybrid_search(
        self, query: str, options: HybridSearchOptions | None = None
    ) -> list[SearchResult]:
        resolved = options or HybridSearchOptions()
        specific = self._specific_term_ranking(query) if resolved.specific_term_heuristic else []
        semantic = self._semantic_ranking(query, resolved.min_score)
        keyword = self._keyword_ranking(query)
        fused = weighted_reciprocal_rank_fusion(
            (
                tuple(item.document for item in specific),
                tuple(item.document for item in semantic),
                tuple(item.document for item in keyword),
            ),
            weights=(1.0, resolved.semantic_weight, resolved.keyword_weight),
            k=resolved.rrf_k,
        )[: resolved.limit]
        specific_chunks = {item.document: item.chunk_index for item in specific}
        semantic_chunks = {item.document: item.chunk_index for item in semantic}
        keyword_chunks = {item.document: item.chunk_index for item in keyword}
        results: list[SearchResult] = []
        for document, score in fused:
            chunk_index = specific_chunks.get(document)
            if chunk_index is None:
                chunk_index = semantic_chunks.get(document)
            if chunk_index is None:
                chunk_index = keyword_chunks[document]
            results.append(_search_result(self.artifacts.chunks[chunk_index], score))
        return results

    semantic = semantic_search
    lexical = keyword_search
    hybrid = hybrid_search
