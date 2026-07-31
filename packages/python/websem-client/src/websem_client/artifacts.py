import asyncio
import json
import math
import struct
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import cast

from websem_types import ArtifactFiles, ChunkRecord, Manifest

from .tokenizer import WordPieceTokenizer

AsyncFetcher = Callable[[str], Awaitable[bytes]]

_MANIFEST_KEYS = {
    "websem_version",
    "model_id",
    "dims",
    "vocab_size",
    "n_chunks",
    "chunk_size",
    "chunk_overlap",
    "title_prefix",
    "chunker_version",
    "doc_scale",
    "files",
}
_FILE_KEYS = {"chunks", "docs", "tokens", "scales", "vocab"}
_CHUNK_REQUIRED_KEYS = {"document", "title", "href", "snippet"}
_CHUNK_OPTIONAL_KEYS = {"heading", "anchor"}


@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    manifest: Manifest
    chunks: tuple[ChunkRecord, ...]
    docs: bytes
    tokens: bytes
    scales: tuple[float, ...]
    vocabulary: tuple[str, ...]

    def tokenizer(self) -> WordPieceTokenizer:
        return WordPieceTokenizer(self.vocabulary)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")


def _string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = " non-empty" if nonempty else ""
        raise ValueError(f"{label} must be a{suffix} string")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _artifact_name(value: object, label: str) -> str:
    name = _string(value, label, nonempty=True)
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"{label} must be a relative artifact path")
    return name


def _parse_manifest(raw: bytes) -> Manifest:
    parsed: object = json.loads(raw.decode("utf-8"))
    value = _mapping(parsed, "manifest")
    _exact_keys(value, _MANIFEST_KEYS, "manifest")
    if value["websem_version"] != "1.0":
        raise ValueError(f"unsupported websem_version: {value['websem_version']!r}")
    if value["doc_scale"] != 127 or isinstance(value["doc_scale"], bool):
        raise ValueError("manifest.doc_scale must be 127")

    raw_files = _mapping(value["files"], "manifest.files")
    _exact_keys(raw_files, _FILE_KEYS, "manifest.files")
    files: ArtifactFiles = {
        "chunks": _artifact_name(raw_files["chunks"], "manifest.files.chunks"),
        "docs": _artifact_name(raw_files["docs"], "manifest.files.docs"),
        "tokens": _artifact_name(raw_files["tokens"], "manifest.files.tokens"),
        "scales": _artifact_name(raw_files["scales"], "manifest.files.scales"),
        "vocab": _artifact_name(raw_files["vocab"], "manifest.files.vocab"),
    }
    return {
        "websem_version": "1.0",
        "model_id": _string(value["model_id"], "manifest.model_id", nonempty=True),
        "dims": _integer(value["dims"], "manifest.dims", minimum=1),
        "vocab_size": _integer(value["vocab_size"], "manifest.vocab_size", minimum=1),
        "n_chunks": _integer(value["n_chunks"], "manifest.n_chunks", minimum=0),
        "chunk_size": _integer(value["chunk_size"], "manifest.chunk_size", minimum=1),
        "chunk_overlap": _integer(value["chunk_overlap"], "manifest.chunk_overlap", minimum=0),
        "title_prefix": _boolean(value["title_prefix"], "manifest.title_prefix"),
        "chunker_version": _string(value["chunker_version"], "manifest.chunker_version"),
        "doc_scale": 127,
        "files": files,
    }


def _parse_chunks(raw: bytes, expected_length: int) -> tuple[ChunkRecord, ...]:
    parsed: object = json.loads(raw.decode("utf-8"))
    values = _list(parsed, "chunks")
    if len(values) != expected_length:
        raise ValueError(f"chunks has {len(values)} entries, expected {expected_length}")

    chunks: list[ChunkRecord] = []
    allowed_keys = _CHUNK_REQUIRED_KEYS | _CHUNK_OPTIONAL_KEYS
    for index, raw_chunk in enumerate(values):
        chunk = _mapping(raw_chunk, f"chunks[{index}]")
        actual_keys = set(chunk)
        if not actual_keys >= _CHUNK_REQUIRED_KEYS or not actual_keys <= allowed_keys:
            missing = sorted(_CHUNK_REQUIRED_KEYS - actual_keys)
            extra = sorted(actual_keys - allowed_keys)
            raise ValueError(f"chunks[{index}] keys differ: missing={missing}, extra={extra}")
        record: ChunkRecord = {
            "document": _string(chunk["document"], f"chunks[{index}].document"),
            "title": _string(chunk["title"], f"chunks[{index}].title"),
            "href": _string(chunk["href"], f"chunks[{index}].href"),
            "snippet": _string(chunk["snippet"], f"chunks[{index}].snippet"),
        }
        if "heading" in chunk:
            record["heading"] = _string(chunk["heading"], f"chunks[{index}].heading")
        if "anchor" in chunk:
            record["anchor"] = _string(chunk["anchor"], f"chunks[{index}].anchor")
        chunks.append(record)
    return tuple(chunks)


def _parse_vocabulary(raw: bytes, expected_length: int) -> tuple[str, ...]:
    parsed: object = json.loads(raw.decode("utf-8"))
    values = _list(parsed, "vocabulary")
    if len(values) != expected_length:
        raise ValueError(f"vocabulary has {len(values)} entries, expected {expected_length}")
    vocabulary: list[str] = []
    for index, value in enumerate(values):
        vocabulary.append(_string(value, f"vocabulary[{index}]"))
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary tokens must be unique")
    return tuple(vocabulary)


def _assemble_artifacts(
    manifest: Manifest,
    *,
    chunks: bytes,
    docs: bytes,
    tokens: bytes,
    scales: bytes,
    vocab: bytes,
) -> LoadedArtifacts:
    expected_docs = manifest["n_chunks"] * manifest["dims"]
    expected_tokens = manifest["vocab_size"] * manifest["dims"]
    expected_scales = manifest["vocab_size"] * 4
    if len(docs) != expected_docs:
        raise ValueError(f"docs byte length is {len(docs)}, expected {expected_docs}")
    if len(tokens) != expected_tokens:
        raise ValueError(f"tokens byte length is {len(tokens)}, expected {expected_tokens}")
    if len(scales) != expected_scales:
        raise ValueError(f"scales byte length is {len(scales)}, expected {expected_scales}")

    scale_values = tuple(value for (value,) in struct.iter_unpack("<f", scales))
    if any(not math.isfinite(value) or value <= 0 for value in scale_values):
        raise ValueError("scales must contain finite positive float32 values")
    return LoadedArtifacts(
        manifest=manifest,
        chunks=_parse_chunks(chunks, manifest["n_chunks"]),
        docs=docs,
        tokens=tokens,
        scales=scale_values,
        vocabulary=_parse_vocabulary(vocab, manifest["vocab_size"]),
    )


def load_artifacts(path: str | PathLike[str]) -> LoadedArtifacts:
    base = Path(path)
    manifest = _parse_manifest((base / "manifest.json").read_bytes())
    files = manifest["files"]
    return _assemble_artifacts(
        manifest,
        chunks=(base / files["chunks"]).read_bytes(),
        docs=(base / files["docs"]).read_bytes(),
        tokens=(base / files["tokens"]).read_bytes(),
        scales=(base / files["scales"]).read_bytes(),
        vocab=(base / files["vocab"]).read_bytes(),
    )


def _join_location(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}" if base else name


async def load_artifacts_from_fetch(base: str, fetch: AsyncFetcher) -> LoadedArtifacts:
    manifest_raw = await fetch(_join_location(base, "manifest.json"))
    if not isinstance(manifest_raw, bytes):
        raise TypeError("fetch must return bytes")
    manifest = _parse_manifest(manifest_raw)
    files = manifest["files"]
    names = (files["chunks"], files["docs"], files["tokens"], files["scales"], files["vocab"])
    fetched = await asyncio.gather(*(fetch(_join_location(base, name)) for name in names))
    if not all(isinstance(value, bytes) for value in fetched):
        raise TypeError("fetch must return bytes")
    chunks, docs, tokens, scales, vocab = fetched
    return _assemble_artifacts(
        manifest,
        chunks=chunks,
        docs=docs,
        tokens=tokens,
        scales=scales,
        vocab=vocab,
    )
