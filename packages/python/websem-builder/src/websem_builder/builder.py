from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from model2vec import StaticModel
from websem_types import ArtifactFiles, BuildDocument, Manifest

from .chunking import chunk_documents

CHUNKER_VERSION = "1"
DOC_SCALE: Final[Literal[127]] = 127
_ARTIFACT_PATTERN = re.compile(r"^(?:chunks|docs|tokens|scales|vocab)\.[0-9a-f]{12}\.(?:bin|json)$")

Encoder = Callable[[Sequence[str]], npt.NDArray[np.floating]]


class Tokenizer(Protocol):
    def get_vocab(self) -> dict[str, int]: ...


class PortableModel(Protocol):
    @property
    def embedding(self) -> npt.ArrayLike: ...

    @property
    def tokenizer(self) -> Tokenizer: ...

    def encode(self, sentences: Sequence[str]) -> npt.NDArray[np.generic]: ...


@dataclass(frozen=True, slots=True)
class BuildResult:
    manifest: Manifest
    path: Path


def _ordered_vocab(vocab: Mapping[str, int] | Sequence[str]) -> list[str]:
    if isinstance(vocab, Mapping):
        ids = sorted(vocab.values())
        if ids != list(range(len(vocab))):
            raise ValueError("vocab token ids must be contiguous from zero")
        return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    return list(vocab)


def _model_embeddings(model: PortableModel, vocab_size: int) -> npt.NDArray[np.float32]:
    embeddings = np.asarray(model.embedding, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != vocab_size:
        raise ValueError("token embeddings must have one row per vocabulary token")
    return embeddings


def _encode_documents(texts: Sequence[str], encoder: Encoder, dims: int) -> npt.NDArray[np.float32]:
    if not texts:
        return np.empty((0, dims), dtype=np.float32)

    encoded = np.asarray(encoder(texts), dtype=np.float32)
    if encoded.shape != (len(texts), dims):
        raise ValueError(f"encoder returned shape {encoded.shape}, expected {(len(texts), dims)}")
    if not np.isfinite(encoded).all():
        raise ValueError("document embeddings must contain only finite values")

    norms = np.linalg.norm(encoded, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        bad_rows = np.flatnonzero(~np.isclose(norms, 1.0, rtol=1e-4, atol=1e-4))
        raise ValueError(
            f"document embeddings must be normalized; invalid rows: {bad_rows.tolist()}"
        )
    return encoded


def _quantize_tokens(
    embeddings: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]:
    if not np.isfinite(embeddings).all():
        raise ValueError("token embeddings must contain only finite values")
    max_values = np.max(np.abs(embeddings), axis=1)
    scales = np.where(max_values > 0, max_values / DOC_SCALE, 1.0).astype("<f4")
    divided = np.zeros_like(embeddings)
    np.divide(embeddings, scales[:, None], out=divided, where=scales[:, None] != 0)
    quantized = np.clip(np.rint(divided), -DOC_SCALE, DOC_SCALE).astype(np.int8)
    return quantized, scales


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _write_artifact(out_dir: Path, stem: str, suffix: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()[:12]
    filename = f"{stem}.{digest}.{suffix}"
    path = out_dir / filename
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"content hash collision at {path}")
    else:
        path.write_bytes(content)
    return filename


def _clean_artifacts(out_dir: Path, current: set[str]) -> None:
    for path in out_dir.iterdir():
        if path.is_file() and _ARTIFACT_PATTERN.fullmatch(path.name) and path.name not in current:
            path.unlink()


def build_from_arrays(
    documents: Iterable[BuildDocument],
    *,
    model_id: str,
    out_dir: str | Path,
    token_embeddings: npt.ArrayLike,
    vocab: Mapping[str, int] | Sequence[str],
    encoder: Encoder,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    title_prefix: bool = True,
) -> BuildResult:
    if not model_id:
        raise ValueError("model_id must not be empty")

    ordered_vocab = _ordered_vocab(vocab)
    embeddings = np.asarray(token_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ordered_vocab):
        raise ValueError("token embeddings must have one row per vocabulary token")
    if embeddings.shape[1] < 1 or not ordered_vocab:
        raise ValueError("token embeddings and vocab must not be empty")

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        title_prefix=title_prefix,
    )
    encoded = _encode_documents([chunk.text for chunk in chunks], encoder, embeddings.shape[1])
    docs = np.clip(np.rint(encoded * DOC_SCALE), -DOC_SCALE, DOC_SCALE).astype(np.int8)
    tokens, scales = _quantize_tokens(embeddings)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: ArtifactFiles = {
        "chunks": _write_artifact(
            output, "chunks", "json", _json_bytes([chunk.record for chunk in chunks])
        ),
        "docs": _write_artifact(output, "docs", "bin", docs.tobytes(order="C")),
        "tokens": _write_artifact(output, "tokens", "bin", tokens.tobytes(order="C")),
        "scales": _write_artifact(output, "scales", "bin", scales.tobytes(order="C")),
        "vocab": _write_artifact(output, "vocab", "json", _json_bytes(ordered_vocab)),
    }
    manifest: Manifest = {
        "websem_version": "1.0",
        "model_id": model_id,
        "dims": embeddings.shape[1],
        "vocab_size": len(ordered_vocab),
        "n_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "title_prefix": title_prefix,
        "chunker_version": CHUNKER_VERSION,
        "doc_scale": DOC_SCALE,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_bytes(_json_bytes(manifest))
    temporary_manifest.replace(manifest_path)
    _clean_artifacts(
        output,
        {files["chunks"], files["docs"], files["tokens"], files["scales"], files["vocab"]},
    )
    return BuildResult(manifest=manifest, path=manifest_path)


def export_model(
    *,
    model_id: str,
    out_dir: str | Path,
    dimensions: int | None = None,
) -> Path:
    model = StaticModel.from_pretrained(model_id, dimensionality=dimensions, normalize=True)
    portable_model = cast(PortableModel, model)
    vocabulary = _ordered_vocab(portable_model.tokenizer.get_vocab())
    embeddings = _model_embeddings(portable_model, len(vocabulary))
    tokens, scales = _quantize_tokens(embeddings)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocab_name = _write_artifact(output, "vocab", "json", _json_bytes(vocabulary))
    tokens_name = _write_artifact(output, "tokens", "bin", tokens.tobytes(order="C"))
    scales_name = _write_artifact(output, "scales", "bin", scales.tobytes(order="C"))
    descriptor = {
        "model_id": model_id,
        "dims": embeddings.shape[1],
        "vocab": vocab_name,
        "tokens": tokens_name,
        "scales": scales_name,
    }
    descriptor_path = output / "model.json"
    descriptor_path.write_bytes(_json_bytes(descriptor))
    return descriptor_path


def build_from_model(
    documents: Iterable[BuildDocument],
    *,
    model: PortableModel,
    model_id: str,
    out_dir: str | Path,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    title_prefix: bool = True,
) -> BuildResult:
    vocab = model.tokenizer.get_vocab()
    embeddings = _model_embeddings(model, len(vocab))
    return build_from_arrays(
        documents,
        model_id=model_id,
        out_dir=out_dir,
        token_embeddings=embeddings,
        vocab=vocab,
        encoder=cast(Encoder, model.encode),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        title_prefix=title_prefix,
    )


def build(
    documents: Iterable[BuildDocument],
    *,
    model_id: str,
    out_dir: str | Path,
    dimensions: int | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    title_prefix: bool = True,
) -> BuildResult:
    model = StaticModel.from_pretrained(model_id, dimensionality=dimensions, normalize=True)
    return build_from_model(
        documents,
        model=cast(PortableModel, model),
        model_id=model_id,
        out_dir=out_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        title_prefix=title_prefix,
    )
