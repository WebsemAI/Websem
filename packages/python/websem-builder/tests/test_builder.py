from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from websem_builder import build_from_arrays, build_from_model, chunk_documents
from websem_types import BuildDocument, Manifest

ArtifactName = Literal["chunks", "docs", "tokens", "scales", "vocab"]


class ToyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"second": 1, "first": 0, "zero": 2}


class ToyModel:
    embedding = np.array([[2.0, -1.0], [0.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    tokenizer = ToyTokenizer()

    def encode(self, sentences: Sequence[str]) -> npt.NDArray[np.float32]:
        return _encoder(sentences)


def _encoder(texts: Sequence[str]) -> npt.NDArray[np.float32]:
    vectors = np.array(
        [[1.0, 0.0] if "First" in text else [0.0, 1.0] for text in texts], dtype=np.float32
    )
    return vectors


def _document() -> BuildDocument:
    return {
        "id": "guide",
        "title": "Guide",
        "href": "/guide",
        "text": "ignored",
        "sections": [
            {
                "heading": "Start",
                "anchor": "start",
                "text": "First sentence. Second sentence. Third sentence.",
            }
        ],
    }


def _artifact(output: Path, manifest: Manifest, name: ArtifactName) -> bytes:
    filename = manifest["files"][name]
    content = (output / filename).read_bytes()
    assert hashlib.sha256(content).hexdigest()[:12] in filename
    return content


def test_chunks_are_section_aware_snapped_and_prefixed() -> None:
    chunks = chunk_documents([_document()], chunk_size=35, chunk_overlap=10)

    assert [chunk.record["snippet"] for chunk in chunks] == [
        "First sentence. Second sentence.",
        "Second sentence. Third sentence.",
    ]
    assert chunks[0].text == "Guide\n\nFirst sentence. Second sentence."
    assert chunks[0].record["heading"] == "Start"
    assert chunks[0].record["anchor"] == "start"


def test_chunks_allow_a_root_document_href() -> None:
    document = _document()
    document["href"] = ""

    chunks = chunk_documents([document])

    assert chunks[0].record["href"] == ""


def test_build_writes_hashed_raw_artifacts_and_manifest(tmp_path: Path) -> None:
    result = build_from_model(
        [_document()],
        model=ToyModel(),
        model_id="toy/model",
        out_dir=tmp_path,
        chunk_size=35,
        chunk_overlap=10,
    )
    manifest = result.manifest

    assert result.path.name == "manifest.json"
    assert manifest["dims"] == 2
    assert manifest["n_chunks"] == 2
    assert set(manifest) == {
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
    assert json.loads(result.path.read_bytes()) == manifest
    assert json.loads(_artifact(tmp_path, manifest, "vocab")) == ["first", "second", "zero"]
    assert np.frombuffer(_artifact(tmp_path, manifest, "docs"), dtype=np.int8).reshape(
        -1, 2
    ).tolist() == [
        [127, 0],
        [0, 127],
    ]
    assert np.frombuffer(_artifact(tmp_path, manifest, "tokens"), dtype=np.int8).reshape(
        -1, 2
    ).tolist() == [
        [127, -64],
        [0, 127],
        [0, 0],
    ]
    scales = np.frombuffer(_artifact(tmp_path, manifest, "scales"), dtype="<f4")
    np.testing.assert_allclose(scales, [2 / 127, 4 / 127, 1])
    assert all(
        re.fullmatch(r"(?:chunks|docs|tokens|scales|vocab)\.[0-9a-f]{12}\.(?:bin|json)", name)
        for name in (
            manifest["files"]["chunks"],
            manifest["files"]["docs"],
            manifest["files"]["tokens"],
            manifest["files"]["scales"],
            manifest["files"]["vocab"],
        )
    )


def test_build_rejects_non_normalized_document_vectors(tmp_path: Path) -> None:
    def unnormalized(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.full((len(texts), 2), 1.0, dtype=np.float32)

    try:
        build_from_arrays(
            [_document()],
            model_id="toy/model",
            out_dir=tmp_path,
            token_embeddings=np.ones((1, 2), dtype=np.float32),
            vocab=["token"],
            encoder=unnormalized,
            chunk_size=35,
            chunk_overlap=10,
        )
    except ValueError as error:
        assert "must be normalized" in str(error)
    else:
        raise AssertionError("expected normalized-vector validation")


def test_cleanup_only_removes_obsolete_hashed_artifacts(tmp_path: Path) -> None:
    keep = tmp_path / "docs.notes.bin"
    keep.write_bytes(b"keep")
    obsolete = tmp_path / f"docs.{'0' * 12}.bin"
    obsolete.write_bytes(b"old")

    build_from_arrays(
        [],
        model_id="toy/model",
        out_dir=tmp_path,
        token_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        vocab=["token"],
        encoder=_encoder,
    )

    assert keep.read_bytes() == b"keep"
    assert not obsolete.exists()
