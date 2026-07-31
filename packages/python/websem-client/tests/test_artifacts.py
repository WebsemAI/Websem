import asyncio
import json
import struct
from pathlib import Path

import pytest
from websem_client import load_artifacts, load_artifacts_from_fetch


def _artifact_files() -> dict[str, bytes]:
    manifest = {
        "websem_version": "1.0",
        "model_id": "test/model",
        "dims": 2,
        "vocab_size": 2,
        "n_chunks": 2,
        "chunk_size": 600,
        "chunk_overlap": 120,
        "title_prefix": True,
        "chunker_version": "1",
        "doc_scale": 127,
        "files": {
            "chunks": "chunks.hash.json",
            "docs": "docs.hash.bin",
            "tokens": "tokens.hash.bin",
            "scales": "scales.hash.bin",
            "vocab": "vocab.hash.json",
        },
    }
    chunks = [
        {"document": "a", "title": "Alpha", "href": "/a", "snippet": "first"},
        {"document": "b", "title": "Beta", "href": "/b", "snippet": "second"},
    ]
    return {
        "manifest.json": json.dumps(manifest).encode(),
        "chunks.hash.json": json.dumps(chunks).encode(),
        "docs.hash.bin": bytes((127, 0, 0, 127)),
        "tokens.hash.bin": bytes((127, 0, 0, 127)),
        "scales.hash.bin": struct.pack("<2f", 0.5, 0.25),
        "vocab.hash.json": b'["alpha", "beta"]',
    }


def test_loads_filesystem_and_async_fetch_artifacts(tmp_path: Path) -> None:
    files = _artifact_files()
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    loaded = load_artifacts(tmp_path)
    assert loaded.scales == (0.5, 0.25)
    assert loaded.vocabulary == ("alpha", "beta")

    async def fetch(location: str) -> bytes:
        return files[location.removeprefix("/search/")]

    fetched = asyncio.run(load_artifacts_from_fetch("/search", fetch))
    assert fetched == loaded


def test_rejects_artifact_length_mismatch(tmp_path: Path) -> None:
    files = _artifact_files()
    files["docs.hash.bin"] = b"\x00"
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    with pytest.raises(ValueError, match="docs byte length is 1, expected 4"):
        load_artifacts(tmp_path)


def test_rejects_unknown_manifest_version(tmp_path: Path) -> None:
    files = _artifact_files()
    manifest = json.loads(files["manifest.json"])
    manifest["websem_version"] = "2.0"
    files["manifest.json"] = json.dumps(manifest).encode()
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    with pytest.raises(ValueError, match="unsupported websem_version"):
        load_artifacts(tmp_path)
