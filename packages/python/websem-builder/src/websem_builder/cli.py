from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from websem_types import BuildDocument

from .builder import build, export_model


def _document(value: object) -> BuildDocument:
    if not isinstance(value, dict):
        raise ValueError("each document must be a JSON object")
    required = {"id", "title", "href", "text"}
    if not required <= value.keys() or not value.keys() <= required | {"sections"}:
        raise ValueError("each document requires id, title, href, text, and optional sections")
    if not all(isinstance(value[key], str) for key in required):
        raise ValueError("document id, title, href, and text must be strings")
    sections = value.get("sections")
    if sections is not None:
        if not isinstance(sections, list):
            raise ValueError("document sections must be a list")
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("each section must be a JSON object")
            if "text" not in section or not section.keys() <= {"text", "heading", "anchor"}:
                raise ValueError("each section requires text and optional heading and anchor")
            if not all(isinstance(item, str) for item in section.values()):
                raise ValueError("section text, heading, and anchor must be strings")
    return cast(BuildDocument, value)


def _load_documents(path: Path) -> list[BuildDocument]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            _document(json.loads(line)) for line in path.read_text().splitlines() if line.strip()
        ]

    value: object = json.loads(path.read_text())
    if isinstance(value, list):
        return [_document(document) for document in value]
    return [_document(value)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build websem browser search artifacts")
    parser.add_argument("corpus", type=Path, help="JSON or JSONL document corpus")
    parser.add_argument("--model", required=True, help="model2vec model name or path")
    parser.add_argument("--out", required=True, type=Path, help="artifact output directory")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--no-title-prefix", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    build(
        _load_documents(args.corpus),
        model_id=args.model,
        out_dir=args.out,
        dimensions=args.dimensions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        title_prefix=not args.no_title_prefix,
    )


def export_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a model2vec model for TypeScript builds")
    parser.add_argument("--model", required=True, help="model2vec model name or path")
    parser.add_argument("--out", required=True, type=Path, help="model output directory")
    parser.add_argument("--dimensions", type=int, default=128)
    args = parser.parse_args(argv)
    export_model(model_id=args.model, out_dir=args.out, dimensions=args.dimensions)


if __name__ == "__main__":
    main()
