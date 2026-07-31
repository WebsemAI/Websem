from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from websem_types import BuildDocument, ChunkRecord, Section

_SPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True, slots=True)
class Chunk:
    record: ChunkRecord
    text: str


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _sentence_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    text = _normalize(text)
    if not text:
        return

    sentences = [sentence for sentence in _SENTENCE_END.split(text) if sentence]
    index = 0
    while index < len(sentences):
        current: list[str] = []
        length = 0
        end = index
        while end < len(sentences):
            next_length = length + len(sentences[end]) + (1 if current else 0)
            if current and next_length > chunk_size:
                break
            current.append(sentences[end])
            length = next_length
            end += 1
        yield " ".join(current)

        if end >= len(sentences):
            break
        overlap_length = 0
        next_index = end
        while next_index > index + 1 and overlap_length < chunk_overlap:
            next_index -= 1
            overlap_length += len(sentences[next_index]) + 1
        index = next_index


def _sections(document: BuildDocument) -> list[Section]:
    sections = document.get("sections")
    if sections:
        return sections
    return [{"text": document["text"]}]


def chunk_documents(
    documents: Iterable[BuildDocument],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    title_prefix: bool = True,
) -> list[Chunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    document_ids: set[str] = set()
    for document in documents:
        if not document["id"] or not document["title"]:
            raise ValueError("document id and title must not be empty")
        if document["id"] in document_ids:
            raise ValueError(f"duplicate document id: {document['id']}")
        document_ids.add(document["id"])
        for section in _sections(document):
            for snippet in _sentence_chunks(section["text"], chunk_size, chunk_overlap):
                record: ChunkRecord = {
                    "document": document["id"],
                    "title": document["title"],
                    "href": document["href"],
                    "snippet": snippet,
                }
                if "heading" in section:
                    record["heading"] = section["heading"]
                if "anchor" in section:
                    record["anchor"] = section["anchor"]

                text = f"{document['title']}\n\n{snippet}" if title_prefix else snippet
                chunks.append(Chunk(record=record, text=text))
    return chunks
