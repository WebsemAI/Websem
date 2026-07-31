from typing import Literal, NotRequired, TypedDict


class Section(TypedDict):
    text: str
    heading: NotRequired[str]
    anchor: NotRequired[str]


class BuildDocument(TypedDict):
    id: str
    title: str
    href: str
    text: str
    sections: NotRequired[list[Section]]


class ChunkRecord(TypedDict):
    document: str
    title: str
    href: str
    snippet: str
    heading: NotRequired[str]
    anchor: NotRequired[str]


class ArtifactFiles(TypedDict):
    chunks: str
    docs: str
    tokens: str
    scales: str
    vocab: str


class Manifest(TypedDict):
    websem_version: Literal["1.0"]
    model_id: str
    dims: int
    vocab_size: int
    n_chunks: int
    chunk_size: int
    chunk_overlap: int
    title_prefix: bool
    chunker_version: str
    doc_scale: Literal[127]
    files: ArtifactFiles


class SearchResult(TypedDict):
    document: str
    title: str
    href: str
    snippet: str
    score: float
    heading: NotRequired[str]
    anchor: NotRequired[str]
