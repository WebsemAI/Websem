from __future__ import annotations

from fnmatch import fnmatchcase
from functools import cache
from glob import has_magic
from pathlib import Path

from mkdocs.config import base
from mkdocs.config import config_options as c
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from websem_builder import build
from websem_types import BuildDocument

from .html import parse_rendered_page


class WebsemBuildConfig(base.Config):
    index_path = c.Type(str, default="websem/index")
    model_id = c.Type(str, default="minishlab/potion-base-8M")
    dims = c.Type(int, default=128)
    chunk_size = c.Type(int, default=600)
    chunk_overlap = c.Type(int, default=120)
    title_prefix = c.Type(bool, default=True)
    include = c.ListOfItems(c.Type(str), default=[])
    exclude = c.ListOfItems(c.Type(str), default=[])


def _normalize_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _glob_matches(path: str, pattern: str) -> bool:
    path_parts = tuple(path.split("/"))
    pattern_parts = tuple(pattern.split("/"))

    @cache
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        part = pattern_parts[pattern_index]
        if part == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], part)
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def _matches_path(path: str, pattern: str) -> bool:
    normalized_path = _normalize_path(path)
    normalized_pattern = _normalize_path(pattern)
    if not normalized_pattern:
        return False
    if not has_magic(normalized_pattern):
        return normalized_path == normalized_pattern or normalized_path.startswith(
            f"{normalized_pattern}/"
        )
    return _glob_matches(normalized_path, normalized_pattern)


class WebsemBuildPlugin(BasePlugin[WebsemBuildConfig]):
    def __init__(self) -> None:
        super().__init__()
        self.documents: list[BuildDocument] = []

    def on_pre_build(self, *, config: MkDocsConfig) -> None:
        self.documents.clear()

    def on_page_content(
        self,
        html: str,
        *,
        page: Page,
        config: MkDocsConfig,
        files: Files,
    ) -> str:
        source_uri = page.file.src_uri
        if self.config.include and not any(
            _matches_path(source_uri, pattern) for pattern in self.config.include
        ):
            return html
        if any(_matches_path(source_uri, pattern) for pattern in self.config.exclude):
            return html

        text, sections = parse_rendered_page(html)
        if not text:
            return html

        document: BuildDocument = {
            "id": source_uri,
            "title": page.title or source_uri,
            "href": page.url,
            "text": text,
        }
        if sections:
            document["sections"] = sections
        self.documents.append(document)
        return html

    def on_post_build(self, *, config: MkDocsConfig) -> None:
        output_dir = Path(config.site_dir) / self.config.index_path
        build(
            self.documents,
            out_dir=output_dir,
            model_id=self.config.model_id,
            dimensions=self.config.dims,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            title_prefix=self.config.title_prefix,
        )
