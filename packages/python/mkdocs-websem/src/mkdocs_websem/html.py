from __future__ import annotations

import re
from html.parser import HTMLParser

from websem_types import Section

_SPACE = re.compile(r"\s+")
_EXCLUDED_TAGS = {"code", "kbd", "math", "noscript", "pre", "samp", "script", "style", "svg"}
_HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


def _clean(parts: list[str]) -> str:
    return _SPACE.sub(" ", " ".join(parts)).strip()


class RenderedPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._excluded_tags: list[str] = []
        self._heading_tag: str | None = None
        self._heading_anchor: str | None = None
        self._heading_parts: list[str] = []
        self._section_heading: str | None = None
        self._section_anchor: str | None = None
        self._section_parts: list[str] = []
        self._all_parts: list[str] = []
        self.sections: list[Section] = []

    @property
    def text(self) -> str:
        return _clean(self._all_parts)

    def finish(self) -> None:
        self._finish_section()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        excluded = (
            tag in _EXCLUDED_TAGS
            or "no-websem" in classes
            or "data-websem-exclude" in attributes
            or "data-search-exclude" in attributes
        )
        if self._excluded_tags or excluded:
            if tag not in _VOID_TAGS:
                self._excluded_tags.append(tag)
            return

        if tag in _HEADING_TAGS:
            self._finish_section()
            self._heading_tag = tag
            self._heading_anchor = attributes.get("id")
            self._heading_parts = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if self._excluded_tags:
            if tag == self._excluded_tags[-1]:
                self._excluded_tags.pop()
            return

        if tag == self._heading_tag:
            heading = _clean(self._heading_parts)
            self._section_heading = heading or None
            self._section_anchor = self._heading_anchor
            self._heading_tag = None
            self._heading_anchor = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._excluded_tags or not data.strip():
            return

        self._all_parts.append(data)
        if self._heading_tag is not None:
            self._heading_parts.append(data)
        else:
            self._section_parts.append(data)

    def _finish_section(self) -> None:
        text = _clean(self._section_parts)
        if text:
            section: Section = {"text": text}
            if self._section_heading:
                section["heading"] = self._section_heading
            if self._section_anchor:
                section["anchor"] = self._section_anchor
            self.sections.append(section)
        self._section_parts = []
        self._section_heading = None
        self._section_anchor = None


def parse_rendered_page(html: str) -> tuple[str, list[Section]]:
    parser = RenderedPageParser()
    parser.feed(html)
    parser.close()
    parser.finish()
    return parser.text, parser.sections
