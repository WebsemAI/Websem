from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mkdocs_websem.build import WebsemBuildPlugin, _matches_path


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("README.md", "README.md", True),
        ("components/forms/input.md", "components", True),
        ("components/forms/input.md", "components/", True),
        ("components/forms/input.md", "components/*.md", False),
        ("components/forms/input.md", "components/**/*.md", True),
        ("README.md", "**/*.md", True),
        ("api/types/Button.md", "api/**", True),
        ("patterns/filter.md", "components/**", False),
        ("components/forms/input.md", r"components\forms\*.md", True),
    ],
)
def test_path_patterns(path: str, pattern: str, expected: bool) -> None:
    assert _matches_path(path, pattern) is expected


def test_build_plugin_collects_pages_and_invokes_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "mkdocs_websem.build.build",
        lambda documents, **kwargs: calls.append({"documents": documents, **kwargs}),
    )

    plugin = WebsemBuildPlugin()
    errors, warnings = plugin.load_config({"model_id": "test/model", "dims": 8})
    assert not errors
    assert not warnings

    mkdocs_config = SimpleNamespace(site_dir=str(tmp_path))
    page = SimpleNamespace(title="Guide", url="guide/", file=SimpleNamespace(src_uri="guide.md"))
    plugin.on_pre_build(config=mkdocs_config)
    html = '<h1 id="guide">Guide</h1><p>Useful text.</p><pre>ignored</pre>'
    assert plugin.on_page_content(html, page=page, config=mkdocs_config, files=[]) == html
    plugin.on_post_build(config=mkdocs_config)

    assert calls == [
        {
            "documents": [
                {
                    "id": "guide.md",
                    "title": "Guide",
                    "href": "guide/",
                    "text": "Guide Useful text.",
                    "sections": [{"text": "Useful text.", "heading": "Guide", "anchor": "guide"}],
                }
            ],
            "out_dir": tmp_path / "websem/index",
            "model_id": "test/model",
            "dimensions": 8,
            "chunk_size": 600,
            "chunk_overlap": 120,
            "title_prefix": True,
        }
    ]


def test_build_plugin_applies_include_and_exclude_patterns() -> None:
    plugin = WebsemBuildPlugin()
    errors, warnings = plugin.load_config(
        {
            "include": ["README.md", "components/", "patterns/**/*.md"],
            "exclude": ["components/generated/**"],
        }
    )
    assert not errors
    assert not warnings
    plugin.on_pre_build(config=SimpleNamespace())

    html = "<p>Indexed text.</p>"
    for source_uri in (
        "README.md",
        "components/button.md",
        "components/generated/Button.md",
        "patterns/forms/filter.md",
        "architecture/theming.md",
    ):
        page = SimpleNamespace(
            title=source_uri,
            url=source_uri,
            file=SimpleNamespace(src_uri=source_uri),
        )
        plugin.on_page_content(html, page=page, config=SimpleNamespace(), files=[])

    assert [document["id"] for document in plugin.documents] == [
        "README.md",
        "components/button.md",
        "patterns/forms/filter.md",
    ]
