from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mkdocs_websem.search import WebsemSearchPlugin


def test_search_plugin_injects_module_and_copies_config(tmp_path: Path) -> None:
    plugin = WebsemSearchPlugin()
    errors, warnings = plugin.load_config(
        {
            "mode": "semantic",
            "limit": 7,
            "min_score": 0.2,
            "rrf_k": 40,
            "factor": 1.5,
            "keyword_weight": 0.5,
        }
    )
    assert not errors
    assert not warnings

    config = SimpleNamespace(site_dir=str(tmp_path), extra_javascript=[])
    assert plugin.on_config(config) is config
    assert str(config.extra_javascript[0]) == "websem/websem-search.mjs"
    assert config.extra_javascript[0].type == "module"

    plugin.on_post_build(config=config)

    assert (tmp_path / "websem/websem-search.mjs").is_file()
    browser_config = json.loads(
        (tmp_path / "websem/websem-search.json").read_text(encoding="utf-8")
    )
    assert browser_config == {
        "manifestUrl": "index/manifest.json",
        "siteRootUrl": "../",
        "mode": "semantic",
        "limit": 7,
        "minScore": 0.2,
        "rrfK": 40,
        "semanticWeight": 1.5,
        "keywordWeight": 0.5,
        "specificTermHeuristic": True,
    }


def test_search_plugin_does_not_inject_twice() -> None:
    plugin = WebsemSearchPlugin()
    plugin.load_config({})
    config = SimpleNamespace(site_dir="site", extra_javascript=[])

    plugin.on_config(config)
    plugin.on_config(config)

    assert len(config.extra_javascript) == 1
