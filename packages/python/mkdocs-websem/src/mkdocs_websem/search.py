from __future__ import annotations

import json
import posixpath
from importlib.resources import files
from pathlib import Path, PurePosixPath
from shutil import copyfileobj

from mkdocs.config import base
from mkdocs.config import config_options as c
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin

_ASSET_PATH = "websem/websem-search.mjs"
_CONFIG_PATH = "websem/websem-search.json"


class WebsemSearchConfig(base.Config):
    index_path = c.Type(str, default="websem/index")
    mode = c.Choice(("semantic", "hybrid"), default="hybrid")
    limit = c.Type(int, default=10)
    min_score = c.Type(float, default=0.2)
    rrf_k = c.Type(int, default=60)
    factor = c.Type(float, default=1.0)
    keyword_weight = c.Type(float, default=1.0)
    specific_term_heuristic = c.Type(bool, default=True)


class WebsemSearchPlugin(BasePlugin[WebsemSearchConfig]):
    def on_config(self, config: MkDocsConfig) -> MkDocsConfig:
        if not any(
            str(script) == _ASSET_PATH
            or isinstance(script, dict)
            and script.get("path") == _ASSET_PATH
            for script in config.extra_javascript
        ):
            script = c.ExtraScriptValue(_ASSET_PATH)
            script.type = "module"
            config.extra_javascript.append(script)
        return config

    def on_post_build(self, *, config: MkDocsConfig) -> None:
        site_dir = Path(config.site_dir)
        asset_path = site_dir / _ASSET_PATH
        config_path = site_dir / _CONFIG_PATH
        asset_path.parent.mkdir(parents=True, exist_ok=True)

        source = files("mkdocs_websem").joinpath("assets/websem-search.mjs")
        with source.open("rb") as source_file, asset_path.open("wb") as target_file:
            copyfileobj(source_file, target_file)

        asset_parent = PurePosixPath(_ASSET_PATH).parent.as_posix()
        manifest_path = posixpath.join(self.config.index_path.strip("/"), "manifest.json")
        manifest_url = posixpath.relpath(manifest_path, asset_parent)
        site_root_url = posixpath.relpath(".", asset_parent) + "/"
        browser_config = {
            "manifestUrl": manifest_url,
            "siteRootUrl": site_root_url,
            "mode": self.config.mode,
            "limit": self.config.limit,
            "minScore": self.config.min_score,
            "rrfK": self.config.rrf_k,
            "semanticWeight": self.config.factor,
            "keywordWeight": self.config.keyword_weight,
            "specificTermHeuristic": self.config.specific_term_heuristic,
        }
        config_path.write_text(json.dumps(browser_config, separators=(",", ":")), encoding="utf-8")
