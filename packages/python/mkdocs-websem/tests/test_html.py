from mkdocs_websem.html import parse_rendered_page


def test_parse_rendered_page_collects_sections_and_excludes_code() -> None:
    text, sections = parse_rendered_page(
        """
        <p>Introduction</p>
        <h2 id="install">Install</h2>
        <p>Run the command.</p><pre><code>secret<br>--token</code></pre>
        <h2 id="usage">Usage</h2><p data-search-exclude>Hidden</p><p>Open it.</p>
        """
    )

    assert text == "Introduction Install Run the command. Usage Open it."
    assert sections == [
        {"text": "Introduction"},
        {"text": "Run the command.", "heading": "Install", "anchor": "install"},
        {"text": "Open it.", "heading": "Usage", "anchor": "usage"},
    ]
