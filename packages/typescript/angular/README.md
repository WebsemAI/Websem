# @websem/angular

Angular v22+ WebMCP tools backed by
[`@websem/client`](https://www.npmjs.com/package/@websem/client).

## Install

```bash
npm install @websem/angular
```

## Packages

TypeScript: [`@websem/client`](https://www.npmjs.com/package/@websem/client),
[`@websem/builder`](https://www.npmjs.com/package/@websem/builder),
[`@websem/angular`](https://www.npmjs.com/package/@websem/angular), and
[`@websem/types`](https://www.npmjs.com/package/@websem/types).

Python: [`websem-client`](https://pypi.org/project/websem-client/),
[`websem-builder`](https://pypi.org/project/websem-builder/),
[`mkdocs-websem`](https://pypi.org/project/mkdocs-websem/), and
[`websem-types`](https://pypi.org/project/websem-types/).

The adapter requires a Websem index with prebuilt embeddings. Build the index ahead of
time with [`@websem/builder`](https://www.npmjs.com/package/@websem/builder) or
[`websem-builder`](https://pypi.org/project/websem-builder/), serve its generated files
as static assets, and point `indexUrl` to that directory. When using `artifacts` or
`client` instead, they must already contain the built index.

The config mirrors `sdl-mcp`: tool instructions, descriptions, query descriptions, and
result messages come from `texts.DOC_*` and optional `texts.ICON_*` fields. Set exactly
one of `indexUrl`, `artifacts`, or `client` for documentation search. Icon search uses
the matching `iconIndexUrl`, `iconArtifacts`, or `iconClient` field.

The Markdown reader is enabled by default. It accepts only relative `.md` paths returned
by search and reads the generated source artifact under `indexUrl/sources/`. Set
`markdownReader: false` to disable it, or use `markdownReaderPath` when files are served
elsewhere. Configure `markdown_reader_path` on the MkDocs plugin to choose its static
output directory.

```ts
provideWebsemTools({
  name: "acme-docs",
  description: "Acme documentation",
  displayName: "Acme",
  documentedName: "Acme",
  projectName: "Acme",
  searchToolName: "acme-search",
  indexUrl: "/search/",
  exampleQuestions: [],
  texts: {
    DOC_SEARCH_INSTRUCTION:
      "Search before answering Acme questions. Search is case-insensitive. Read a result's Markdown File path when its snippet is insufficient.",
    DOC_SEARCH_TOOL_DESCRIPTION:
      "Case-insensitive semantic search over Acme docs.",
    DOC_SEARCH_QUERY_DESCRIPTION:
      "Case-insensitive documentation search query.",
    DOC_NO_RESULTS: "No results for {query}.",
    DOC_SUCCESS_HEADER: "Found {count} results for {query}.",
    DOC_RESULT_NOTE: "Only use relevant results.",
  },
});
```

Angular API: https://angular.dev/api/core/provideExperimentalWebMcpTools
