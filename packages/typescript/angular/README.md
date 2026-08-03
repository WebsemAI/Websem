# @websem/angular

Angular v22+ WebMCP tools backed by `@websem/client`.

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
    DOC_SEARCH_INSTRUCTION: "Search before answering Acme questions.",
    DOC_SEARCH_TOOL_DESCRIPTION: "Semantic search over Acme docs.",
    DOC_SEARCH_QUERY_DESCRIPTION: "Documentation search query.",
    DOC_NO_RESULTS: "No results for {query}.",
    DOC_SUCCESS_HEADER: "Found {count} results for {query}.",
    DOC_RESULT_NOTE: "Only use relevant results.",
  },
});
```

Angular API: https://angular.dev/api/core/provideExperimentalWebMcpTools
