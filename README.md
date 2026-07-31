# websem

Static semantic search for browsers, with no search server and no runtime model inference.

The monorepo contains separate Python and TypeScript types, builders, and clients, plus
MkDocs Material plugins and an Angular WebMCP adapter. Artifacts use one versioned binary
format, so either client can read output from either builder.

## Build

```bash
uv sync --all-packages
npm install

websem-build docs.json --model minishlab/potion-base-8M --dimensions 128 --out site/search
websem-export-model --model minishlab/potion-base-8M --dimensions 128 --out model
npx websem-build --docs docs.json --model model --out site/search
```

Documents are JSON objects with `id`, `title`, `href`, `text`, and optional `sections`.
The default chunk size is 600 characters with 120 characters of sentence overlap.

## MkDocs

```yaml
plugins:
  - search
  - websem-build:
      dims: 128
      include:
        - README.md
        - components/
        - patterns/**/*.md
      exclude:
        - api/**
  - websem-search:
      mode: hybrid
      factor: 1.0
      limit: 10
```

`include` and `exclude` match paths relative to `docs_dir`. Exact files and bare folder
paths are supported alongside `*`, `**`, `?`, and character-class patterns. Exclusions
take precedence, and an empty `include` list indexes every page.

`factor` weights semantic ranks against the regular keyword ranks. Set `mode: semantic`
to disable keyword fusion.

Terms without an exact model vocabulary row are matched case-insensitively against the
documents and surfaced by default. Set `specificTermHeuristic: false` in TypeScript or
`specific_term_heuristic=False` in Python to disable this behavior.

## Angular WebMCP

```ts
provideWebsemTools({
  name: "acme-docs",
  description: "Acme documentation search",
  displayName: "Acme",
  documentedName: "Acme",
  projectName: "Acme",
  searchToolName: "acme-search",
  indexUrl: "/search/",
  exampleQuestions: ["How do I configure Acme?"],
  texts: {
    DOC_SEARCH_INSTRUCTION: "Use this tool before answering Acme questions.",
    DOC_SEARCH_TOOL_DESCRIPTION: "Semantic search over Acme documentation.",
    DOC_SEARCH_QUERY_DESCRIPTION: "Documentation search query.",
    DOC_NO_RESULTS: "No results for {query}.",
    DOC_SUCCESS_HEADER: "Found {count} results for {query}.",
    DOC_RESULT_NOTE: "Only use relevant results.",
  },
});
```

WebMCP has no global instruction API. `DOC_SEARCH_INSTRUCTION` and optional icon
instructions are therefore injected into each tool description.

Based on the approach in [Client-side semantic search for your static site](https://bart.degoe.de/semantic-search-in-your-browser/).
