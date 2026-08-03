import type { SearchResult } from "@websem/types";
import { describe, expect, it, vi } from "vitest";

import {
  createWebsemTools,
  formatSearchResults,
  provideWebsemTools,
  type WebsemAngularConfig,
  type WebsemSearchClient,
} from "../src/index.js";

const emptyClient = (): WebsemSearchClient => ({
  semantic: () => [],
  lexical: () => [],
  hybrid: () => [],
});

const config = (
  overrides: Partial<WebsemAngularConfig> = {},
): WebsemAngularConfig => ({
  name: "@acme/docs-mcp",
  description: "Acme MCP",
  displayName: "Acme",
  documentedName: "Acme",
  projectName: "Acme",
  searchToolName: "acme-search",
  client: emptyClient(),
  instructionFileName: "Acme.instructions.md",
  exampleQuestions: ["How do I install Acme?"],
  texts: {
    DOC_SEARCH_INSTRUCTION: "Always search before answering Acme questions.",
    DOC_SEARCH_TOOL_DESCRIPTION: "Semantic search over Acme documentation.",
    DOC_SEARCH_QUERY_DESCRIPTION: "Acme documentation query.",
    DOC_NO_RESULTS: "No docs for {query}.",
    DOC_SUCCESS_HEADER: "Found {count} docs for {query}.",
    DOC_RESULT_NOTE: "Only use relevant results.",
  },
  ...overrides,
});

const result: SearchResult = {
  document: "guide",
  title: "Guide",
  heading: "Setup",
  anchor: "setup",
  href: "/guide/",
  snippet: "Install the package.",
  score: 0.91,
};

describe("result formatting", () => {
  it("uses configured placeholders and anchors", () => {
    expect(
      formatSearchResults([result], "install", {
        successHeader: "{count} hits for {query}",
        resultNote: "Only use {count} relevant result.",
        noResults: "Nothing for {query}.",
      }),
    ).toContain("Source: /guide/#setup");
  });

  it("uses the configured no-results text", () => {
    expect(
      formatSearchResults([], "missing", {
        successHeader: "unused",
        resultNote: "unused",
        noResults: "Nothing for {query} ({count}).",
      }),
    ).toBe("Nothing for missing (0).");
  });
});

describe("sdl-mcp style configuration", () => {
  it("injects configurable instructions and descriptions", () => {
    const [tool] = createWebsemTools(config());
    expect(tool?.name).toBe("acme-search");
    expect(tool?.description).toContain(
      "Always search before answering Acme questions.",
    );
    expect(tool?.description).toContain(
      "Semantic search over Acme documentation.",
    );
    expect(tool?.description).toContain("Acme.instructions.md");
    expect(tool?.inputSchema.properties.query.description).toBe(
      "Acme documentation query.",
    );
  });

  it("creates optional icon search from ICON fields", () => {
    const tools = createWebsemTools(
      config({
        iconSearch: true,
        iconSearchToolName: "acme-icon-search",
        iconName: "@acme/icons",
        iconClient: emptyClient(),
        texts: {
          ...config().texts,
          ICON_SEARCH_INSTRUCTION: "Search before choosing an icon.",
          ICON_SEARCH_TOOL_DESCRIPTION: "Search Acme icons.",
          ICON_SEARCH_QUERY_DESCRIPTION: "Icon query.",
          ICON_NO_RESULTS: "No icons for {query}.",
          ICON_SUCCESS_HEADER: "Found {count} icons for {query}.",
          ICON_RESULT_NOTE: "Check icon usage guidance.",
        },
      }),
    );
    expect(tools.map((tool) => tool.name)).toEqual([
      "acme-search",
      "acme-icon-search",
    ]);
    expect(tools[1]?.description).toContain("Search before choosing an icon.");
  });

  it("creates a Markdown-only reader by default", async () => {
    const tools = createWebsemTools(config());
    const reader = tools[1];
    expect(reader?.name).toBe("acme-search-read");
    expect(reader?.description).toContain("Markdown (.md)");
    await expect(
      reader?.execute({ path: "../secrets.txt" } as never, {} as never),
    ).rejects.toThrow("relative Markdown");
  });

  it("allows disabling the Markdown reader", () => {
    expect(createWebsemTools(config({ markdownReader: false }))).toHaveLength(1);
  });

  it("validates input and forwards hybrid options", async () => {
    const hybrid = vi.fn(() => [result]);
    const [tool] = createWebsemTools(
      config({
        client: { ...emptyClient(), hybrid },
        mode: "hybrid",
        minScore: 0.3,
        semanticWeight: 0.7,
        keywordWeight: 0.3,
        rrfK: 40,
        specificTermHeuristic: true,
      }),
    );
    await tool?.execute({ query: " setup ", limit: 2 }, {} as never);
    expect(hybrid).toHaveBeenCalledWith("setup", {
      limit: 2,
      minScore: 0.3,
      semanticWeight: 0.7,
      keywordWeight: 0.3,
      rrfK: 40,
      specificTermHeuristic: true,
    });
    await expect(
      tool?.execute({ query: "setup", limit: 21 } as never, {} as never),
    ).rejects.toThrow("between 1 and 20");
  });

  it("creates Angular environment providers", () => {
    expect(provideWebsemTools(config())).toBeDefined();
  });
});
