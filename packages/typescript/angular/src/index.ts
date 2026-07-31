import {
  provideExperimentalWebMcpTools,
  type EnvironmentProviders,
  type WebMcpToolDescriptor,
} from "@angular/core";
import { WebsemClient } from "@websem/client";
import type {
  HybridSearchOptions,
  LoadedArtifacts,
  SearchOptions,
  SearchResult,
} from "@websem/types";

export type SearchMode = "semantic" | "keyword" | "hybrid";

export interface WebsemSearchClient {
  semantic(
    query: string,
    options?: SearchOptions,
  ): SearchResult[] | Promise<SearchResult[]>;
  lexical(
    query: string,
    options?: SearchOptions,
  ): SearchResult[] | Promise<SearchResult[]>;
  hybrid(
    query: string,
    options?: HybridSearchOptions,
  ): SearchResult[] | Promise<SearchResult[]>;
}

export interface SearchSourceConfig {
  indexUrl?: string | undefined;
  artifacts?: LoadedArtifacts | undefined;
  client?: WebsemSearchClient | undefined;
}

export interface WebsemMcpTexts {
  DOC_SEARCH_INSTRUCTION: string;
  DOC_SEARCH_TOOL_DESCRIPTION: string;
  DOC_SEARCH_QUERY_DESCRIPTION: string;
  DOC_NO_RESULTS: string;
  DOC_SUCCESS_HEADER: string;
  DOC_RESULT_NOTE: string;
  ICON_SEARCH_INSTRUCTION?: string;
  ICON_SEARCH_TOOL_DESCRIPTION?: string;
  ICON_SEARCH_QUERY_DESCRIPTION?: string;
  ICON_NO_RESULTS?: string;
  ICON_SUCCESS_HEADER?: string;
  ICON_RESULT_NOTE?: string;
}

export interface WebsemAngularConfig extends SearchSourceConfig {
  name: string;
  description: string;
  displayName: string;
  binName?: string;
  compatTokenName?: string;
  documentedName: string;
  projectName: string;
  searchToolName: string;
  iconSearch?: boolean;
  iconSearchToolName?: string;
  iconName?: string;
  iconIndexUrl?: string;
  iconArtifacts?: LoadedArtifacts;
  iconClient?: WebsemSearchClient;
  agentsFile?: boolean;
  instructionFileName?: string;
  observabilityLogging?: boolean;
  exampleQuestions: string[];
  texts: WebsemMcpTexts;
  mode?: SearchMode;
  limit?: number;
  minScore?: number;
  semanticWeight?: number;
  keywordWeight?: number;
  rrfK?: number;
  specificTermHeuristic?: boolean;
}

export interface McpTextResult {
  content: [{ type: "text"; text: string }];
}

interface ToolTextConfig {
  type: "documentation" | "icon";
  name: string;
  label: string;
  instruction: string;
  description: string;
  queryDescription: string;
  noResults: string;
  successHeader: string;
  resultNote: string;
  source: SearchSourceConfig;
}

export type ResultTexts = Pick<
  ToolTextConfig,
  "noResults" | "successHeader" | "resultNote"
>;

const DEFAULT_LIMIT = 5;
const MAX_LIMIT = 20;

const createInputSchema = (queryDescription: string) =>
  ({
    type: "object",
    properties: {
      query: {
        type: "string",
        description: queryDescription,
        minLength: 1,
      },
      limit: {
        type: "integer",
        description: `Maximum number of results to return (default: ${DEFAULT_LIMIT}, max: ${MAX_LIMIT}).`,
        default: DEFAULT_LIMIT,
        minimum: 1,
        maximum: MAX_LIMIT,
      },
    },
    required: ["query"],
    additionalProperties: false,
  }) as const;

type SearchInputSchema = ReturnType<typeof createInputSchema>;

const replacePlaceholders = (
  template: string,
  query: string,
  count: number,
): string =>
  template.replaceAll("{query}", query).replaceAll("{count}", String(count));

const resultHref = (result: SearchResult): string => {
  if (!result.anchor || result.href.includes("#")) {
    return result.href;
  }
  return `${result.href}#${result.anchor}`;
};

export const formatSearchResults = (
  results: SearchResult[],
  query: string,
  texts: ResultTexts,
): string => {
  if (results.length === 0) {
    return replacePlaceholders(texts.noResults, query, 0);
  }

  const sections = results.map((result, index) => {
    const heading = result.heading
      ? `${result.title} > ${result.heading}`
      : result.title;
    return `## Result ${index + 1}: ${heading}\n\n${result.snippet}\n\nSource: ${resultHref(result)}`;
  });
  return [
    replacePlaceholders(texts.successHeader, query, results.length),
    ...sections,
    replacePlaceholders(texts.resultNote, query, results.length),
  ]
    .filter((value) => value.length > 0)
    .join("\n\n");
};

const buildToolDescription = (
  config: WebsemAngularConfig,
  tool: ToolTextConfig,
): string => {
  const values = [
    `${tool.label} ${tool.type === "icon" ? "Icon" : "Documentation"} Search`,
    tool.instruction,
    tool.description,
  ];
  if (config.instructionFileName) {
    values.push(`Project instruction file: ${config.instructionFileName}`);
  }
  if (config.exampleQuestions.length > 0) {
    values.push(`Example questions: ${config.exampleQuestions.join("; ")}`);
  }
  return values.join("\n\n");
};

const validateSource = (source: SearchSourceConfig, label: string): void => {
  const sourceCount = [source.indexUrl, source.artifacts, source.client].filter(
    (value) => value !== undefined,
  ).length;
  if (sourceCount !== 1) {
    throw new Error(
      `Configure exactly one ${label} search source: indexUrl, artifacts, or client.`,
    );
  }
};

const resolveClient = async (
  source: SearchSourceConfig,
): Promise<WebsemSearchClient> => {
  if (source.client) {
    return source.client;
  }
  if (source.artifacts) {
    return new WebsemClient(source.artifacts);
  }

  const configuredUrl = source.indexUrl as string;
  const indexUrl = new URL(configuredUrl, globalThis.location.href);
  if (indexUrl.pathname.endsWith(".json")) {
    const manifest = indexUrl.pathname.split("/").at(-1) as string;
    return WebsemClient.fetch(new URL(".", indexUrl), { manifest });
  }
  if (!indexUrl.pathname.endsWith("/")) {
    indexUrl.pathname += "/";
  }
  return WebsemClient.fetch(indexUrl);
};

const validateToolArgs = (
  value: unknown,
): { query: string; limit?: number } => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Search arguments must be an object.");
  }
  const args = value as Record<string, unknown>;
  if (Object.keys(args).some((key) => key !== "query" && key !== "limit")) {
    throw new TypeError("Search arguments contain an unexpected field.");
  }
  if (typeof args.query !== "string" || args.query.trim().length === 0) {
    throw new TypeError("Search query must be a non-empty string.");
  }
  if (
    args.limit !== undefined &&
    (!Number.isInteger(args.limit) ||
      (args.limit as number) < 1 ||
      (args.limit as number) > MAX_LIMIT)
  ) {
    throw new TypeError(`Search limit must be between 1 and ${MAX_LIMIT}.`);
  }
  const validated: { query: string; limit?: number } = {
    query: args.query.trim(),
  };
  if (args.limit !== undefined) {
    validated.limit = args.limit as number;
  }
  return validated;
};

const performSearch = async (
  client: WebsemSearchClient,
  query: string,
  config: WebsemAngularConfig,
  limit: number,
): Promise<SearchResult[]> => {
  const options: HybridSearchOptions = { limit };
  options.specificTermHeuristic = config.specificTermHeuristic ?? true;
  if (config.minScore !== undefined) {
    options.minScore = config.minScore;
  }
  if (config.mode === "keyword") {
    return client.lexical(query, options);
  }
  if (config.mode === "hybrid") {
    options.semanticWeight = config.semanticWeight ?? 1;
    options.keywordWeight = config.keywordWeight ?? 1;
    options.rrfK = config.rrfK ?? 60;
    return client.hybrid(query, options);
  }
  return client.semantic(query, options);
};

const requireIconText = (value: string | undefined, field: string): string => {
  if (!value) {
    throw new Error(`Icon search is enabled but texts.${field} is missing.`);
  }
  return value;
};

const buildToolConfigs = (config: WebsemAngularConfig): ToolTextConfig[] => {
  const tools: ToolTextConfig[] = [
    {
      type: "documentation",
      name: config.searchToolName,
      label: config.documentedName,
      instruction: config.texts.DOC_SEARCH_INSTRUCTION,
      description: config.texts.DOC_SEARCH_TOOL_DESCRIPTION,
      queryDescription: config.texts.DOC_SEARCH_QUERY_DESCRIPTION,
      noResults: config.texts.DOC_NO_RESULTS,
      successHeader: config.texts.DOC_SUCCESS_HEADER,
      resultNote: config.texts.DOC_RESULT_NOTE,
      source: {
        indexUrl: config.indexUrl,
        artifacts: config.artifacts,
        client: config.client,
      },
    },
  ];
  if (!config.iconSearch) {
    return tools;
  }
  if (!config.iconSearchToolName || !config.iconName) {
    throw new Error(
      "Icon search is enabled but iconSearchToolName or iconName is missing.",
    );
  }
  tools.push({
    type: "icon",
    name: config.iconSearchToolName,
    label: config.iconName,
    instruction: requireIconText(
      config.texts.ICON_SEARCH_INSTRUCTION,
      "ICON_SEARCH_INSTRUCTION",
    ),
    description: requireIconText(
      config.texts.ICON_SEARCH_TOOL_DESCRIPTION,
      "ICON_SEARCH_TOOL_DESCRIPTION",
    ),
    queryDescription: requireIconText(
      config.texts.ICON_SEARCH_QUERY_DESCRIPTION,
      "ICON_SEARCH_QUERY_DESCRIPTION",
    ),
    noResults: requireIconText(config.texts.ICON_NO_RESULTS, "ICON_NO_RESULTS"),
    successHeader: requireIconText(
      config.texts.ICON_SUCCESS_HEADER,
      "ICON_SUCCESS_HEADER",
    ),
    resultNote: requireIconText(
      config.texts.ICON_RESULT_NOTE,
      "ICON_RESULT_NOTE",
    ),
    source: {
      indexUrl: config.iconIndexUrl,
      artifacts: config.iconArtifacts,
      client: config.iconClient,
    },
  });
  return tools;
};

const createWebsemTool = (
  config: WebsemAngularConfig,
  tool: ToolTextConfig,
): WebMcpToolDescriptor<SearchInputSchema> => {
  validateSource(tool.source, tool.type);
  let clientPromise: Promise<WebsemSearchClient> | undefined;
  return {
    name: tool.name,
    description: buildToolDescription(config, tool),
    inputSchema: createInputSchema(tool.queryDescription),
    execute: async (untrustedArgs): Promise<McpTextResult> => {
      const args = validateToolArgs(untrustedArgs);
      clientPromise ??= resolveClient(tool.source);
      const results = await performSearch(
        await clientPromise,
        args.query,
        config,
        args.limit ?? config.limit ?? DEFAULT_LIMIT,
      );
      return {
        content: [
          {
            type: "text",
            text: formatSearchResults(results, args.query, tool),
          },
        ],
      };
    },
  };
};

export const createWebsemTools = (
  config: WebsemAngularConfig,
): WebMcpToolDescriptor<SearchInputSchema>[] =>
  buildToolConfigs(config).map((tool) => createWebsemTool(config, tool));

export const defineWebsemConfig = (
  config: WebsemAngularConfig,
): WebsemAngularConfig => config;

export const provideWebsemTools = (
  config: WebsemAngularConfig,
): EnvironmentProviders =>
  provideExperimentalWebMcpTools(createWebsemTools(config));
