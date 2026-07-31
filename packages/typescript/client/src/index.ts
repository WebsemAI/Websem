import type {
  ChunkRecord,
  HybridSearchOptions,
  LoadedArtifacts,
  Manifest,
  SearchOptions,
  SearchResult,
} from "@websem/types";

export type {
  ChunkRecord,
  HybridSearchOptions,
  LoadedArtifacts,
  Manifest,
  SearchOptions,
  SearchResult,
} from "@websem/types";

const FORMAT_VERSION = "1.0";
const MAX_WORDPIECE_CHARS = 100;
const textDecoder = new TextDecoder("utf-8", { fatal: true });

export interface StaticModel {
  modelId: string;
  dimensions: number;
  vocabulary: readonly string[];
  tokens: Int8Array;
  scales: Float32Array;
}

export interface InMemoryArtifactFiles {
  chunks: string | Uint8Array;
  docs: Int8Array | Uint8Array;
  tokens: Int8Array | Uint8Array;
  scales: Uint8Array | ArrayBuffer | Float32Array;
  vocab: string | Uint8Array;
}

export interface FetchArtifactsOptions {
  fetch?: typeof globalThis.fetch;
  manifest?: string;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const assertInteger = (
  value: unknown,
  name: string,
  minimum: number,
): number => {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new TypeError(`${name} must be an integer >= ${minimum}`);
  }
  return value as number;
};

const assertString = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${name} must be a non-empty string`);
  }
  return value;
};

const assertText = (value: unknown, name: string): string => {
  if (typeof value !== "string") {
    throw new TypeError(`${name} must be a string`);
  }
  return value;
};

const assertArtifactName = (value: unknown, name: string): string => {
  const artifactName = assertString(value, name);
  if (
    artifactName.startsWith("/") ||
    artifactName.includes("\\") ||
    artifactName.split("/").includes("..")
  ) {
    throw new TypeError(`${name} must be a relative artifact path`);
  }
  return artifactName;
};

const assertExactKeys = (
  value: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void => {
  const keys = Object.keys(value);
  if (
    keys.length !== expected.length ||
    keys.some((key) => !expected.includes(key))
  ) {
    throw new TypeError(`${name} contains unexpected or missing fields`);
  }
};

const parseManifest = (value: unknown): Manifest => {
  if (!isObject(value)) {
    throw new TypeError("manifest must be an object");
  }
  assertExactKeys(
    value,
    [
      "websem_version",
      "model_id",
      "dims",
      "vocab_size",
      "n_chunks",
      "chunk_size",
      "chunk_overlap",
      "title_prefix",
      "chunker_version",
      "doc_scale",
      "files",
    ],
    "manifest",
  );
  if (value.websem_version !== FORMAT_VERSION) {
    throw new Error(
      `Unsupported websem artifact version: ${String(value.websem_version)}`,
    );
  }
  if (value.doc_scale !== 127) {
    throw new TypeError("manifest.doc_scale must be 127");
  }
  if (typeof value.title_prefix !== "boolean") {
    throw new TypeError("manifest.title_prefix must be a boolean");
  }
  if (typeof value.chunker_version !== "string") {
    throw new TypeError("manifest.chunker_version must be a string");
  }
  if (!isObject(value.files)) {
    throw new TypeError("manifest.files must be an object");
  }
  assertExactKeys(
    value.files,
    ["chunks", "docs", "tokens", "scales", "vocab"],
    "manifest.files",
  );

  return {
    websem_version: FORMAT_VERSION,
    model_id: assertString(value.model_id, "manifest.model_id"),
    dims: assertInteger(value.dims, "manifest.dims", 1),
    vocab_size: assertInteger(value.vocab_size, "manifest.vocab_size", 1),
    n_chunks: assertInteger(value.n_chunks, "manifest.n_chunks", 0),
    chunk_size: assertInteger(value.chunk_size, "manifest.chunk_size", 1),
    chunk_overlap: assertInteger(
      value.chunk_overlap,
      "manifest.chunk_overlap",
      0,
    ),
    title_prefix: value.title_prefix,
    chunker_version: value.chunker_version,
    doc_scale: 127,
    files: {
      chunks: assertArtifactName(value.files.chunks, "manifest.files.chunks"),
      docs: assertArtifactName(value.files.docs, "manifest.files.docs"),
      tokens: assertArtifactName(value.files.tokens, "manifest.files.tokens"),
      scales: assertArtifactName(value.files.scales, "manifest.files.scales"),
      vocab: assertArtifactName(value.files.vocab, "manifest.files.vocab"),
    },
  };
};

const bytesToText = (value: string | Uint8Array): string =>
  typeof value === "string" ? value : textDecoder.decode(value);

const parseJson = (value: string | Uint8Array, name: string): unknown => {
  const parsed: unknown = JSON.parse(bytesToText(value));
  if (parsed === undefined) {
    throw new TypeError(`${name} is invalid JSON`);
  }
  return parsed;
};

const parseChunks = (value: string | Uint8Array): ChunkRecord[] => {
  const parsed = parseJson(value, "chunks");
  if (!Array.isArray(parsed)) {
    throw new TypeError("chunks must be a JSON array");
  }
  return parsed.map((item, index) => {
    if (!isObject(item)) {
      throw new TypeError(`chunks[${index}] must be an object`);
    }
    const allowed = [
      "document",
      "title",
      "href",
      "snippet",
      "heading",
      "anchor",
    ];
    if (Object.keys(item).some((key) => !allowed.includes(key))) {
      throw new TypeError(`chunks[${index}] contains an unexpected field`);
    }
    const record: ChunkRecord = {
      document: assertString(item.document, `chunks[${index}].document`),
      title: assertString(item.title, `chunks[${index}].title`),
      href: assertText(item.href, `chunks[${index}].href`),
      snippet: assertString(item.snippet, `chunks[${index}].snippet`),
    };
    if (item.heading !== undefined) {
      record.heading = assertString(item.heading, `chunks[${index}].heading`);
    }
    if (item.anchor !== undefined) {
      record.anchor = assertString(item.anchor, `chunks[${index}].anchor`);
    }
    return record;
  });
};

const parseVocabulary = (value: string | Uint8Array): string[] => {
  const parsed = parseJson(value, "vocab");
  if (
    !Array.isArray(parsed) ||
    parsed.some((token) => typeof token !== "string")
  ) {
    throw new TypeError("vocab must be a JSON string array");
  }
  const vocabulary = parsed as string[];
  if (new Set(vocabulary).size !== vocabulary.length) {
    throw new TypeError("vocab tokens must be unique");
  }
  return vocabulary;
};

const toInt8 = (value: Int8Array | Uint8Array): Int8Array =>
  new Int8Array(
    value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength),
  );

const parseScales = (
  value: Uint8Array | ArrayBuffer | Float32Array,
): Float32Array => {
  if (value instanceof Float32Array) {
    const scales = new Float32Array(value);
    if (scales.some((scale) => !Number.isFinite(scale) || scale <= 0)) {
      throw new TypeError("token scales must be finite and positive");
    }
    return scales;
  }
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  if (bytes.byteLength % 4 !== 0) {
    throw new TypeError("token scales byte length must be divisible by 4");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scales = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < scales.length; index += 1) {
    scales[index] = view.getFloat32(index * 4, true);
  }
  if (scales.some((scale) => !Number.isFinite(scale) || scale <= 0)) {
    throw new TypeError("token scales must be finite and positive");
  }
  return scales;
};

export const loadArtifacts = (
  manifestValue: unknown,
  files: InMemoryArtifactFiles,
): LoadedArtifacts => {
  const manifest = parseManifest(manifestValue);
  if (manifest.chunk_overlap >= manifest.chunk_size) {
    throw new TypeError(
      "manifest.chunk_overlap must be smaller than manifest.chunk_size",
    );
  }
  const chunks = parseChunks(files.chunks);
  const vocabulary = parseVocabulary(files.vocab);
  const docs = toInt8(files.docs);
  const tokens = toInt8(files.tokens);
  const scales = parseScales(files.scales);

  if (chunks.length !== manifest.n_chunks) {
    throw new TypeError("chunk count does not match manifest.n_chunks");
  }
  if (vocabulary.length !== manifest.vocab_size) {
    throw new TypeError("vocab length does not match manifest.vocab_size");
  }
  if (docs.length !== manifest.n_chunks * manifest.dims) {
    throw new TypeError("docs byte length does not match n_chunks * dims");
  }
  if (tokens.length !== manifest.vocab_size * manifest.dims) {
    throw new TypeError("tokens byte length does not match vocab_size * dims");
  }
  if (scales.length !== manifest.vocab_size) {
    throw new TypeError("scale count does not match manifest.vocab_size");
  }
  return { manifest, chunks, docs, tokens, scales, vocabulary };
};

const fetchBytes = async (
  url: URL,
  fetcher: typeof globalThis.fetch,
  init?: RequestInit,
): Promise<Uint8Array> => {
  const response = await fetcher(url, init);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${url.href}: ${response.status} ${response.statusText}`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
};

export const fetchArtifacts = async (
  baseUrl: string | URL,
  options: FetchArtifactsOptions = {},
): Promise<LoadedArtifacts> => {
  const fetcher = options.fetch ?? globalThis.fetch;
  const manifestUrl = new URL(options.manifest ?? "manifest.json", baseUrl);
  const manifestBytes = await fetchBytes(manifestUrl, fetcher, {
    cache: "no-cache",
  });
  const manifest = parseManifest(parseJson(manifestBytes, "manifest"));
  const [chunks, docs, tokens, scales, vocab] = await Promise.all([
    fetchBytes(new URL(manifest.files.chunks, manifestUrl), fetcher),
    fetchBytes(new URL(manifest.files.docs, manifestUrl), fetcher),
    fetchBytes(new URL(manifest.files.tokens, manifestUrl), fetcher),
    fetchBytes(new URL(manifest.files.scales, manifestUrl), fetcher),
    fetchBytes(new URL(manifest.files.vocab, manifestUrl), fetcher),
  ]);
  return loadArtifacts(manifest, { chunks, docs, tokens, scales, vocab });
};

const isWhitespace = (character: string): boolean =>
  character === " " ||
  character === "\t" ||
  character === "\n" ||
  character === "\r" ||
  /\p{Zs}/u.test(character);

const isControl = (character: string): boolean => {
  if (character === "\t" || character === "\n" || character === "\r") {
    return false;
  }
  return /\p{C}/u.test(character);
};

const isChinese = (codePoint: number): boolean =>
  (codePoint >= 0x4e00 && codePoint <= 0x9fff) ||
  (codePoint >= 0x3400 && codePoint <= 0x4dbf) ||
  (codePoint >= 0x20000 && codePoint <= 0x2a6df) ||
  (codePoint >= 0x2a700 && codePoint <= 0x2b73f) ||
  (codePoint >= 0x2b740 && codePoint <= 0x2b81f) ||
  (codePoint >= 0x2b820 && codePoint <= 0x2ceaf) ||
  (codePoint >= 0xf900 && codePoint <= 0xfaff) ||
  (codePoint >= 0x2f800 && codePoint <= 0x2fa1f);

const cleanAndSpaceChinese = (text: string): string => {
  let result = "";
  for (const character of text) {
    const codePoint = character.codePointAt(0) as number;
    if (codePoint === 0 || codePoint === 0xfffd || isControl(character)) {
      continue;
    }
    if (isWhitespace(character)) {
      result += " ";
    } else if (isChinese(codePoint)) {
      result += ` ${character} `;
    } else {
      result += character;
    }
  }
  return result;
};

const splitPunctuationAndSymbols = (token: string): string[] => {
  const output: string[] = [];
  let current = "";
  for (const character of token) {
    const codePoint = character.codePointAt(0) as number;
    const asciiPunctuation =
      (codePoint >= 33 && codePoint <= 47) ||
      (codePoint >= 58 && codePoint <= 64) ||
      (codePoint >= 91 && codePoint <= 96) ||
      (codePoint >= 123 && codePoint <= 126);
    if (asciiPunctuation || /\p{P}|\p{S}/u.test(character)) {
      if (current.length > 0) {
        output.push(current);
        current = "";
      }
      output.push(character);
    } else {
      current += character;
    }
  }
  if (current.length > 0) {
    output.push(current);
  }
  return output;
};

export const basicTokenize = (text: string): string[] => {
  const cleaned = cleanAndSpaceChinese(text);
  const output: string[] = [];
  for (const token of cleaned.trim().split(/\s+/u)) {
    if (token.length === 0) {
      continue;
    }
    const normalized = token
      .toLowerCase()
      .normalize("NFD")
      .replace(/\p{Mn}/gu, "");
    output.push(...splitPunctuationAndSymbols(normalized));
  }
  return output;
};

const wordPieceIds = (
  token: string,
  vocabulary: ReadonlyMap<string, number>,
): number[] => {
  const characters = Array.from(token);
  if (characters.length > MAX_WORDPIECE_CHARS) {
    return [];
  }
  const pieces: number[] = [];
  let start = 0;
  while (start < characters.length) {
    let end = characters.length;
    let matchedId: number | undefined;
    while (start < end) {
      const candidate = `${start === 0 ? "" : "##"}${characters.slice(start, end).join("")}`;
      matchedId = vocabulary.get(candidate);
      if (matchedId !== undefined) {
        break;
      }
      end -= 1;
    }
    if (matchedId === undefined) {
      return [];
    }
    pieces.push(matchedId);
    start = end;
  }
  return pieces;
};

export const tokenize = (
  text: string,
  vocabulary: readonly string[],
): number[] => {
  const tokenIds = new Map(vocabulary.map((token, index) => [token, index]));
  return basicTokenize(text).flatMap((token) => wordPieceIds(token, tokenIds));
};

export const embedText = (text: string, model: StaticModel): Float32Array => {
  if (model.modelId.length === 0 || model.vocabulary.length === 0) {
    throw new TypeError("model id and vocabulary must be non-empty");
  }
  if (model.dimensions < 1 || !Number.isInteger(model.dimensions)) {
    throw new TypeError("model.dimensions must be a positive integer");
  }
  if (model.tokens.length !== model.vocabulary.length * model.dimensions) {
    throw new TypeError(
      "model token length does not match vocabulary * dimensions",
    );
  }
  if (model.scales.length !== model.vocabulary.length) {
    throw new TypeError("model scale count does not match vocabulary");
  }
  if (model.scales.some((scale) => !Number.isFinite(scale) || scale <= 0)) {
    throw new TypeError("model scales must be finite and positive");
  }
  const ids = tokenize(text, model.vocabulary);
  const embedding = new Float32Array(model.dimensions);
  if (ids.length === 0) {
    return embedding;
  }
  for (const id of ids) {
    const scale = model.scales[id] as number;
    const offset = id * model.dimensions;
    for (let dimension = 0; dimension < model.dimensions; dimension += 1) {
      embedding[dimension] =
        (embedding[dimension] as number) +
        (model.tokens[offset + dimension] as number) * scale;
    }
  }
  for (let dimension = 0; dimension < embedding.length; dimension += 1) {
    embedding[dimension] = (embedding[dimension] as number) / ids.length;
  }
  let normSquared = 0;
  for (const value of embedding) {
    normSquared += value * value;
  }
  if (normSquared === 0) {
    return embedding;
  }
  const norm = Math.sqrt(normSquared);
  for (let dimension = 0; dimension < embedding.length; dimension += 1) {
    embedding[dimension] = (embedding[dimension] as number) / norm;
  }
  return embedding;
};

export const modelFromArtifacts = (
  artifacts: LoadedArtifacts,
): StaticModel => ({
  modelId: artifacts.manifest.model_id,
  dimensions: artifacts.manifest.dims,
  vocabulary: artifacts.vocabulary,
  tokens: artifacts.tokens,
  scales: artifacts.scales,
});

interface RankedChunk {
  index: number;
  score: number;
}

const bestDocuments = (
  artifacts: LoadedArtifacts,
  rankedChunks: readonly RankedChunk[],
  options: SearchOptions,
): SearchResult[] => {
  const limit = options.limit ?? 10;
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError("search limit must be a non-negative integer");
  }
  if (options.minScore !== undefined && !Number.isFinite(options.minScore)) {
    throw new RangeError("minScore must be finite");
  }
  if (
    options.specificTermHeuristic !== undefined &&
    typeof options.specificTermHeuristic !== "boolean"
  ) {
    throw new TypeError("specificTermHeuristic must be a boolean");
  }
  const minimum = options.minScore ?? 0;
  const best = new Map<string, RankedChunk>();
  for (const ranked of rankedChunks) {
    if (ranked.score < minimum) {
      continue;
    }
    const chunk = artifacts.chunks[ranked.index] as ChunkRecord;
    const previous = best.get(chunk.document);
    if (
      previous === undefined ||
      ranked.score > previous.score ||
      (ranked.score === previous.score && ranked.index < previous.index)
    ) {
      best.set(chunk.document, ranked);
    }
  }
  return [...best.values()]
    .sort((left, right) => {
      const scoreOrder = right.score - left.score;
      if (scoreOrder !== 0) {
        return scoreOrder;
      }
      return (
        artifacts.chunks[left.index] as ChunkRecord
      ).document.localeCompare(
        (artifacts.chunks[right.index] as ChunkRecord).document,
      );
    })
    .slice(0, limit)
    .map(({ index, score }) => ({
      ...(artifacts.chunks[index] as ChunkRecord),
      score,
    }));
};

export const semanticSearch = (
  query: string,
  artifacts: LoadedArtifacts,
  options: SearchOptions = {},
): SearchResult[] => {
  const limit = options.limit ?? 10;
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError("search limit must be a non-negative integer");
  }
  const specific =
    options.specificTermHeuristic === false
      ? []
      : specificTermSearch(query, artifacts, {
          limit: artifacts.chunks.length,
        });
  const queryEmbedding = embedText(query, modelFromArtifacts(artifacts));
  let queryNormSquared = 0;
  for (const value of queryEmbedding) {
    queryNormSquared += value * value;
  }
  const ranked: RankedChunk[] = [];
  if (queryNormSquared > 0) {
    for (let chunk = 0; chunk < artifacts.chunks.length; chunk += 1) {
      const offset = chunk * artifacts.manifest.dims;
      let dot = 0;
      let docNormSquared = 0;
      for (
        let dimension = 0;
        dimension < artifacts.manifest.dims;
        dimension += 1
      ) {
        const value = artifacts.docs[offset + dimension] as number;
        dot += (queryEmbedding[dimension] as number) * value;
        docNormSquared += value * value;
      }
      const score = docNormSquared === 0 ? 0 : dot / Math.sqrt(docNormSquared);
      ranked.push({ index: chunk, score });
    }
  }
  const semantic = bestDocuments(artifacts, ranked, {
    ...options,
    limit: artifacts.chunks.length,
  });
  const combined = new Map<string, SearchResult>();
  for (const result of [...specific, ...semantic]) {
    if (!combined.has(result.document)) {
      combined.set(result.document, result);
    }
  }
  return [...combined.values()].slice(0, limit);
};

const lexicalTerms = (text: string): string[] =>
  basicTokenize(text).filter((term) => /[\p{L}\p{N}]/u.test(term));

const SPECIFIC_TERM_STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "has",
  "have",
  "he",
  "her",
  "hers",
  "him",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "me",
  "my",
  "no",
  "not",
  "of",
  "on",
  "or",
  "our",
  "ours",
  "she",
  "should",
  "so",
  "than",
  "that",
  "the",
  "their",
  "theirs",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "to",
  "too",
  "us",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "would",
  "you",
  "your",
  "yours",
]);

const specificTerms = (
  query: string,
  vocabulary: readonly string[],
): string[] => {
  const embeddedTerms = new Set(
    vocabulary
      .filter((token) => !token.startsWith("##"))
      .map((token) => token.toLowerCase()),
  );
  return [...new Set(lexicalTerms(query))].filter(
    (term) =>
      Array.from(term).length > 1 &&
      !SPECIFIC_TERM_STOP_WORDS.has(term.toLowerCase()) &&
      !embeddedTerms.has(term.toLowerCase()),
  );
};

export const specificTermSearch = (
  query: string,
  artifacts: LoadedArtifacts,
  options: SearchOptions = {},
): SearchResult[] => {
  const terms = specificTerms(query, artifacts.vocabulary);
  if (terms.length === 0) {
    return [];
  }
  const ranked = artifacts.chunks.map((chunk, index): RankedChunk => {
    const title = new Set(lexicalTerms(chunk.title));
    const heading = new Set(lexicalTerms(chunk.heading ?? ""));
    const snippet = new Set(lexicalTerms(chunk.snippet));
    const score = terms.reduce(
      (total, term) =>
        total +
        (title.has(term) ? 4 : 0) +
        (heading.has(term) ? 2 : 0) +
        (snippet.has(term) ? 1 : 0),
      0,
    );
    return { index, score };
  });
  return bestDocuments(artifacts, ranked, {
    ...options,
    minScore: Number.MIN_VALUE,
  });
};

export const lexicalSearch = (
  query: string,
  artifacts: LoadedArtifacts,
  options: SearchOptions = {},
): SearchResult[] => {
  const queryTerms = [...new Set(lexicalTerms(query))];
  if (queryTerms.length === 0) {
    return [];
  }
  const chunkTerms = artifacts.chunks.map((chunk) =>
    lexicalTerms(`${chunk.title} ${chunk.heading ?? ""} ${chunk.snippet}`),
  );
  const documentFrequency = new Map<string, number>();
  for (const terms of chunkTerms) {
    const unique = new Set(terms);
    for (const term of queryTerms) {
      if (unique.has(term)) {
        documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
      }
    }
  }
  const averageLength = Math.max(
    1,
    chunkTerms.reduce((sum, terms) => sum + terms.length, 0) /
      Math.max(1, chunkTerms.length),
  );
  const ranked = chunkTerms.map((terms, index): RankedChunk => {
    const frequencies = new Map<string, number>();
    for (const term of terms) {
      frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
    }
    let score = 0;
    for (const term of queryTerms) {
      const frequency = frequencies.get(term) ?? 0;
      const frequencyWeight =
        (frequency * 2.2) /
        (frequency + 1.2 * (0.25 + 0.75 * (terms.length / averageLength)));
      const inverseFrequency = Math.log(
        1 +
          (chunkTerms.length - (documentFrequency.get(term) ?? 0) + 0.5) /
            ((documentFrequency.get(term) ?? 0) + 0.5),
      );
      score += inverseFrequency * frequencyWeight;
    }
    return { index, score };
  });
  return bestDocuments(artifacts, ranked, {
    ...options,
    minScore: options.minScore ?? Number.MIN_VALUE,
  });
};

export const hybridSearch = (
  query: string,
  artifacts: LoadedArtifacts,
  options: HybridSearchOptions = {},
): SearchResult[] => {
  const limit = options.limit ?? 10;
  const semanticWeight = options.semanticWeight ?? 1;
  const keywordWeight = options.keywordWeight ?? 1;
  const rrfK = options.rrfK ?? 60;
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError("hybrid search limit must be a non-negative integer");
  }
  if (
    ![rrfK, semanticWeight, keywordWeight].every(
      (value) => Number.isFinite(value) && value >= 0,
    )
  ) {
    throw new RangeError(
      "hybrid search weights and rrfK must be finite and non-negative",
    );
  }
  const candidateLimit = artifacts.chunks.length;
  const semanticOptions: SearchOptions = { limit: candidateLimit };
  semanticOptions.specificTermHeuristic = false;
  if (options.minScore !== undefined) {
    semanticOptions.minScore = options.minScore;
  }
  const semantic = semanticSearch(query, artifacts, semanticOptions);
  const lexical = lexicalSearch(query, artifacts, { limit: candidateLimit });
  const specific =
    options.specificTermHeuristic === false
      ? []
      : specificTermSearch(query, artifacts, { limit: candidateLimit });
  const scores = new Map<
    string,
    { result: SearchResult; score: number; order: number }
  >();
  const add = (results: readonly SearchResult[], weight: number): void => {
    if (weight === 0) {
      return;
    }
    results.forEach((result, index) => {
      const previous = scores.get(result.document);
      const contribution = weight / (rrfK + index + 1);
      if (previous === undefined) {
        scores.set(result.document, {
          result,
          score: contribution,
          order: scores.size,
        });
      } else {
        previous.score += contribution;
      }
    });
  };
  add(specific, 1);
  add(semantic, semanticWeight);
  add(lexical, keywordWeight);
  return [...scores.values()]
    .sort(
      (left, right) =>
        right.score - left.score ||
        left.result.document.localeCompare(right.result.document),
    )
    .slice(0, limit)
    .map(({ result, score }) => ({ ...result, score }));
};

export class WebsemClient {
  public constructor(public readonly artifacts: LoadedArtifacts) {}

  public static async fetch(
    baseUrl: string | URL,
    options: FetchArtifactsOptions = {},
  ): Promise<WebsemClient> {
    return new WebsemClient(await fetchArtifacts(baseUrl, options));
  }

  public embed(query: string): Float32Array {
    return embedText(query, modelFromArtifacts(this.artifacts));
  }

  public semantic(query: string, options: SearchOptions = {}): SearchResult[] {
    return semanticSearch(query, this.artifacts, options);
  }

  public lexical(query: string, options: SearchOptions = {}): SearchResult[] {
    return lexicalSearch(query, this.artifacts, options);
  }

  public hybrid(
    query: string,
    options: HybridSearchOptions = {},
  ): SearchResult[] {
    return hybridSearch(query, this.artifacts, options);
  }
}
