const moduleUrl = new URL(import.meta.url);
const configUrl = new URL("websem-search.json", moduleUrl);
const initializedInputs = new WeakSet();
const maxWordPieceChars = 100;
let configPromise;
let artifactsPromise;

const fetchJson = async (url, init) => {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`websem failed to load ${url}: ${response.status}`);
  }
  return response.json();
};

const fetchBuffer = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`websem failed to load ${url}: ${response.status}`);
  }
  return response.arrayBuffer();
};

const loadConfig = () => {
  configPromise ??= fetchJson(configUrl).then((config) => ({
    ...config,
    manifestUrl: new URL(config.manifestUrl, moduleUrl),
    siteRootUrl: new URL(config.siteRootUrl, moduleUrl),
  }));
  return configPromise;
};

const loadArtifacts = async () => {
  if (artifactsPromise) {
    return artifactsPromise;
  }

  artifactsPromise = loadConfig().then(async (config) => {
    const manifest = await fetchJson(config.manifestUrl, { cache: "no-cache" });
    const artifactUrl = (name) => new URL(name, config.manifestUrl);
    const [chunks, docs, tokens, scaleBuffer, vocabulary] = await Promise.all([
      fetchJson(artifactUrl(manifest.files.chunks)),
      fetchBuffer(artifactUrl(manifest.files.docs)),
      fetchBuffer(artifactUrl(manifest.files.tokens)),
      fetchBuffer(artifactUrl(manifest.files.scales)),
      fetchJson(artifactUrl(manifest.files.vocab)),
    ]);
    const scaleBytes = new Uint8Array(scaleBuffer);
    const scaleView = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
    const scales = new Float32Array(scaleBytes.byteLength / 4);
    for (let index = 0; index < scales.length; index += 1) {
      scales[index] = scaleView.getFloat32(index * 4, true);
    }
    if (manifest.websem_version !== "1.0") {
      throw new Error(`websem unsupported artifact version: ${manifest.websem_version}`);
    }
    if (chunks.length !== manifest.n_chunks
      || docs.byteLength !== manifest.n_chunks * manifest.dims
      || tokens.byteLength !== manifest.vocab_size * manifest.dims
      || scaleBytes.byteLength !== manifest.vocab_size * 4
      || vocabulary.length !== manifest.vocab_size) {
      throw new Error("websem artifact lengths do not match the manifest");
    }
    return {
      config,
      manifest,
      chunks,
      docs: new Int8Array(docs),
      tokens: new Int8Array(tokens),
      scales,
      vocabulary,
      vocabularyIndex: new Map(vocabulary.map((token, index) => [token, index])),
    };
  });
  return artifactsPromise;
};

const isWhitespace = (character) => /\s|\p{Zs}/u.test(character);

const isControl = (character) => {
  if (character === "\t" || character === "\n" || character === "\r") {
    return false;
  }
  return /\p{C}/u.test(character);
};

const isChinese = (codePoint) =>
  (codePoint >= 0x4e00 && codePoint <= 0x9fff)
  || (codePoint >= 0x3400 && codePoint <= 0x4dbf)
  || (codePoint >= 0x20000 && codePoint <= 0x2a6df)
  || (codePoint >= 0x2a700 && codePoint <= 0x2b73f)
  || (codePoint >= 0x2b740 && codePoint <= 0x2b81f)
  || (codePoint >= 0x2b820 && codePoint <= 0x2ceaf)
  || (codePoint >= 0xf900 && codePoint <= 0xfaff)
  || (codePoint >= 0x2f800 && codePoint <= 0x2fa1f);

const splitPunctuation = (token) => {
  const output = [];
  let current = "";
  for (const character of token) {
    if (/\p{P}|\p{S}/u.test(character)) {
      if (current) {
        output.push(current);
        current = "";
      }
      output.push(character);
    } else {
      current += character;
    }
  }
  if (current) {
    output.push(current);
  }
  return output;
};

const basicTokenize = (value) => {
  let cleaned = "";
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint === 0 || codePoint === 0xfffd || isControl(character)) {
      continue;
    }
    cleaned += isWhitespace(character)
      ? " "
      : isChinese(codePoint) ? ` ${character} ` : character;
  }
  return cleaned.trim().split(/\s+/u).flatMap((token) => splitPunctuation(
    token.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, ""),
  ));
};

const wordPieceIds = (token, vocabularyIndex) => {
  const characters = Array.from(token);
  if (characters.length > maxWordPieceChars) {
    return [];
  }
  const pieces = [];
  let start = 0;
  while (start < characters.length) {
    let end = characters.length;
    let matched;
    while (start < end) {
      const prefix = start === 0 ? "" : "##";
      matched = vocabularyIndex.get(`${prefix}${characters.slice(start, end).join("")}`);
      if (matched !== undefined) {
        break;
      }
      end -= 1;
    }
    if (matched === undefined) {
      return [];
    }
    pieces.push(matched);
    start = end;
  }
  return pieces;
};

const tokenize = (value, artifacts) => basicTokenize(value)
  .flatMap((token) => wordPieceIds(token, artifacts.vocabularyIndex));

const normalize = (vector) => {
  let norm = 0;
  for (const value of vector) {
    norm += value * value;
  }
  norm = Math.sqrt(norm);
  if (norm === 0) {
    return vector;
  }
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] /= norm;
  }
  return vector;
};

const embedQuery = (query, artifacts) => {
  const { dims } = artifacts.manifest;
  const vector = new Float32Array(dims);
  const tokenIds = tokenize(query, artifacts);
  if (tokenIds.length === 0) {
    return vector;
  }
  for (const tokenIndex of tokenIds) {
    const scale = artifacts.scales[tokenIndex];
    const offset = tokenIndex * dims;
    for (let dimension = 0; dimension < dims; dimension += 1) {
      vector[dimension] += artifacts.tokens[offset + dimension] * scale;
    }
  }
  for (let dimension = 0; dimension < dims; dimension += 1) {
    vector[dimension] /= tokenIds.length;
  }
  return normalize(vector);
};

const bestDocuments = (ranked, artifacts, count, minScore = Number.NEGATIVE_INFINITY) => {
  const best = new Map();
  for (const rankedChunk of ranked) {
    if (rankedChunk.score < minScore) {
      continue;
    }
    const chunk = artifacts.chunks[rankedChunk.index];
    const previous = best.get(chunk.document);
    if (!previous || rankedChunk.score > previous.score) {
      best.set(chunk.document, rankedChunk);
    }
  }
  return [...best.values()]
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, count)
    .map(({ index, score }) => ({ ...artifacts.chunks[index], score }));
};

const semanticResults = (query, artifacts, count, minScore) => {
  const vector = embedQuery(query, artifacts);
  if (!vector.some((value) => value !== 0)) {
    return [];
  }
  const { dims } = artifacts.manifest;
  const ranked = artifacts.chunks.map((_, chunkIndex) => {
      let dot = 0;
      let documentNorm = 0;
      const offset = chunkIndex * dims;
      for (let dimension = 0; dimension < dims; dimension += 1) {
        const value = artifacts.docs[offset + dimension];
        dot += vector[dimension] * value;
        documentNorm += value * value;
      }
      return { index: chunkIndex, score: documentNorm === 0 ? 0 : dot / Math.sqrt(documentNorm) };
    });
  return bestDocuments(ranked, artifacts, count, minScore);
};

const lexicalTerms = (value) => basicTokenize(value)
  .filter((term) => /[\p{L}\p{N}]/u.test(term));

const specificTermStopWords = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
  "did", "do", "does", "for", "from", "has", "have", "he", "her", "hers",
  "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me",
  "my", "no", "not", "of", "on", "or", "our", "ours", "she", "should", "so",
  "than", "that", "the", "their", "theirs", "them", "then", "there", "these",
  "they", "this", "those", "to", "too", "us", "was", "we", "were", "what",
  "when", "where", "which", "who", "why", "will", "with", "would", "you",
  "your", "yours",
]);

const specificTermResults = (query, artifacts, count) => {
  const terms = [...new Set(lexicalTerms(query))]
    .filter((term) => Array.from(term).length > 1
      && !specificTermStopWords.has(term)
      && !artifacts.vocabularyIndex.has(term));
  if (terms.length === 0) {
    return [];
  }
  const ranked = artifacts.chunks.map((chunk, index) => {
    const title = new Set(lexicalTerms(chunk.title));
    const heading = new Set(lexicalTerms(chunk.heading ?? ""));
    const snippet = new Set(lexicalTerms(chunk.snippet));
    const score = terms.reduce((total, term) => total
      + (title.has(term) ? 4 : 0)
      + (heading.has(term) ? 2 : 0)
      + (snippet.has(term) ? 1 : 0), 0);
    return { index, score };
  });
  return bestDocuments(ranked, artifacts, count, Number.MIN_VALUE);
};

const keywordResults = (query, artifacts, count) => {
  const queryTerms = [...new Set(lexicalTerms(query))];
  if (queryTerms.length === 0) {
    return [];
  }
  const chunkTerms = artifacts.chunks.map((chunk) => lexicalTerms(
    `${chunk.title} ${chunk.heading ?? ""} ${chunk.snippet}`,
  ));
  const documentFrequency = new Map();
  for (const terms of chunkTerms) {
    const unique = new Set(terms);
    for (const term of queryTerms) {
      if (unique.has(term)) {
        documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
      }
    }
  }
  const averageLength = Math.max(1, chunkTerms.reduce((sum, terms) => sum + terms.length, 0)
    / Math.max(1, chunkTerms.length));
  const ranked = chunkTerms.map((terms, index) => {
    const frequencies = new Map();
    for (const term of terms) {
      frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
    }
    let score = 0;
    for (const term of queryTerms) {
      const frequency = frequencies.get(term) ?? 0;
      const frequencyWeight = (frequency * 2.2)
        / (frequency + 1.2 * (0.25 + 0.75 * (terms.length / averageLength)));
      const frequencyInDocuments = documentFrequency.get(term) ?? 0;
      score += Math.log(1 + (chunkTerms.length - frequencyInDocuments + 0.5)
        / (frequencyInDocuments + 0.5)) * frequencyWeight;
    }
    return { index, score };
  });
  return bestDocuments(ranked, artifacts, count, Number.MIN_VALUE);
};

const hybridResults = (specific, semantic, keyword, config) => {
  const combined = new Map();
  const addRanked = (results, weight) => {
    results.forEach((result, rank) => {
      const key = result.document;
      const current = combined.get(key) ?? { ...result, score: 0 };
      current.score += weight / (config.rrfK + rank + 1);
      combined.set(key, current);
    });
  };
  addRanked(specific, 1);
  addRanked(semantic, config.semanticWeight);
  addRanked(keyword, config.keywordWeight);
  return [...combined.values()].sort((left, right) => right.score - left.score);
};

const search = async (query) => {
  const artifacts = await loadArtifacts();
  const { config } = artifacts;
  const candidates = artifacts.chunks.length;
  const specific = config.specificTermHeuristic
    ? specificTermResults(query, artifacts, candidates)
    : [];
  const semantic = semanticResults(query, artifacts, candidates, config.minScore);
  const results = config.mode === "semantic"
    ? [...new Map([...specific, ...semantic].map((result) => [result.document, result])).values()]
    : hybridResults(
      specific,
      semantic,
      keywordResults(query, artifacts, candidates),
      config,
    );
  return {
    config,
    results: results.slice(0, config.limit),
  };
};

const findResultsContainer = (input) => {
  const searchRoot = input.closest('[data-md-component="search"]') ?? document;
  return searchRoot.querySelector('[data-md-component="search-result"]')
    ?? searchRoot.querySelector(".md-search-result");
};

const clearNode = (node) => {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
};

const resultHref = (result, config) => {
  const href = new URL(result.href, config.siteRootUrl);
  if (result.anchor) {
    href.hash = result.anchor;
  }
  return href.href;
};

const render = (container, query, results, config) => {
  clearNode(container);
  container.setAttribute("aria-live", "polite");
  container.setAttribute("aria-label", `Search results for ${query}`);

  const meta = document.createElement("div");
  meta.className = "md-search-result__meta";
  meta.textContent = results.length === 1 ? "1 matching document" : `${results.length} matching documents`;
  container.append(meta);

  const list = document.createElement("ol");
  list.className = "md-search-result__list";
  list.setAttribute("role", "list");
  for (const result of results) {
    const item = document.createElement("li");
    item.className = "md-search-result__item";

    const link = document.createElement("a");
    link.className = "md-search-result__link";
    link.href = resultHref(result, config);

    const article = document.createElement("article");
    article.className = "md-search-result__article md-search-result__article--document";

    const title = document.createElement("h2");
    title.className = "md-search-result__title";
    title.textContent = result.heading ? `${result.title} - ${result.heading}` : result.title;

    const teaser = document.createElement("p");
    teaser.className = "md-search-result__teaser";
    teaser.textContent = result.snippet;

    article.append(title, teaser);
    link.append(article);
    item.append(link);
    list.append(item);
  }
  container.append(list);
};

const renderError = (container) => {
  clearNode(container);
  const message = document.createElement("div");
  message.className = "md-search-result__meta";
  message.setAttribute("role", "status");
  message.textContent = "Search is currently unavailable.";
  container.append(message);
};

const initialize = (root = document) => {
  const inputs = root.querySelectorAll(
    'input[data-md-component="search-query"], [data-md-component="search-query"] input, input.md-search__input',
  );
  for (const input of inputs) {
    if (initializedInputs.has(input)) {
      continue;
    }
    initializedInputs.add(input);
    let request = 0;

    input.addEventListener("focus", () => {
      void loadArtifacts();
    }, { once: true });

    input.addEventListener("input", async (event) => {
      event.stopImmediatePropagation();
      const currentRequest = request + 1;
      request = currentRequest;
      const query = input.value.trim();
      if (!query) {
        const container = findResultsContainer(input);
        if (container) {
          clearNode(container);
        }
        return;
      }
      const container = findResultsContainer(input);
      if (!container) {
        return;
      }
      try {
        const response = await search(query);
        if (request === currentRequest && input.value.trim() === query) {
          render(container, query, response.results, response.config);
        }
      } catch (error) {
        console.error(error);
        if (request === currentRequest) {
          renderError(container);
        }
      }
    }, { capture: true });
  }
};

if (globalThis.document$?.subscribe) {
  globalThis.document$.subscribe(() => initialize());
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initialize(), { once: true });
} else {
  initialize();
}

if (!globalThis.document$?.subscribe) {
  const observer = new MutationObserver(() => initialize());
  observer.observe(document.documentElement, { childList: true, subtree: true });
}
