import { describe, expect, it } from "vitest";

import {
  basicTokenize,
  embedText,
  hybridSearch,
  loadArtifacts,
  semanticSearch,
  tokenize,
  type StaticModel,
} from "../src/index.js";

const vocabulary = ["hello", "world", "你", "cafe", "!", "play", "##ing"];
const model: StaticModel = {
  modelId: "test",
  dimensions: 2,
  vocabulary,
  tokens: new Int8Array([127, 0, 120, 0, 0, 127, 0, 100, 0, 0, 0, 80, 0, 127]),
  scales: new Float32Array(vocabulary.length).fill(1 / 127),
};

const artifacts = loadArtifacts(
  {
    websem_version: "1.0",
    model_id: "test",
    dims: 2,
    vocab_size: vocabulary.length,
    n_chunks: 3,
    chunk_size: 600,
    chunk_overlap: 120,
    title_prefix: true,
    chunker_version: "1",
    doc_scale: 127,
    files: {
      chunks: "chunks.x.json",
      docs: "docs.x.bin",
      tokens: "tokens.x.bin",
      scales: "scales.x.bin",
      vocab: "vocab.x.json",
    },
  },
  {
    chunks: JSON.stringify([
      { document: "b", title: "Other", href: "/b", snippet: "world" },
      { document: "a", title: "Hello", href: "/a", snippet: "hello world" },
      {
        document: "a",
        title: "Hello later",
        href: "/a#later",
        snippet: "hello",
      },
    ]),
    docs: new Int8Array([0, 127, 127, 0, 127, 0]),
    tokens: model.tokens,
    scales: model.scales,
    vocab: JSON.stringify(vocabulary),
  },
);

describe("BERT tokenization", () => {
  it("cleans, lowercases, strips accents, spaces CJK, and splits punctuation", () => {
    expect(basicTokenize("\u0000 HéLLo你!")).toEqual(["hello", "你", "!"]);
  });

  it("uses greedy continuation pieces and drops unknown words", () => {
    expect(tokenize("playing unknown", vocabulary)).toEqual([5, 6]);
    expect(tokenize("x".repeat(101), vocabulary)).toEqual([]);
  });
});

describe("embedding and search", () => {
  it("mean-pools and normalizes known token vectors", () => {
    expect([...embedText("hello world", model)]).toEqual([1, 0]);
  });

  it("keeps one deterministic best chunk per document", () => {
    expect(
      semanticSearch("hello", artifacts).map((result) => result.href),
    ).toEqual(["/a", "/b"]);
  });

  it("combines semantic and lexical ranks with weighted RRF", () => {
    const results = hybridSearch("world", artifacts, {
      semanticWeight: 0,
      keywordWeight: 1,
      rrfK: 0,
    });
    expect(results[0]?.document).toBe("b");
    expect(results[0]?.score).toBe(1);
  });

  it("surfaces case-insensitive exact matches for terms absent from the model", () => {
    expect(semanticSearch("OTHER", artifacts)[0]?.document).toBe("b");
    expect(
      semanticSearch("OTHER", artifacts, { specificTermHeuristic: false }),
    ).toEqual([]);
    expect(
      hybridSearch("OTHER", artifacts, {
        semanticWeight: 0,
        keywordWeight: 0,
      })[0]?.document,
    ).toBe("b");
    const stopwordArtifacts = {
      ...artifacts,
      chunks: artifacts.chunks.map((chunk, index) =>
        index === 0 ? { ...chunk, snippet: "world a the" } : chunk,
      ),
    };
    expect(semanticSearch("A", stopwordArtifacts)).toEqual([]);
    expect(semanticSearch("THE", stopwordArtifacts)).toEqual([]);
  });
});

describe("artifact validation", () => {
  it("rejects incompatible versions and malformed byte lengths", () => {
    expect(() =>
      loadArtifacts(
        { ...artifacts.manifest, websem_version: "2.0" },
        {
          chunks: "[]",
          docs: new Int8Array(),
          tokens: new Int8Array(),
          scales: new Uint8Array(),
          vocab: "[]",
        },
      ),
    ).toThrow(/Unsupported/);
    expect(() =>
      loadArtifacts(artifacts.manifest, {
        chunks: JSON.stringify(artifacts.chunks),
        docs: new Int8Array(),
        tokens: artifacts.tokens,
        scales: artifacts.scales,
        vocab: JSON.stringify(vocabulary),
      }),
    ).toThrow(/docs byte length/);
  });
});
