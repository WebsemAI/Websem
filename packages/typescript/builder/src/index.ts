import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { embedText, type StaticModel } from "@websem/client";
import type { BuildDocument, ChunkRecord, Manifest } from "@websem/types";

const CHUNKER_VERSION = "1";
const ARTIFACT_PATTERN =
  /^(?:chunks|docs|tokens|scales|vocab)\.[a-f0-9]{12}\.(?:bin|json)$/u;
const textEncoder = new TextEncoder();

export type { BuildDocument } from "@websem/types";
export type { StaticModel } from "@websem/client";

export interface BuildOptions {
  chunkSize?: number;
  chunkOverlap?: number;
  titlePrefix?: boolean;
}

export interface BuiltArtifacts {
  manifest: Manifest;
  chunks: ChunkRecord[];
  docs: Int8Array;
  tokens: Int8Array;
  scales: Uint8Array;
  vocabulary: string[];
  files: Readonly<Record<string, Uint8Array>>;
}

export interface PortableModelDescriptor {
  model_id: string;
  dims: number;
  vocab: string;
  tokens: string;
  scales: string;
}

interface SourceSection {
  text: string;
  heading?: string;
  anchor?: string;
}

const assertPositiveInteger = (value: number, name: string): void => {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer`);
  }
};

const normalizeText = (text: string): string =>
  text.replace(/\s+/gu, " ").trim();

const chunkSection = (
  textValue: string,
  chunkSize: number,
  overlap: number,
): string[] => {
  const text = normalizeText(textValue);
  if (text.length === 0) {
    return [];
  }
  const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'([])/u);
  const chunks: string[] = [];
  let index = 0;
  while (index < sentences.length) {
    const current: string[] = [];
    let length = 0;
    let end = index;
    while (end < sentences.length) {
      const sentence = sentences[end] as string;
      const nextLength =
        length + sentence.length + (current.length > 0 ? 1 : 0);
      if (current.length > 0 && nextLength > chunkSize) {
        break;
      }
      current.push(sentence);
      length = nextLength;
      end += 1;
    }
    chunks.push(current.join(" "));
    if (end >= sentences.length) {
      break;
    }
    let overlapLength = 0;
    let nextIndex = end;
    while (nextIndex > index + 1 && overlapLength < overlap) {
      nextIndex -= 1;
      overlapLength += (sentences[nextIndex] as string).length + 1;
    }
    index = nextIndex;
  }
  return chunks;
};

const documentSections = (document: BuildDocument): SourceSection[] => {
  if (document.sections === undefined || document.sections.length === 0) {
    return [{ text: document.text }];
  }
  return document.sections.map((section) => {
    const source: SourceSection = { text: section.text };
    if (section.heading !== undefined) {
      source.heading = section.heading;
    }
    if (section.anchor !== undefined) {
      source.anchor = section.anchor;
    }
    return source;
  });
};

export const chunkDocuments = (
  documents: readonly BuildDocument[],
  options: BuildOptions = {},
): ChunkRecord[] => {
  const chunkSize = options.chunkSize ?? 600;
  const overlap = options.chunkOverlap ?? 120;
  assertPositiveInteger(chunkSize, "chunkSize");
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= chunkSize) {
    throw new RangeError(
      "chunkOverlap must be an integer from 0 to chunkSize - 1",
    );
  }
  const chunks: ChunkRecord[] = [];
  const identifiers = new Set<string>();
  for (const document of documents) {
    if (document.id.length === 0 || document.title.length === 0) {
      throw new TypeError("document id and title must be non-empty");
    }
    if (identifiers.has(document.id)) {
      throw new TypeError(`duplicate document id: ${document.id}`);
    }
    identifiers.add(document.id);
    for (const section of documentSections(document)) {
      for (const snippet of chunkSection(section.text, chunkSize, overlap)) {
        const chunk: ChunkRecord = {
          document: document.id,
          title: document.title,
          href: document.href,
          snippet,
        };
        if (section.heading !== undefined) {
          chunk.heading = section.heading;
        }
        if (section.anchor !== undefined) {
          chunk.anchor = section.anchor;
        }
        chunks.push(chunk);
      }
    }
  }
  return chunks;
};

const encodeScales = (scales: Float32Array): Uint8Array => {
  const bytes = new Uint8Array(scales.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < scales.length; index += 1) {
    view.setFloat32(index * 4, scales[index] as number, true);
  }
  return bytes;
};

const quantizeDocuments = (
  embeddings: readonly Float32Array[],
  dimensions: number,
): Int8Array => {
  const output = new Int8Array(embeddings.length * dimensions);
  embeddings.forEach((embedding, chunkIndex) => {
    const norm = Math.sqrt(
      embedding.reduce((sum, value) => sum + value * value, 0),
    );
    if (!Number.isFinite(norm) || Math.abs(norm - 1) > 1e-3) {
      throw new TypeError(
        `document embedding ${chunkIndex} must be L2-normalized`,
      );
    }
    embedding.forEach((value, dimension) => {
      output[chunkIndex * dimensions + dimension] = Math.max(
        -127,
        Math.min(127, Math.round(value * 127)),
      );
    });
  });
  return output;
};

const jsonBytes = (value: unknown): Uint8Array =>
  textEncoder.encode(JSON.stringify(value));

const hash = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex").slice(0, 12);

const hashedName = (
  prefix: string,
  extension: string,
  bytes: Uint8Array,
): string => `${prefix}.${hash(bytes)}.${extension}`;

export const buildArtifacts = (
  documents: readonly BuildDocument[],
  model: StaticModel,
  options: BuildOptions = {},
): BuiltArtifacts => {
  assertPositiveInteger(model.dimensions, "model.dimensions");
  if (model.modelId.length === 0 || model.vocabulary.length === 0) {
    throw new TypeError("model id and vocabulary must be non-empty");
  }
  if (new Set(model.vocabulary).size !== model.vocabulary.length) {
    throw new TypeError("model vocabulary tokens must be unique");
  }
  if (model.tokens.length !== model.vocabulary.length * model.dimensions) {
    throw new TypeError(
      "model token length does not match vocabulary * dimensions",
    );
  }
  if (
    model.scales.length !== model.vocabulary.length ||
    model.scales.some((scale) => !Number.isFinite(scale) || scale <= 0)
  ) {
    throw new TypeError(
      "model scales must match vocabulary and be finite and positive",
    );
  }
  const chunkSize = options.chunkSize ?? 600;
  const chunkOverlap = options.chunkOverlap ?? 120;
  const titlePrefix = options.titlePrefix ?? true;
  const chunks = chunkDocuments(documents, { chunkSize, chunkOverlap });
  const embeddings = chunks.map((chunk) =>
    embedText(
      titlePrefix ? `${chunk.title}\n\n${chunk.snippet}` : chunk.snippet,
      model,
    ),
  );
  const docs = quantizeDocuments(embeddings, model.dimensions);
  const tokens = new Int8Array(model.tokens);
  const scales = encodeScales(model.scales);
  const vocabulary = [...model.vocabulary];
  const chunkBytes = jsonBytes(chunks);
  const docBytes = new Uint8Array(
    docs.buffer,
    docs.byteOffset,
    docs.byteLength,
  );
  const tokenBytes = new Uint8Array(
    tokens.buffer,
    tokens.byteOffset,
    tokens.byteLength,
  );
  const vocabBytes = jsonBytes(vocabulary);
  const names = {
    chunks: hashedName("chunks", "json", chunkBytes),
    docs: hashedName("docs", "bin", docBytes),
    tokens: hashedName("tokens", "bin", tokenBytes),
    scales: hashedName("scales", "bin", scales),
    vocab: hashedName("vocab", "json", vocabBytes),
  };
  const manifest: Manifest = {
    websem_version: "1.0",
    model_id: model.modelId,
    dims: model.dimensions,
    vocab_size: vocabulary.length,
    n_chunks: chunks.length,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
    title_prefix: titlePrefix,
    chunker_version: CHUNKER_VERSION,
    doc_scale: 127,
    files: names,
  };
  return {
    manifest,
    chunks,
    docs,
    tokens,
    scales,
    vocabulary,
    files: {
      [names.chunks]: chunkBytes,
      [names.docs]: docBytes,
      [names.tokens]: tokenBytes,
      [names.scales]: scales,
      [names.vocab]: vocabBytes,
      "manifest.json": jsonBytes(manifest),
    },
  };
};

export const writeArtifacts = async (
  outputDirectory: string,
  artifacts: BuiltArtifacts,
): Promise<void> => {
  await mkdir(outputDirectory, { recursive: true });
  const current = new Set(Object.keys(artifacts.files));
  for (const entry of await readdir(outputDirectory, { withFileTypes: true })) {
    if (
      entry.isFile() &&
      ARTIFACT_PATTERN.test(entry.name) &&
      !current.has(entry.name)
    ) {
      await rm(join(outputDirectory, entry.name));
    }
  }
  await Promise.all(
    Object.entries(artifacts.files).map(([name, bytes]) =>
      writeFile(join(outputDirectory, name), bytes),
    ),
  );
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const descriptorString = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`model.json ${name} must be a non-empty string`);
  }
  return value;
};

const parseDescriptor = (value: unknown): PortableModelDescriptor => {
  if (!isObject(value)) {
    throw new TypeError("model.json must contain an object");
  }
  const expected = ["model_id", "dims", "vocab", "tokens", "scales"];
  if (
    Object.keys(value).length !== expected.length ||
    Object.keys(value).some((key) => !expected.includes(key))
  ) {
    throw new TypeError("model.json contains unexpected or missing fields");
  }
  if (!Number.isInteger(value.dims) || (value.dims as number) < 1) {
    throw new TypeError("model.json dims must be a positive integer");
  }
  return {
    model_id: descriptorString(value.model_id, "model_id"),
    dims: value.dims as number,
    vocab: descriptorString(value.vocab, "vocab"),
    tokens: descriptorString(value.tokens, "tokens"),
    scales: descriptorString(value.scales, "scales"),
  };
};

const decodeScales = (bytes: Uint8Array): Float32Array => {
  if (bytes.byteLength % 4 !== 0) {
    throw new TypeError("model scales byte length must be divisible by four");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const scales = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < scales.length; index += 1) {
    scales[index] = view.getFloat32(index * 4, true);
  }
  return scales;
};

export const loadPortableModel = async (
  directory: string,
): Promise<StaticModel> => {
  const descriptorValue: unknown = JSON.parse(
    await readFile(join(directory, "model.json"), "utf8"),
  );
  const descriptor = parseDescriptor(descriptorValue);
  const vocabularyValue: unknown = JSON.parse(
    await readFile(join(directory, descriptor.vocab), "utf8"),
  );
  if (
    !Array.isArray(vocabularyValue) ||
    vocabularyValue.some((token) => typeof token !== "string")
  ) {
    throw new TypeError("model vocabulary must be a JSON string array");
  }
  const vocabulary = vocabularyValue as string[];
  if (
    vocabulary.length === 0 ||
    new Set(vocabulary).size !== vocabulary.length
  ) {
    throw new TypeError("model vocabulary must be non-empty and unique");
  }
  const tokenBytes = await readFile(join(directory, descriptor.tokens));
  const tokens = new Int8Array(
    tokenBytes.buffer.slice(
      tokenBytes.byteOffset,
      tokenBytes.byteOffset + tokenBytes.byteLength,
    ),
  );
  const scaleBytes = await readFile(join(directory, descriptor.scales));
  const scales = decodeScales(scaleBytes);
  if (tokens.length !== vocabulary.length * descriptor.dims) {
    throw new TypeError(
      "model token byte length does not match vocab length * dims",
    );
  }
  if (
    scales.length !== vocabulary.length ||
    scales.some((scale) => !Number.isFinite(scale) || scale <= 0)
  ) {
    throw new TypeError(
      "model scales must match vocabulary and be finite and positive",
    );
  }
  return {
    modelId: descriptor.model_id,
    dimensions: descriptor.dims,
    vocabulary,
    tokens,
    scales,
  };
};
