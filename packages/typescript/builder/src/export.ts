import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { StaticModel } from "@websem/client";

interface SafeTensor {
  dtype: string;
  shape: unknown;
  data_offsets: unknown;
}

export interface ExportModelOptions {
  model: string;
  outputDirectory: string;
  dimensions?: number;
}

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

const hash = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex").slice(0, 12);

const artifactName = (
  prefix: string,
  extension: string,
  bytes: Uint8Array,
): string => `${prefix}.${hash(bytes)}.${extension}`;

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isSafeTensor = (value: unknown): value is SafeTensor =>
  isObject(value) &&
  typeof value.dtype === "string" &&
  "shape" in value &&
  "data_offsets" in value;

const parseVocabulary = (value: unknown): string[] => {
  if (
    !isObject(value) ||
    !isObject(value.model) ||
    !isObject(value.model.vocab)
  ) {
    throw new TypeError("tokenizer.json must contain a WordPiece vocabulary");
  }
  const entries = Object.entries(value.model.vocab).map(([token, id]) => {
    if (!Number.isInteger(id)) {
      throw new TypeError("tokenizer vocabulary must use contiguous token ids");
    }
    return [token, id as number] as const;
  });
  if (
    entries.length === 0 ||
    entries.some(
      ([token, id]) => token.length === 0 || id < 0 || id >= entries.length,
    )
  ) {
    throw new TypeError("tokenizer vocabulary must use contiguous token ids");
  }
  const vocabulary = new Array<string | undefined>(entries.length);
  for (const [token, id] of entries) {
    vocabulary[id] = token;
  }
  if (vocabulary.some((token) => token === undefined)) {
    throw new TypeError("tokenizer vocabulary must use contiguous token ids");
  }
  return vocabulary as string[];
};

const parseOffsets = (value: unknown, name: string): [number, number] => {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    value.some((offset) => !Number.isSafeInteger(offset) || offset < 0) ||
    (value[0] as number) > (value[1] as number)
  ) {
    throw new TypeError(`${name} must be two increasing byte offsets`);
  }
  return [value[0] as number, value[1] as number];
};

const parseShape = (value: unknown): [number, number] => {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    value.some((dimension) => !Number.isInteger(dimension) || dimension < 1)
  ) {
    throw new TypeError("embeddings must be a two-dimensional tensor");
  }
  return [value[0] as number, value[1] as number];
};

const decodeFloat16 = (value: number): number => {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 2 ** 10);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 2 ** 10);
};

const parseEmbeddings = (value: Uint8Array): Float32Array => {
  if (value.byteLength < 8) {
    throw new TypeError("model.safetensors is missing its header");
  }
  const view = new DataView(value.buffer, value.byteOffset, value.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength > value.byteLength - 8
  ) {
    throw new TypeError("model.safetensors has an invalid header length");
  }
  const headerValue: unknown = JSON.parse(
    textDecoder.decode(value.subarray(8, 8 + headerLength)),
  );
  if (!isObject(headerValue) || !isObject(headerValue.embeddings)) {
    throw new TypeError("model.safetensors is missing embeddings");
  }
  if (!isObject(headerValue) || !isSafeTensor(headerValue.embeddings)) {
    throw new TypeError("model.safetensors is missing embeddings");
  }
  const tensor = headerValue.embeddings;
  const [rows, dimensions] = parseShape(tensor.shape);
  const [start, end] = parseOffsets(tensor.data_offsets, "embedding offsets");
  const byteLength = end - start;
  const dtypeSize = tensor.dtype === "F16" ? 2 : tensor.dtype === "F32" ? 4 : 1;
  if (
    !["F16", "F32", "I8"].includes(tensor.dtype) ||
    byteLength !== rows * dimensions * dtypeSize
  ) {
    throw new TypeError("embeddings must use F16, F32, or I8 values");
  }
  const dataOffset = 8 + headerLength + start;
  if (dataOffset + byteLength > value.byteLength) {
    throw new TypeError("embedding offsets exceed model.safetensors");
  }
  const embeddings = new Float32Array(rows * dimensions);
  for (let index = 0; index < embeddings.length; index += 1) {
    const offset = dataOffset + index * dtypeSize;
    if (tensor.dtype === "F32") {
      embeddings[index] = view.getFloat32(offset, true);
    } else if (tensor.dtype === "F16") {
      embeddings[index] = decodeFloat16(view.getUint16(offset, true));
    } else {
      embeddings[index] = view.getInt8(offset);
    }
  }
  if (embeddings.some((embedding) => !Number.isFinite(embedding))) {
    throw new TypeError("embeddings must be finite");
  }
  return embeddings;
};

const quantize = (
  embeddings: Float32Array,
  vocabularySize: number,
  dimensions: number,
): Pick<StaticModel, "tokens" | "scales"> => {
  const tokens = new Int8Array(embeddings.length);
  const scales = new Float32Array(vocabularySize);
  for (let row = 0; row < vocabularySize; row += 1) {
    const offset = row * dimensions;
    let maximum = 0;
    for (let column = 0; column < dimensions; column += 1) {
      maximum = Math.max(
        maximum,
        Math.abs(embeddings[offset + column] as number),
      );
    }
    const scale = maximum === 0 ? 1 : maximum / 127;
    scales[row] = scale;
    for (let column = 0; column < dimensions; column += 1) {
      tokens[offset + column] = Math.max(
        -127,
        Math.min(
          127,
          Math.round((embeddings[offset + column] as number) / scale),
        ),
      );
    }
  }
  return { tokens, scales };
};

const isDirectory = async (path: string): Promise<boolean> => {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
};

const modelFiles = async (model: string): Promise<[Uint8Array, unknown]> => {
  if (await isDirectory(model)) {
    const [weights, tokenizer] = await Promise.all([
      readFile(join(model, "model.safetensors")),
      readFile(join(model, "tokenizer.json"), "utf8"),
    ]);
    return [weights, JSON.parse(tokenizer) as unknown];
  }
  if (!/^[\w.-]+\/[\w.-]+$/u.test(model)) {
    throw new TypeError(
      "model must be a local directory or Hugging Face model id",
    );
  }
  const baseUrl = `https://huggingface.co/${model}/resolve/main`;
  const [weightsResponse, tokenizerResponse] = await Promise.all([
    fetch(`${baseUrl}/model.safetensors`),
    fetch(`${baseUrl}/tokenizer.json`),
  ]);
  if (!weightsResponse.ok || !tokenizerResponse.ok) {
    throw new Error(`could not download Model2Vec files for ${model}`);
  }
  return [
    new Uint8Array(await weightsResponse.arrayBuffer()),
    (await tokenizerResponse.json()) as unknown,
  ];
};

const encodeScales = (scales: Float32Array): Uint8Array => {
  const bytes = new Uint8Array(scales.length * 4);
  const view = new DataView(bytes.buffer);
  scales.forEach((scale, index) => {
    view.setFloat32(index * 4, scale, true);
  });
  return bytes;
};

export const exportPortableModel = async ({
  model,
  outputDirectory,
  dimensions = 128,
}: ExportModelOptions): Promise<void> => {
  if (!Number.isInteger(dimensions) || dimensions < 1) {
    throw new RangeError("dimensions must be a positive integer");
  }
  const [weights, tokenizer] = await modelFiles(model);
  const vocabulary = parseVocabulary(tokenizer);
  const sourceEmbeddings = parseEmbeddings(weights);
  if (sourceEmbeddings.length % vocabulary.length !== 0) {
    throw new TypeError("embedding rows must match tokenizer vocabulary");
  }
  const sourceDimensions = sourceEmbeddings.length / vocabulary.length;
  if (dimensions > sourceDimensions) {
    throw new RangeError(`dimensions must not exceed ${sourceDimensions}`);
  }
  const embeddings = new Float32Array(vocabulary.length * dimensions);
  for (let row = 0; row < vocabulary.length; row += 1) {
    const sourceOffset = row * sourceDimensions;
    embeddings.set(
      sourceEmbeddings.subarray(sourceOffset, sourceOffset + dimensions),
      row * dimensions,
    );
  }
  const { tokens, scales } = quantize(
    embeddings,
    vocabulary.length,
    dimensions,
  );
  const vocabBytes = textEncoder.encode(JSON.stringify(vocabulary));
  const tokenBytes = new Uint8Array(tokens.buffer);
  const scaleBytes = encodeScales(scales);
  const names = {
    vocab: artifactName("vocab", "json", vocabBytes),
    tokens: artifactName("tokens", "bin", tokenBytes),
    scales: artifactName("scales", "bin", scaleBytes),
  };
  const descriptor = textEncoder.encode(
    JSON.stringify({ model_id: model, dims: dimensions, ...names }),
  );
  await mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    writeFile(join(outputDirectory, names.vocab), vocabBytes),
    writeFile(join(outputDirectory, names.tokens), tokenBytes),
    writeFile(join(outputDirectory, names.scales), scaleBytes),
    writeFile(join(outputDirectory, "model.json"), descriptor),
  ]);
};
