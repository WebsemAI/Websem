#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import type { BuildDocument } from "@websem/types";

import { buildArtifacts, loadPortableModel, writeArtifacts } from "./index.js";

interface Arguments {
  docs: string;
  model: string;
  out: string;
  chunkSize?: number;
  chunkOverlap?: number;
  titlePrefix?: boolean;
}

const usage =
  "Usage: websem-build --docs <docs.json|docs.jsonl> --model <model-dir> --out <output-dir> [--chunk-size 600] [--chunk-overlap 120] [--no-title-prefix]";

const requiredValue = (
  values: readonly string[],
  index: number,
  option: string,
): string => {
  const value = values[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${option} requires a value\n${usage}`);
  }
  return value;
};

const parseInteger = (value: string, option: string): number => {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${option} requires an integer`);
  }
  return parsed;
};

const parseArguments = (values: readonly string[]): Arguments => {
  const parsed: Partial<Arguments> = {};
  for (let index = 0; index < values.length; index += 1) {
    const option = values[index];
    if (option === "--docs") {
      parsed.docs = requiredValue(values, index, option);
      index += 1;
    } else if (option === "--model") {
      parsed.model = requiredValue(values, index, option);
      index += 1;
    } else if (option === "--out") {
      parsed.out = requiredValue(values, index, option);
      index += 1;
    } else if (option === "--chunk-size") {
      parsed.chunkSize = parseInteger(
        requiredValue(values, index, option),
        option,
      );
      index += 1;
    } else if (option === "--chunk-overlap") {
      parsed.chunkOverlap = parseInteger(
        requiredValue(values, index, option),
        option,
      );
      index += 1;
    } else if (option === "--no-title-prefix") {
      parsed.titlePrefix = false;
    } else {
      throw new Error(`Unknown option: ${String(option)}\n${usage}`);
    }
  }
  if (
    parsed.docs === undefined ||
    parsed.model === undefined ||
    parsed.out === undefined
  ) {
    throw new Error(usage);
  }
  return parsed as Arguments;
};

const isBuildDocument = (value: unknown): value is BuildDocument => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const document = value as Record<string, unknown>;
  if (
    ![document.id, document.title, document.href, document.text].every(
      (field) => typeof field === "string",
    )
  ) {
    return false;
  }
  if (document.sections === undefined) {
    return true;
  }
  if (!Array.isArray(document.sections)) {
    return false;
  }
  return document.sections.every((value) => {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return false;
    }
    const section = value as Record<string, unknown>;
    if (
      typeof section.text !== "string" ||
      (section.heading !== undefined && typeof section.heading !== "string") ||
      (section.anchor !== undefined && typeof section.anchor !== "string")
    ) {
      return false;
    }
    return Object.keys(section).every((key) =>
      ["text", "heading", "anchor"].includes(key),
    );
  });
};

const loadDocuments = async (path: string): Promise<BuildDocument[]> => {
  const content = await readFile(path, "utf8");
  let values: unknown[];
  if (path.endsWith(".jsonl")) {
    values = content
      .split(/\r?\n/u)
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as unknown);
  } else {
    const parsed: unknown = JSON.parse(content);
    if (!Array.isArray(parsed)) {
      throw new TypeError("docs JSON must contain an array");
    }
    values = parsed;
  }
  if (!values.every(isBuildDocument)) {
    throw new TypeError("every input item must be a BuildDocument");
  }
  return values;
};

export const main = async (
  values: readonly string[] = process.argv.slice(2),
): Promise<void> => {
  const args = parseArguments(values);
  const documents = await loadDocuments(resolve(args.docs));
  const model = await loadPortableModel(resolve(args.model));
  const options: Parameters<typeof buildArtifacts>[2] = {};
  if (args.chunkSize !== undefined) {
    options.chunkSize = args.chunkSize;
  }
  if (args.chunkOverlap !== undefined) {
    options.chunkOverlap = args.chunkOverlap;
  }
  if (args.titlePrefix !== undefined) {
    options.titlePrefix = args.titlePrefix;
  }
  const artifacts = buildArtifacts(documents, model, options);
  await writeArtifacts(resolve(args.out), artifacts);
};

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedPath)).href
) {
  await main();
}
