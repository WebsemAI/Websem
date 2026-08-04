import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { exportPortableModel } from "./export.js";

const defaultModel = "minishlab/potion-base-8M";
const usage =
  "Usage: websem-export-model --out <output-dir> [--model <model-id|model-dir>] [--dimensions 128]";

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

export const main = async (
  values: readonly string[] = process.argv.slice(2),
): Promise<void> => {
  let model = defaultModel;
  let outputDirectory: string | undefined;
  let dimensions = 128;
  for (let index = 0; index < values.length; index += 1) {
    const option = values[index];
    if (option === "--model") {
      model = requiredValue(values, index, option);
      index += 1;
    } else if (option === "--out") {
      outputDirectory = requiredValue(values, index, option);
      index += 1;
    } else if (option === "--dimensions") {
      dimensions = Number(requiredValue(values, index, option));
      if (!Number.isInteger(dimensions) || dimensions < 1) {
        throw new Error("--dimensions requires a positive integer");
      }
      index += 1;
    } else {
      throw new Error(`Unknown option: ${String(option)}\n${usage}`);
    }
  }
  if (outputDirectory === undefined) {
    throw new Error(usage);
  }
  await exportPortableModel({
    model,
    outputDirectory: resolve(outputDirectory),
    dimensions,
  });
};

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedPath)).href
) {
  await main();
}
