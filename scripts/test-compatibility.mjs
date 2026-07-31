import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { URL } from "node:url";

import {
  buildArtifacts,
  writeArtifacts,
} from "../packages/typescript/builder/dist/index.js";
import {
  loadArtifacts,
  semanticSearch,
} from "../packages/typescript/client/dist/index.js";

const runPython = (source, directory) => {
  const result = spawnSync("uv", ["run", "python", "-c", source, directory], {
    cwd: new URL("..", import.meta.url),
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
};

const root = await mkdtemp(join(tmpdir(), "websem-compat-"));
const pythonOutput = join(root, "python");
const typescriptOutput = join(root, "typescript");

try {
  runPython(
    `
import sys
import numpy as np
from websem_builder import build_from_arrays

build_from_arrays(
    [{"id": "python", "title": "Alpha", "href": "/python/", "text": "Alpha guide."}],
    model_id="compat/model",
    out_dir=sys.argv[1],
    token_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    vocab=["alpha", "beta"],
    encoder=lambda texts: np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)),
)
`,
    pythonOutput,
  );

  const manifest = JSON.parse(
    await readFile(join(pythonOutput, "manifest.json"), "utf8"),
  );
  const files = manifest.files;
  const pythonArtifacts = loadArtifacts(manifest, {
    chunks: await readFile(join(pythonOutput, files.chunks)),
    docs: await readFile(join(pythonOutput, files.docs)),
    tokens: await readFile(join(pythonOutput, files.tokens)),
    scales: await readFile(join(pythonOutput, files.scales)),
    vocab: await readFile(join(pythonOutput, files.vocab)),
  });
  const [typescriptResult] = semanticSearch("alpha", pythonArtifacts);
  if (typescriptResult?.document !== "python") {
    throw new Error("TypeScript client could not read Python artifacts");
  }

  const model = {
    modelId: "compat/model",
    dimensions: 2,
    vocabulary: ["alpha", "beta"],
    tokens: new Int8Array([127, 0, 0, 127]),
    scales: new Float32Array([1 / 127, 1 / 127]),
  };
  await writeArtifacts(
    typescriptOutput,
    buildArtifacts(
      [
        {
          id: "typescript",
          title: "Alpha",
          href: "/typescript/",
          text: "Alpha guide.",
        },
      ],
      model,
    ),
  );
  runPython(
    `
import sys
from websem_client import WebsemClient, load_artifacts

results = WebsemClient(load_artifacts(sys.argv[1])).semantic_search("alpha")
assert results and results[0]["document"] == "typescript"
`,
    typescriptOutput,
  );
} finally {
  await rm(root, { recursive: true, force: true });
}
