import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { main } from "../src/cli.js";
import {
  buildArtifacts,
  chunkDocuments,
  writeArtifacts,
  type StaticModel,
} from "../src/index.js";

const model: StaticModel = {
  modelId: "portable/test",
  dimensions: 2,
  vocabulary: ["title", "first", "sentence", "second", "third"],
  tokens: new Int8Array([127, 0, 0, 127, 0, 120, 127, 0, 100, 0]),
  scales: new Float32Array(5).fill(1 / 127),
};

describe("chunking", () => {
  it("keeps sections separate and snaps chunks at sentence boundaries", () => {
    const chunks = chunkDocuments(
      [
        {
          id: "doc",
          title: "Title",
          href: "/doc",
          text: "ignored",
          sections: [
            {
              heading: "One",
              anchor: "one",
              text: "First sentence. Second sentence. Third sentence.",
            },
            { heading: "Two", anchor: "two", text: "First sentence." },
          ],
        },
      ],
      { chunkSize: 35, chunkOverlap: 10 },
    );
    expect(chunks).toHaveLength(3);
    expect(chunks[0]).toMatchObject({
      heading: "One",
      anchor: "one",
      href: "/doc",
    });
    expect(chunks.at(-1)).toMatchObject({
      heading: "Two",
      snippet: "First sentence.",
    });
  });

  it("allows the root document to use an empty href", () => {
    const [chunk] = chunkDocuments([
      { id: "home", title: "Home", href: "", text: "First sentence." },
    ]);
    expect(chunk?.href).toBe("");
  });
});

describe("artifact building", () => {
  it("writes deterministic content-hashed bytes and an unhashed manifest", async () => {
    const built = buildArtifacts(
      [{ id: "doc", title: "Title", href: "/doc", text: "First sentence." }],
      model,
    );
    expect(built.manifest.files.docs).toMatch(/^docs\.[a-f0-9]{12}\.bin$/u);
    expect(built.docs).toHaveLength(2);
    expect(new DataView(built.scales.buffer).getFloat32(0, true)).toBeCloseTo(
      1 / 127,
    );

    const output = await mkdtemp(join(tmpdir(), "websem-builder-"));
    await writeFile(join(output, "docs.000000000000.bin"), "obsolete");
    await writeArtifacts(output, built);
    const names = await readdir(output);
    expect(names).toContain("manifest.json");
    expect(names).not.toContain("docs.000000000000.bin");
    const manifest = JSON.parse(
      await readFile(join(output, "manifest.json"), "utf8"),
    ) as unknown;
    expect(manifest).toEqual(built.manifest);
  });

  it("builds JSONL input from a portable model directory", async () => {
    const root = await mkdtemp(join(tmpdir(), "websem-cli-"));
    const modelDirectory = join(root, "model");
    const outputDirectory = join(root, "output");
    await writeFile(
      join(root, "docs.jsonl"),
      `${JSON.stringify({ id: "doc", title: "Title", href: "/doc", text: "First sentence." })}\n`,
    );
    await mkdir(modelDirectory);
    await writeFile(
      join(modelDirectory, "model.json"),
      JSON.stringify({
        model_id: model.modelId,
        dims: model.dimensions,
        vocab: "vocab.json",
        tokens: "tokens.bin",
        scales: "scales.bin",
      }),
    );
    const scaleBytes = new Uint8Array(model.scales.length * 4);
    const scaleView = new DataView(scaleBytes.buffer);
    model.scales.forEach((scale, index) =>
      scaleView.setFloat32(index * 4, scale, true),
    );
    await Promise.all([
      writeFile(
        join(modelDirectory, "vocab.json"),
        JSON.stringify(model.vocabulary),
      ),
      writeFile(join(modelDirectory, "tokens.bin"), model.tokens),
      writeFile(join(modelDirectory, "scales.bin"), scaleBytes),
    ]);
    await main([
      "--docs",
      join(root, "docs.jsonl"),
      "--model",
      modelDirectory,
      "--out",
      outputDirectory,
    ]);
    expect(await readdir(outputDirectory)).toContain("manifest.json");
  });
});
