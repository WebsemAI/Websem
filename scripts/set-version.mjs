import { readFile, writeFile } from "node:fs/promises";
import { argv, stdout } from "node:process";

const npmPackages = [
  "package.json",
  "packages/typescript/types/package.json",
  "packages/typescript/client/package.json",
  "packages/typescript/builder/package.json",
  "packages/typescript/angular/package.json",
];
const pythonPackages = [
  "pyproject.toml",
  "packages/python/websem-types/pyproject.toml",
  "packages/python/websem-client/pyproject.toml",
  "packages/python/websem-builder/pyproject.toml",
  "packages/python/mkdocs-websem/pyproject.toml",
];
const internalPythonRequirement =
  /"(mkdocs-websem|websem-(?:types|client|builder))[<>=!~][^"]*"/g;

const rootPackage = JSON.parse(await readFile("package.json", "utf8"));
const requestedVersion = argv[2];
if (!requestedVersion) {
  throw new Error("Provide patch, minor, major, or an explicit x.y.z version.");
}

const current = rootPackage.version.split(".").map(Number);
const bumps = {
  major: [current[0] + 1, 0, 0],
  minor: [current[0], current[1] + 1, 0],
  patch: [current[0], current[1], current[2] + 1],
};
const bump = bumps[requestedVersion];
const version = bump ? bump.join(".") : requestedVersion;
if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
  throw new Error(`Invalid version: ${requestedVersion}`);
}

for (const packagePath of npmPackages) {
  const packageJson = JSON.parse(await readFile(packagePath, "utf8"));
  packageJson.version = version;
  for (const dependencyGroup of [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
  ]) {
    for (const dependency of Object.keys(packageJson[dependencyGroup] ?? {})) {
      if (dependency.startsWith("@websem/")) {
        packageJson[dependencyGroup][dependency] = version;
      }
    }
  }
  await writeFile(
    packagePath,
    `${JSON.stringify(packageJson, undefined, 2)}\n`,
  );
}

for (const packagePath of pythonPackages) {
  const pyproject = await readFile(packagePath, "utf8");
  const updated = pyproject
    .replace(/^version = "[^"]+"$/m, `version = "${version}"`)
    .replace(internalPythonRequirement, `"$1==${version}"`);
  await writeFile(packagePath, updated);
}

stdout.write(`${version}\n`);
