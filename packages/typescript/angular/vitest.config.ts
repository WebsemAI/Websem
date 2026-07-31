import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@websem/client": fileURLToPath(
        new URL("../client/src/index.ts", import.meta.url),
      ),
    },
  },
  test: {
    environment: "node",
  },
});
