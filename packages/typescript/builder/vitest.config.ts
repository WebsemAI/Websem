import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
  },
  resolve: {
    alias: {
      "@websem/client": fileURLToPath(
        new URL("../client/src/index.ts", import.meta.url),
      ),
    },
  },
});
