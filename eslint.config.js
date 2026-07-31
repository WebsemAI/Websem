import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  {
    ignores: ["**/dist/**", "**/coverage/**", "packages/python/**"],
  },
  {
    files: ["packages/typescript/**/*.ts"],
    extends: [...tseslint.configs.strictTypeChecked],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true },
      ],
    },
  },
  {
    ...tseslint.configs.disableTypeChecked,
    files: [
      "packages/typescript/**/test/**/*.ts",
      "packages/typescript/**/vitest.config.ts",
    ],
    languageOptions: {
      parserOptions: {
        project: false,
        projectService: false,
      },
    },
  },
);
