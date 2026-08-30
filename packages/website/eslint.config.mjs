import { FlatCompat } from "@eslint/eslintrc";

// eslint-config-next still ships the legacy shape, so it is bridged into flat
// config rather than hand-rewriting Next's rule set.
const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  {
    ignores: ["node_modules/**", ".next/**", "out/**", "next-env.d.ts"],
  },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // Every figure states what it shows; an empty alt would be a bug here.
      "jsx-a11y/alt-text": "error",
    },
  },
];

export default config;
