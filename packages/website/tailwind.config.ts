import type { Config } from "tailwindcss";

export default {
  // Dark follows the operating system with no JavaScript at all, and the
  // toggle overrides it in either direction by putting .dark or .light on the
  // root. A class-only strategy would leave a JavaScript-disabled visitor
  // permanently in light mode.
  darkMode: [
    "variant",
    [
      "@media (prefers-color-scheme: dark) { &:not(.light *) }",
      "&:is(.dark *)",
    ],
  ],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Album paper and photographic ink, rather than a generic slate ramp.
        paper: {
          50: "#fbfaf7",
          100: "#f4f1ea",
          200: "#e7e1d5",
          300: "#d4cbb8",
        },
        ink: {
          700: "#3d3a34",
          800: "#2a2823",
          900: "#1a1917",
          950: "#111110",
        },
        ember: {
          400: "#e8a13c", // 8.63:1 on ink-950 — text on dark
          500: "#d8892a", // focus rings and borders, not text
          600: "#b56d1c", // 3.89:1 on paper-50 — large text and icons only
          700: "#96590c", // 5.39:1 on paper-50 — body text on light
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      maxWidth: { prose: "68ch" },
    },
  },
  plugins: [],
} satisfies Config;
