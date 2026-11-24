import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        alabaster: "#FAF8F4",
        parchment: "#F3EFE7",
        paper: "#FFFFFF",
        stonehair: "#E5DFD3",
        obsidian: "#181715",
        inkmuted: "#5C574F",
        forest: "#1C4D3A",
        terracotta: "#A84420",
        amberdeep: "#9A6218",
      },
      fontFamily: {
        serif: ["var(--font-newsreader)", "Georgia", "serif"],
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
