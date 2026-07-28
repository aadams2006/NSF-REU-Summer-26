import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/NSF-REU-Summer-26/" : "/",
  server: {
    host: "0.0.0.0",
    allowedHosts: ["terminal.local"],
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
});
