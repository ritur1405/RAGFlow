import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app calls the API with relative paths (see API_BASE in the dashboard), so
// the dev server proxies every backend route to the local backend. In the
// Docker image the same relative paths are proxied by nginx instead, which
// means one set of URLs works in development and in production.
const backend = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";

const proxied = ["/api", "/query", "/upload", "/documents"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      proxied.map((path) => [
        path,
        { target: backend, changeOrigin: true, secure: false },
      ]),
    ),
  },
});
