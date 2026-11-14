import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000 (uvicorn); requests to /api and /health are proxied.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
