import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Dev: Vite serves the app with HMR and proxies /api to the Express backend.
// Prod: `vite build` emits dist/, which server.js serves statically.
export default defineConfig(({ mode }) => {
  // The project's single .env lives at the repo root, not in frontend/, and the
  // '' prefix lets us read plain names (no VITE_ prefix) — CLIENT_EMBEDDING_MODEL
  // is one setting shared with fetch_model.mjs, which vendors that same model.
  const env = loadEnv(mode, ROOT, '');

  return {
    plugins: [react()],
    define: {
      __CLIENT_EMBEDDING_MODEL__: JSON.stringify(
        env.CLIENT_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L12-v2'),
    },
    // transformers.js pulls in onnxruntime-web, which ships CommonJS. It must go
    // through dep pre-bundling (CJS->ESM) or its backend registers against an
    // undefined module at runtime; do NOT add it to optimizeDeps.exclude.
    optimizeDeps: { include: ['@xenova/transformers'] },
    server: {
      port: 5173,
      proxy: {
        '/api': 'http://localhost:3000',
        // The browser embedding model + onnx wasm, vendored by npm run fetch:model
        '/models': 'http://localhost:3000',
      },
    },
  };
});
