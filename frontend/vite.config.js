import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev: Vite serves the app with HMR and proxies /api to the Express backend.
// Prod: `vite build` emits dist/, which server.js serves statically.
export default defineConfig({
  plugins: [react()],
  // transformers.js pulls in onnxruntime-web, which ships CommonJS. It must go
  // through dep pre-bundling (CJS->ESM) or its backend registers against an
  // undefined module at runtime; do NOT add it to optimizeDeps.exclude.
  optimizeDeps: { include: ['@xenova/transformers'] },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:3000',
    },
  },
});
