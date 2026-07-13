import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev: Vite serves the app with HMR and proxies /api to the Express backend.
// Prod: `vite build` emits dist/, which server.js serves statically.
export default defineConfig({
  plugins: [react()],
  // transformers.js resolves its ONNX wasm binaries at runtime; esbuild
  // pre-bundling breaks those URL lookups, so leave the package alone.
  optimizeDeps: { exclude: ['@xenova/transformers'] },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:3000',
    },
  },
});
