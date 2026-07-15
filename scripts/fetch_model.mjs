/**
 * fetch_model.mjs — vendor the browser embedding model into models/
 *
 * Chat.jsx embeds the user's query in the browser with the browser cache
 * deliberately OFF, so the model is re-fetched on every session. Pulling it
 * from huggingface.co each time makes chat depend on the public internet and
 * fail with an opaque network error whenever HF is slow, rate-limiting, or
 * unreachable. Instead we vendor the files once and serve them from our own
 * backend (/models), which keeps the embedding in-browser and the cache off
 * while removing the external dependency.
 *
 * Also copies out of node_modules:
 *   - onnxruntime's wasm (transformers.js otherwise pulls it from
 *     cdn.jsdelivr.net at runtime)
 *   - pdf.js's wasm image decoders + standard fonts. Scanned PDFs (e.g.
 *     JBIG2-compressed scans) render as blank white pages without the wasm
 *     decoders, and pdf.js can't resolve them itself in a bundled app.
 *
 * Run: npm run fetch:model     (idempotent; skips files already present)
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT       = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MODELS_DIR = path.join(ROOT, 'models');
// The model the BROWSER loads — same env var vite.config.js bakes into the app,
// so vendoring and loading can never drift apart.
const MODEL_ID   = process.env.CLIENT_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L12-v2';
const BASE       = `https://huggingface.co/${MODEL_ID}/resolve/main`;

// quantized: true (Chat.jsx) → model_quantized.onnx, not model.onnx.
const FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'onnx/model_quantized.onnx',
];

const ORT_SRC = path.join(ROOT, 'frontend', 'node_modules', '@xenova', 'transformers', 'dist');
const ORT_WASM = ['ort-wasm.wasm', 'ort-wasm-simd.wasm', 'ort-wasm-threaded.wasm', 'ort-wasm-simd-threaded.wasm'];

const megabytes = (byteCount) => `${(byteCount / 1048576).toFixed(1)} MB`;

async function exists(filePath) {
  return fs.access(filePath).then(() => true).catch(() => false);
}

async function download(relPath) {
  const destPath = path.join(MODELS_DIR, MODEL_ID, relPath);
  if (await exists(destPath)) {
    console.log(`  = ${relPath} (already present)`);
    return;
  }
  const url = `${BASE}/${relPath}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} → HTTP ${response.status}`);
  const fileBytes = Buffer.from(await response.arrayBuffer());
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  await fs.writeFile(destPath, fileBytes);
  console.log(`  ↓ ${relPath} (${megabytes(fileBytes.length)})`);
}

console.log(`[fetch_model] ${MODEL_ID} → models/`);
for (const modelFile of FILES) await download(modelFile);

console.log('[fetch_model] onnxruntime wasm → models/ort/');
const ortDest = path.join(MODELS_DIR, 'ort');
await fs.mkdir(ortDest, { recursive: true });
for (const wasmFile of ORT_WASM) {
  const srcPath = path.join(ORT_SRC, wasmFile);
  if (!(await exists(srcPath))) {
    console.warn(`  ! ${wasmFile} not in node_modules — run npm run install:web first`);
    continue;
  }
  const destPath = path.join(ortDest, wasmFile);
  if (await exists(destPath)) { console.log(`  = ${wasmFile} (already present)`); continue; }
  await fs.copyFile(srcPath, destPath);
  console.log(`  → ${wasmFile} (${megabytes((await fs.stat(destPath)).size)})`);
}

console.log('[fetch_model] pdf.js wasm decoders + standard fonts → models/pdfjs/');
const PDFJS_SRC = path.join(ROOT, 'frontend', 'node_modules', 'pdfjs-dist');
for (const assetDir of ['wasm', 'standard_fonts']) {
  const srcDir = path.join(PDFJS_SRC, assetDir);
  if (!(await exists(srcDir))) {
    console.warn(`  ! pdfjs-dist/${assetDir} not in node_modules — run npm run install:web first`);
    continue;
  }
  const destDir = path.join(MODELS_DIR, 'pdfjs', assetDir);
  await fs.mkdir(destDir, { recursive: true });
  let copiedCount = 0;
  for (const assetFile of await fs.readdir(srcDir)) {
    const destPath = path.join(destDir, assetFile);
    if (await exists(destPath)) continue;
    await fs.copyFile(path.join(srcDir, assetFile), destPath);
    copiedCount++;
  }
  console.log(`  → pdfjs/${assetDir} (${copiedCount} file(s) copied)`);
}

console.log('[fetch_model] Done. The backend serves these at /models.');
