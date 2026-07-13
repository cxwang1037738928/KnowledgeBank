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
 * Also copies onnxruntime's wasm out of node_modules — transformers.js
 * otherwise pulls it from cdn.jsdelivr.net at runtime.
 *
 * Run: npm run fetch:model     (idempotent; skips files already present)
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT       = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MODELS_DIR = path.join(ROOT, 'models');
const MODEL_ID   = 'Xenova/all-MiniLM-L12-v2';   // must match EMBED_MODEL in Chat.jsx
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

const mb = (n) => `${(n / 1048576).toFixed(1)} MB`;

async function exists(p) {
  return fs.access(p).then(() => true).catch(() => false);
}

async function download(relPath) {
  const dest = path.join(MODELS_DIR, MODEL_ID, relPath);
  if (await exists(dest)) {
    console.log(`  = ${relPath} (already present)`);
    return;
  }
  const url = `${BASE}/${relPath}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
  const buf = Buffer.from(await resp.arrayBuffer());
  await fs.mkdir(path.dirname(dest), { recursive: true });
  await fs.writeFile(dest, buf);
  console.log(`  ↓ ${relPath} (${mb(buf.length)})`);
}

console.log(`[fetch_model] ${MODEL_ID} → models/`);
for (const f of FILES) await download(f);

console.log('[fetch_model] onnxruntime wasm → models/ort/');
const ortDest = path.join(MODELS_DIR, 'ort');
await fs.mkdir(ortDest, { recursive: true });
for (const w of ORT_WASM) {
  const src = path.join(ORT_SRC, w);
  if (!(await exists(src))) {
    console.warn(`  ! ${w} not in node_modules — run npm run install:web first`);
    continue;
  }
  const dst = path.join(ortDest, w);
  if (await exists(dst)) { console.log(`  = ${w} (already present)`); continue; }
  await fs.copyFile(src, dst);
  console.log(`  → ${w} (${mb((await fs.stat(dst)).size)})`);
}

console.log('[fetch_model] Done. The backend serves these at /models.');
