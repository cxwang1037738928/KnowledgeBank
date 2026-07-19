/**
 * server.js — OpenCrawl Express entry point
 *
 * Mounts (login required for everything but /api/auth):
 *   /api/auth        — register / login / me
 *   /api/corpus      — global model settings
 *   /api/collections — document collections + nested documents/pipeline/corpus
 *   /api/chats       — chats (each bound to a collection) + RAG chat
 *
 * Serves frontend/dist statically when it exists (npm run build:web);
 * during development run the Vite dev server instead (npm run dev:web).
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { requireAuth } from './middleware/auth.js';
import { authRouter } from './routes/auth.js';
import { chatsRouter } from './routes/chats.js';
import { collectionsRouter } from './routes/collections.js';
import { modelsRouter } from './routes/corpus.js';

const app  = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'frontend', 'dist');
const MODELS = path.join(ROOT, 'models');

// ── Middleware ────────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json());

// ── Routers ───────────────────────────────────────────────────────────────────

app.use('/api/auth', authRouter);
app.use('/api/corpus', requireAuth, modelsRouter);
app.use('/api/collections', requireAuth, collectionsRouter);
app.use('/api/chats', requireAuth, chatsRouter);

// ── Browser embedding model (npm run fetch:model) ────────────────────────────
// Chat.jsx embeds queries in-browser with the cache off, so it re-fetches the
// model every session. Serving it here instead of huggingface.co keeps chat
// working without internet access. Vite proxies /models in dev.
if (existsSync(MODELS)) {
  app.use('/models', express.static(MODELS));
} else {
  console.warn('[server] models/ not found — chat will fall back to huggingface.co. Run: npm run fetch:model');
}

// ── Frontend (production build) ──────────────────────────────────────────────

if (existsSync(DIST)) {
  app.use(express.static(DIST));
  // SPA fallback: any non-API GET serves index.html so client routes work.
  app.get(/^\/(?!api\/).*/, (req, res) => res.sendFile(path.join(DIST, 'index.html')));
}

// ── Fallthrough handlers ──────────────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({ error: `No route ${req.method} ${req.path}` });
});

// Central error handler. asyncHandler in routes calls next(err) for all
// unhandled rejections; explicit throws may carry a .status property.
app.use((err, req, res, next) => { // eslint-disable-line no-unused-vars
  const status = err.status ?? 500;
  if (status >= 500) console.error('[server] Internal error:', err);
  res.status(status).json({ error: err.message });
});

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[server] http://localhost:${PORT}`);
});

export default app;
