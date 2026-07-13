/**
 * server.js — OpenCrawl Express entry point
 *
 * Mounts:
 *   /api/pipeline  — pipeline control (enhance, extract, embed, categorize,
 *                    heuristic, build-graph, run, status)
 *   /api/corpus    — read-only data access for the frontend (embedding map,
 *                    knowledge graph) + model settings
 *
 * Serves frontend/dist statically when it exists (npm run build:web);
 * during development run the Vite dev server instead (npm run dev:web).
 *
 * Routers to add later:
 *   /api/documents — upload, list, status, delete  (clean_pdf.js)
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { pipelineRouter } from './routes/pipeline.js';
import { corpusRouter } from './routes/corpus.js';

const app  = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const DIST = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'frontend', 'dist');

// ── Middleware ────────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json());

// ── Routers ───────────────────────────────────────────────────────────────────

app.use('/api/pipeline', pipelineRouter);
app.use('/api/corpus', corpusRouter);

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
