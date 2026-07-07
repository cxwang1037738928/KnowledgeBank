/**
 * server.js — BookWyrm Express entry point
 *
 * Mounts:
 *   /api/pipeline  — pipeline control (enhance, extract, embed, categorize,
 *                    heuristic, bootstrap, build-graph, run, status)
 *
 * Routers to add later:
 *   /api/documents — upload, list, status, delete  (clean_pdf.js)
 *   /api/query     — classify + retrieve / reason   (retriever/)
 *   /api/corpus    — read-only data access          (embeddings, graph, scores)
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { pipelineRouter } from './routes/pipeline.js';

const app  = express();
const PORT = parseInt(process.env.PORT || '3000', 10);

// ── Middleware ────────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json());

// ── Routers ───────────────────────────────────────────────────────────────────

app.use('/api/pipeline', pipelineRouter);

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
