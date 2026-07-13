/**
 * routes/pipeline.js — /api/pipeline/*
 *
 * Controls the extraction and ranking pipeline. Every stage is a runner in
 * STAGES keyed by name; the per-stage POST routes validate params and call
 * their runner, and POST /run iterates the same runners in STAGE_ORDER.
 * Swapping a stage implementation = replacing one runner.
 *
 * Stage order and dependencies:
 *   1. enhance      — rasterize + denoise + deskew + binarize (per-document,
 *                     excluded from /run)
 *   2. extract      — docling PDF → text/sections/refs/metadata
 *   3. embed        — chunk + MiniLM encode
 *   4. categorize   — cosine-similarity cluster + keyword/medoid index
 *   5. heuristic    — BM25 + PageRank top-k
 *   6. graph        — document/section/citation graph
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn }  from 'child_process';
import { Router } from 'express';

import { annotateDois }       from '../extraction/sapphire/doi_regex.js';
import { enrichDoclings }     from '../extraction/sapphire/search_doi.js';
import { embedAll }           from '../extraction/embed.js';
import { generateCategories } from '../extraction/generate_categories.js';
import { buildGraph }         from '../extraction/sapphire/build_graph.js';
import { processDocument }    from '../parser/cleaning/enhance_pdf.js';
import { getDocumentStatus }  from '../parser/cleaning/clean_pdf.js';

// ── Paths ─────────────────────────────────────────────────────────────────────

const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const ROOT       = path.resolve(__dirname, '..', '..');
const DATA_DIR   = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const EXTRACT_PY   = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'extract.py');
const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'sapphire', 'heuristic.py');
const PYTHON     = process.env.PYTHON || 'python';

// ── Helpers ───────────────────────────────────────────────────────────────────

const wrap = fn => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

/** Spawn a child process; resolve on exit 0, reject (status 502) otherwise. */
function spawnAsync(cmd, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: ROOT,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';
    proc.stdout.on('data', d => process.stdout.write(d.toString()));
    proc.stderr.on('data', d => { const s = d.toString(); stderr += s; process.stderr.write(s); });

    proc.on('close', code => {
      if (code !== 0) {
        const err = new Error(`${path.basename(cmd)} exited with code ${code}`);
        err.status = 502;
        err.detail = stderr.slice(-500);
        return reject(err);
      }
      resolve();
    });

    proc.on('error', err => {
      err.status = 502;
      reject(err);
    });
  });
}

async function readData(filename) {
  return JSON.parse(await fs.readFile(path.join(DATA_DIR, filename), 'utf-8'));
}

/** Missing-input errors become 503 (run the earlier stage first). */
function toHttp(err) {
  if (!err.status && /not found/i.test(err.message)) err.status = 503;
  return err;
}

// ── Stage runners ─────────────────────────────────────────────────────────────
//
// Each runner takes validated params and returns the JSON summary the route
// responds with. /run reuses them verbatim.

export const STAGE_ORDER = ['extract', 'embed', 'categorize', 'heuristic', 'graph'];

export const STAGES = {
  async extract({ force = false } = {}) {
    await spawnAsync(PYTHON, [EXTRACT_PY, ...(force ? ['--force'] : [])]);
    await annotateDois();
    try {
      await enrichDoclings();               // network enrichment — never fatal
    } catch (err) {
      console.warn(`[pipeline] Crossref enrichment skipped: ${err.message}`);
    }
    const report = await readData('extract_report.json');
    return { extracted: report.extracted.length, skipped: report.skipped, errors: report.errors };
  },

  async embed({ force = false } = {}) {
    await embedAll({ force });
    const store = await readData('embeddings.json');
    return {
      chunks:     store.chunks.length,
      docs:       store.metadata.totalDocs,
      model:      store.metadata.model,
      dimensions: store.metadata.dimensions,
    };
  },

  async categorize({ threshold } = {}) {
    const result = await generateCategories(threshold);
    return {
      threshold:  result.threshold,
      categories: result.categories.length,
      docs:       result.categories.reduce((n, c) => n + c.members.length, 0),
    };
  },

  async heuristic({ k = 2 } = {}) {
    await spawnAsync(PYTHON, [HEURISTIC_PY, '--k', String(k)]);
    const output = await readData('heuristic_output.json');
    return { k: output.k, topK: output.topK, edges: output.edges.length };
  },

  async graph() {
    const g = await buildGraph();
    return {
      nodes:        g.nodes.length,
      edges:        g.edges.length,
      docNodes:     g.nodes.filter(n => n.type === 'document').length,
      sectionNodes: g.nodes.filter(n => n.type === 'section').length,
      citeEdges:    g.edges.filter(e => e.type === 'cites').length,
      sectionEdges: g.edges.filter(e => e.type === 'has_section').length,
    };
  },
};

// ── Router ────────────────────────────────────────────────────────────────────

export const pipelineRouter = Router();

// GET /status — last-run timestamp per stage (null = never ran).
pipelineRouter.get('/status', wrap(async (req, res) => {
  const timestamp = async (filename, getter) => {
    try {
      return getter(await readData(filename)) ?? null;
    } catch {
      return null;
    }
  };

  const [doclings, embeddings, categories, heuristic, graph] = await Promise.all([
    timestamp('doclings.json', o => {
      const times = Object.values(o).map(e => e.extractedAt).filter(Boolean);
      return times.length ? times.reduce((a, b) => (a > b ? a : b)) : null;
    }),
    timestamp('embeddings.json',       o => o?.metadata?.updated),
    timestamp('categories.json',       o => o?.generatedAt),
    timestamp('heuristic_output.json', o => o?.generatedAt),
    timestamp('graph.json',            o => o?.createdAt),
  ]);

  res.json({ doclings, embeddings, categories, heuristic, graph });
}));

// POST /enhance { docId, dpi? } — per-document page enhancement; run before
// extract so extract.py can route scanned pages to the OCR converter.
pipelineRouter.post('/enhance', wrap(async (req, res) => {
  const { docId, dpi = 300 } = req.body ?? {};
  if (!docId) throw httpError(400, '"docId" is required');

  const record = await getDocumentStatus(docId);
  if (!record) throw httpError(404, `Document "${docId}" not found`);

  res.json(await processDocument(record.filePath, { docId, dpi }));
}));

// POST /extract { force? }
pipelineRouter.post('/extract', wrap(async (req, res) => {
  res.json(await STAGES.extract({ force: req.body?.force === true }));
}));

// POST /embed { force? }
pipelineRouter.post('/embed', wrap(async (req, res) => {
  res.json(await STAGES.embed({ force: req.body?.force === true }).catch(err => { throw toHttp(err); }));
}));

// POST /categorize { threshold } — cosine similarity in (0, 1], required.
pipelineRouter.post('/categorize', wrap(async (req, res) => {
  const { threshold } = req.body ?? {};
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }
  res.json(await STAGES.categorize({ threshold }).catch(err => { throw toHttp(err); }));
}));

// POST /heuristic { k? }
pipelineRouter.post('/heuristic', wrap(async (req, res) => {
  const k = parseInt(req.body?.k ?? '2', 10);
  if (!Number.isInteger(k) || k < 1) throw httpError(400, '"k" must be a positive integer');
  res.json(await STAGES.heuristic({ k }));
}));

// POST /build-graph
pipelineRouter.post('/build-graph', wrap(async (req, res) => {
  res.json(await STAGES.graph().catch(err => { throw toHttp(err); }));
}));

// POST /run { threshold?, k?, force? } — stages 2–6 in order; a failing stage
// stops the run and the response shows how far it got.
pipelineRouter.post('/run', wrap(async (req, res) => {
  const {
    threshold = parseFloat(process.env.CLUSTER_SIMILARITY || '0.75'),
    k         = 2,
    force     = false,
  } = req.body ?? {};

  if (typeof threshold !== 'number' || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }

  const params = {
    extract:    { force },
    embed:      { force },
    categorize: { threshold },
    heuristic:  { k },
    graph:      {},
  };

  const stages = {};
  for (const name of STAGE_ORDER) {
    try {
      stages[name] = { ok: true, ...(await STAGES[name](params[name])) };
    } catch (err) {
      stages[name] = { ok: false, error: err.message };
      break;
    }
  }

  res.json({ stages });
}));
