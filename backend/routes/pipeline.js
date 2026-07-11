/**
 * routes/pipeline.js — /api/pipeline/*
 *
 * Controls the extraction and ranking pipeline.  Each route is a thin wrapper
 * around the corresponding module function or Python subprocess; all heavy
 * logic lives in backend/extraction/ and backend/parser/.
 *
 * Stage order and dependencies:
 *   1. enhance      — rasterize + denoise + deskew + binarize (per-document)
 *   2. extract      — docling PDF → text/sections/refs/metadata  (needs: docs + enhanced/)
 *   3. embed        — chunk + MiniLM encode                      (needs: doclings.json)
 *   4. categorize   — cosine-similarity cluster + Mistral desc   (needs: doclings.json)
 *   5. heuristic    — BM25 + PageRank top-k                      (needs: doclings + categories)
 *   6. bootstrap    — Mistral synthetic queries per category      (needs: categories + embeddings)
 *   7. build-graph  — document/section/citation graph            (needs: doclings + heuristic_output)
 *
 * POST /run executes stages 2–7 in order (enhancement is per-document and
 * excluded; run it individually for each uploaded file before calling /run).
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn }  from 'child_process';
import { Router } from 'express';

import { annotateDois }       from '../extraction/doi_regex.js';
import { enrichDoclings }     from '../extraction/search_doi.js';
import { embedAll }           from '../extraction/embed.js';
import { generateCategories } from '../extraction/generate_categories.js';
import { bootstrapQueries }   from '../extraction/bootstrap_queries.js';
import { buildGraph }         from '../extraction/build_graph.js';
import { processDocument }    from '../parser/cleaning/enhance_pdf.js';
import { getDocumentStatus }  from '../parser/cleaning/clean_pdf.js';

// ── Paths ─────────────────────────────────────────────────────────────────────

const __dirname  = path.dirname(fileURLToPath(import.meta.url));
const ROOT       = path.resolve(__dirname, '..', '..');
const DATA_DIR   = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const EXTRACT_PY  = path.join(ROOT, 'backend', 'extraction', 'extract.py');
const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'heuristic.py');
const PYTHON     = process.env.PYTHON || 'python';

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Wraps an async route handler so unhandled rejections reach Express's error
 * middleware instead of crashing the process.
 */
const wrap = fn => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

/**
 * Creates an Error with an HTTP status code attached.
 * @param {number} status
 * @param {string} message
 */
function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

/**
 * Spawns a child process and resolves when it exits 0.
 * Pipes stdout/stderr to the server console in real time.
 * Rejects with an error (including stderr tail) on non-zero exit.
 *
 * @param {string}   cmd
 * @param {string[]} args
 * @returns {Promise<{ stdout: string, stderr: string }>}
 */
function spawnAsync(cmd, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: ROOT,
      env: process.env,   // DATA_DIR, PYTHON, Ollama config, etc. all pass through
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', d => { const s = d.toString(); stdout += s; process.stdout.write(s); });
    proc.stderr.on('data', d => { const s = d.toString(); stderr += s; process.stderr.write(s); });

    proc.on('close', code => {
      if (code !== 0) {
        const err = new Error(`${path.basename(cmd)} exited with code ${code}`);
        err.status = 502;
        err.detail = stderr.slice(-500);
        return reject(err);
      }
      resolve({ stdout, stderr });
    });

    proc.on('error', err => {
      err.status = 502;
      reject(err);
    });
  });
}

/**
 * Safely reads a JSON file and extracts a timestamp field.
 * Returns null if the file doesn't exist or the field is absent.
 *
 * @param {string}          filePath
 * @param {(obj: any) => string|null} getter
 */
async function fileTimestamp(filePath, getter) {
  try {
    const obj = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    return getter(obj) ?? null;
  } catch {
    return null;
  }
}

/**
 * Returns the most recent extractedAt timestamp across all entries in
 * doclings.json (the file has no top-level generatedAt).
 */
async function doclingsTimestamp(filePath) {
  try {
    const entries = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    const times = Object.values(entries).map(e => e.extractedAt).filter(Boolean);
    return times.length ? times.reduce((a, b) => (a > b ? a : b)) : null;
  } catch {
    return null;
  }
}

// ── Router ────────────────────────────────────────────────────────────────────

export const pipelineRouter = Router();

// ── GET /api/pipeline/status ─────────────────────────────────────────────────
//
// Returns the last-run timestamp for every pipeline stage by reading the
// generatedAt/createdAt/updated field from each output file.  null means the
// stage has never been run (output file absent or unreadable).
//
// Response shape:
//   {
//     doclings:   string|null,   // max extractedAt across all entries
//     embeddings: string|null,   // embeddings.json → metadata.updated
//     categories: string|null,   // categories.json → generatedAt
//     heuristic:  string|null,   // heuristic_output.json → generatedAt
//     bootstrap:  string|null,   // bootstrap_queries.json → generatedAt
//     graph:      string|null    // graph.json → createdAt
//   }

pipelineRouter.get('/status', wrap(async (req, res) => {
  const [doclings, embeddings, categories, heuristic, bootstrap, graph] = await Promise.all([
    doclingsTimestamp(path.join(DATA_DIR, 'doclings.json')),
    fileTimestamp(path.join(DATA_DIR, 'embeddings.json'),       o => o?.metadata?.updated),
    fileTimestamp(path.join(DATA_DIR, 'categories.json'),       o => o?.generatedAt),
    fileTimestamp(path.join(DATA_DIR, 'heuristic_output.json'), o => o?.generatedAt),
    fileTimestamp(path.join(DATA_DIR, 'bootstrap_queries.json'),o => o?.generatedAt),
    fileTimestamp(path.join(DATA_DIR, 'graph.json'),            o => o?.createdAt),
  ]);

  res.json({ doclings, embeddings, categories, heuristic, bootstrap, graph });
}));

// ── POST /api/pipeline/enhance ───────────────────────────────────────────────
//
// Rasterizes one document's pages at the requested DPI and runs the full
// enhancement chain: denoise → contrast-normalize → deskew → Otsu binarize.
// Writes the per-page report to data/enhanced/<docId>.json and saves each
// enhanced page as data/enhanced/<docId>/page_<N>.png.
//
// extract.py reads this report to decide between the digital and OCR
// converters — a doc where < 30% of pages are scanned uses the faster digital
// pipeline.  Run enhance before extract.
//
// Request body:
//   { docId: string, dpi?: number }     dpi defaults to 300
//
// Response (200):
//   { docId, numPages, pages: [{ pageNumber, pageType, dimensions, enhancement }] }
//
// Errors:
//   400 — missing docId
//   404 — docId not found in documents.json
//   500 — enhancement failed (rasterization or sharp error)

pipelineRouter.post('/enhance', wrap(async (req, res) => {
  const { docId, dpi = 300 } = req.body ?? {};
  if (!docId) throw httpError(400, '"docId" is required');

  const record = await getDocumentStatus(docId);
  if (!record) throw httpError(404, `Document "${docId}" not found`);

  const report = await processDocument(record.filePath, { docId, dpi });
  res.json(report);
}));

// ── POST /api/pipeline/extract ───────────────────────────────────────────────
//
// Spawns extract.py, which converts every PDF in documents.json that has not
// yet been extracted (or all of them when force=true).  Uses the enhancement
// report in data/enhanced/<docId>.json to choose the converter; falls back to
// the OCR converter if the report is absent.
//
// Writes each document's full text, markdown, sections array, tables array,
// references array, and docling metadata to data/doclings.json.
//
// Request body:
//   { force?: boolean }     re-extract already-processed docs (default false)
//
// Response (200):
//   { extracted: number, skipped: number, errors: [{ docId, filename, error }] }
//
// Errors:
//   502 — extract.py exited non-zero (Python/docling error)

pipelineRouter.post('/extract', wrap(async (req, res) => {
  const { force = false } = req.body ?? {};
  const args = [EXTRACT_PY, ...(force ? ['--force'] : [])];

  await spawnAsync(PYTHON, args);

  // Stamp metadata.doi on every extracted document (regex over the doc head)
  await annotateDois();

  // Overlay Crossref metadata for DOI'd docs (network; never fatal)
  try {
    await enrichDoclings();
  } catch (err) {
    console.warn(`[pipeline] Crossref enrichment skipped: ${err.message}`);
  }

  // extract.py writes a per-run summary — doclings.json alone can't
  // distinguish this run's work from historical entries, and failed docs
  // never appear in it (the old code here reported errors: [] always).
  const report = JSON.parse(
    await fs.readFile(path.join(DATA_DIR, 'extract_report.json'), 'utf-8'));

  res.json({
    extracted: report.extracted.length,
    skipped:   report.skipped,
    errors:    report.errors,
  });
}));

// ── POST /api/pipeline/embed ─────────────────────────────────────────────────
//
// Runs embed.js.  Reads doclings.json, splits each document's text into
// structure-aware chunks (section boundaries + heading prefixes + standalone
// table chunks), and encodes each chunk with Xenova/all-MiniLM-L12-v2
// (384-dim).  Writes the full embedding store to data/embeddings.json.
//
// Incremental by default: only documents not already in embeddings.json are
// processed.  Pass force=true to discard existing embeddings and re-encode all.
//
// Request body:
//   { force?: boolean }     re-embed all documents (default false)
//
// Response (200):
//   { chunks: number, docs: number, model: string, dimensions: number }
//
// Errors:
//   503 — doclings.json not found (run extract first)

pipelineRouter.post('/embed', wrap(async (req, res) => {
  const { force = false } = req.body ?? {};

  try {
    await embedAll({ force });
  } catch (err) {
    if (err.message.includes('doclings.json not found')) throw httpError(503, err.message);
    throw err;
  }

  const store = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'embeddings.json'), 'utf-8'));
  res.json({
    chunks:     store.chunks.length,
    docs:       store.metadata.totalDocs,
    model:      store.metadata.model,
    dimensions: store.metadata.dimensions,
  });
}));

// ── POST /api/pipeline/categorize ────────────────────────────────────────────
//
// Runs generate_categories.js.  Embeds each document's title + abstract (or
// first 200 body words when both are absent), clusters with Union-Find
// single-linkage at the given cosine similarity threshold, then calls
// Ministral-3b via Ollama to generate a 1–2 sentence description for each
// cluster from up to CATEGORY_ABSTRACTS_J member abstracts.
//
// Writes data/categories.json:
//   { threshold, generatedAt, categories: [{ members, description }] }
//
// Request body:
//   { threshold: number }   cosine similarity in (0, 1] — required
//
// Response (200):
//   { threshold: number, categories: number, docs: number }
//
// Errors:
//   400 — missing or out-of-range threshold
//   503 — doclings.json not found (run extract first)

pipelineRouter.post('/categorize', wrap(async (req, res) => {
  const { threshold } = req.body ?? {};

  if (threshold === undefined || threshold === null) {
    throw httpError(400, '"threshold" is required');
  }
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }

  let result;
  try {
    result = await generateCategories(threshold);
  } catch (err) {
    if (err.message.includes('doclings.json not found')) throw httpError(503, err.message);
    throw err;
  }

  res.json({
    threshold:  result.threshold,
    categories: result.categories.length,
    docs: result.categories.reduce((n, c) => n + c.members.length, 0),
  });
}));

// ── POST /api/pipeline/heuristic ─────────────────────────────────────────────
//
// Spawns heuristic.py.  Scores every document with BM25 top-m chunk
// representativeness + IDF novelty (blended, percentile-normalised), parses
// reference strings with Phi-4 via Ollama to build a citation graph, computes
// PageRank over that graph, and blends the two signals:
//   final = 0.25 × BM25 + 0.75 × PageRank
//
// Writes data/heuristic_output.json:
//   { k, generatedAt, topK: [...], edges: [...] }
//
// Requires Ollama to be running with the model named in CITATION_MODEL
// (default phi4).  Run categorize before heuristic so cluster keywords
// are available.
//
// Request body:
//   { k?: number }    number of top documents to select (default 2)
//
// Response (200):
//   { k, topK: [{ docId, filename, finalScore, bm25Score,
//                 bm25Representativeness, bm25Novelty, pagerankScore }],
//     edges: number }
//
// Errors:
//   502 — heuristic.py exited non-zero (Python or Ollama error)

pipelineRouter.post('/heuristic', wrap(async (req, res) => {
  const k = parseInt(req.body?.k ?? '2', 10);
  if (!Number.isInteger(k) || k < 1) throw httpError(400, '"k" must be a positive integer');

  await spawnAsync(PYTHON, [HEURISTIC_PY, '--k', String(k)]);

  const output = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'heuristic_output.json'), 'utf-8'));
  res.json({
    k:      output.k,
    topK:   output.topK,
    edges:  output.edges.length,
  });
}));

// ── POST /api/pipeline/bootstrap ─────────────────────────────────────────────
//
// Runs bootstrap_queries.js.  For each category in categories.json, assembles
// a context pack (BM25 keywords, member titles, abstracts, distinctive
// headings, medoid chunk excerpt) and prompts Ministral-3b to generate
// perCategory diverse queries covering factual, definition, comparison,
// synthesis, and methodology types.
//
// Writes data/bootstrap_queries.json:
//   { model, perCategory, generatedAt, threshold, categories: [...] }
//
// Requires Ollama with BOOTSTRAP_MODEL (default ministral:3b).
//
// Request body:
//   { perCategory?: number }    queries per cluster (default 8)
//
// Response (200):
//   { model: string, perCategory: number, totalQueries: number, categories: number }
//
// Errors:
//   503 — categories.json or doclings.json not found (run categorize first)

pipelineRouter.post('/bootstrap', wrap(async (req, res) => {
  const perCategory = parseInt(req.body?.perCategory ?? '8', 10);
  if (!Number.isInteger(perCategory) || perCategory < 1) {
    throw httpError(400, '"perCategory" must be a positive integer');
  }

  let result;
  try {
    result = await bootstrapQueries({ perCategory });
  } catch (err) {
    if (/not found/i.test(err.message)) throw httpError(503, err.message);
    throw err;
  }

  const totalQueries = result.categories.reduce((n, c) => n + c.queries.length, 0);
  res.json({
    model:        result.model,
    perCategory:  result.perCategory,
    totalQueries,
    categories:   result.categories.length,
  });
}));

// ── POST /api/pipeline/build-graph ───────────────────────────────────────────
//
// Runs build_graph.js.  Creates document nodes and section sub-nodes for the
// top-k documents from heuristic_output.json, adds citation edges from the
// connectivity analysis, and adds has_section edges.
//
// Writes data/graph.json:
//   { createdAt, topKDocIds, nodes: [...], edges: [...] }
//
// Request body:  (none)
//
// Response (200):
//   { nodes: number, edges: number, docNodes: number, sectionNodes: number,
//     citeEdges: number, sectionEdges: number }
//
// Errors:
//   503 — doclings.json or heuristic_output.json not found

pipelineRouter.post('/build-graph', wrap(async (req, res) => {
  let graph;
  try {
    graph = await buildGraph();
  } catch (err) {
    if (/not found/i.test(err.message)) throw httpError(503, err.message);
    throw err;
  }

  const docNodes     = graph.nodes.filter(n => n.type === 'document').length;
  const sectionNodes = graph.nodes.filter(n => n.type === 'section').length;
  const citeEdges    = graph.edges.filter(e => e.type === 'cites').length;
  const sectionEdges = graph.edges.filter(e => e.type === 'has_section').length;

  res.json({
    nodes:        graph.nodes.length,
    edges:        graph.edges.length,
    docNodes,
    sectionNodes,
    citeEdges,
    sectionEdges,
  });
}));

// ── POST /api/pipeline/run ───────────────────────────────────────────────────
//
// Runs the full pipeline in order: extract → embed → categorize → heuristic
// → bootstrap → build-graph.
//
// Enhancement is excluded — it is per-document and must be run individually
// via POST /enhance before uploading documents, or via the BullMQ worker.
//
// Stages are run sequentially; a failure in any stage stops the run and
// returns a partial summary showing which stages completed.
//
// Request body:
//   {
//     threshold?:   number,   cosine similarity for categorize (default 0.75)
//     k?:           number,   top-k for heuristic              (default 2)
//     force?:       boolean,  re-extract and re-embed          (default false)
//     perCategory?: number,   queries per cluster in bootstrap (default 8)
//   }
//
// Response (200):
//   {
//     stages: {
//       extract:    { ok: boolean, extracted?: number, error?: string },
//       embed:      { ok: boolean, chunks?: number, docs?: number, error?: string },
//       categorize: { ok: boolean, categories?: number, error?: string },
//       heuristic:  { ok: boolean, topK?: number, edges?: number, error?: string },
//       bootstrap:  { ok: boolean, totalQueries?: number, error?: string },
//       graph:      { ok: boolean, nodes?: number, edges?: number, error?: string }
//     }
//   }

pipelineRouter.post('/run', wrap(async (req, res) => {
  const {
    // Same default the tests use: CLUSTER_SIMILARITY from .env is the single
    // source of truth, so API runs and test runs cluster identically.
    threshold   = parseFloat(process.env.CLUSTER_SIMILARITY || '0.75'),
    k           = 2,
    force       = false,
    perCategory = 8,
  } = req.body ?? {};

  if (typeof threshold !== 'number' || threshold <= 0 || threshold > 1) {
    throw httpError(400, '"threshold" must be a number in (0, 1]');
  }

  const stages = {};

  // 1 — Extract
  try {
    const args = [EXTRACT_PY, ...(force ? ['--force'] : [])];
    await spawnAsync(PYTHON, args);
    await annotateDois();
    try {
      await enrichDoclings();
    } catch (err) {
      console.warn(`[pipeline] Crossref enrichment skipped: ${err.message}`);
    }
    const report = JSON.parse(
      await fs.readFile(path.join(DATA_DIR, 'extract_report.json'), 'utf-8'));
    // ok = the stage ran; per-document failures are reported in errors[]
    // without aborting the run (the successful docs still flow downstream).
    stages.extract = {
      ok:        true,
      extracted: report.extracted.length,
      skipped:   report.skipped,
      errors:    report.errors,
    };
  } catch (err) {
    stages.extract = { ok: false, error: err.message };
    return res.json({ stages });
  }

  // 2 — Embed
  try {
    await embedAll({ force });
    const store = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'embeddings.json'), 'utf-8'));
    stages.embed = { ok: true, chunks: store.chunks.length, docs: store.metadata.totalDocs };
  } catch (err) {
    stages.embed = { ok: false, error: err.message };
    return res.json({ stages });
  }

  // 3 — Categorize
  try {
    const result = await generateCategories(threshold);
    stages.categorize = { ok: true, categories: result.categories.length };
  } catch (err) {
    stages.categorize = { ok: false, error: err.message };
    return res.json({ stages });
  }

  // 4 — Heuristic
  try {
    await spawnAsync(PYTHON, [HEURISTIC_PY, '--k', String(k)]);
    const output = JSON.parse(await fs.readFile(path.join(DATA_DIR, 'heuristic_output.json'), 'utf-8'));
    stages.heuristic = { ok: true, topK: output.topK.length, edges: output.edges.length };
  } catch (err) {
    stages.heuristic = { ok: false, error: err.message };
    return res.json({ stages });
  }

  // 5 — Bootstrap
  try {
    const result = await bootstrapQueries({ perCategory });
    stages.bootstrap = {
      ok: true,
      totalQueries: result.categories.reduce((n, c) => n + c.queries.length, 0),
    };
  } catch (err) {
    stages.bootstrap = { ok: false, error: err.message };
    return res.json({ stages });
  }

  // 6 — Build graph
  try {
    const graph = await buildGraph();
    stages.graph = { ok: true, nodes: graph.nodes.length, edges: graph.edges.length };
  } catch (err) {
    stages.graph = { ok: false, error: err.message };
    return res.json({ stages });
  }

  res.json({ stages });
}));
