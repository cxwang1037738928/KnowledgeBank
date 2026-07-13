/**
 * corpus.js — read-only corpus data + model settings for the frontend
 *
 * Mounted at /api/corpus (server.js).
 *
 *   GET  /embedding-map — document points projected to 3D and 2D (UMAP over
 *          the title+abstract vectors persisted by generate_categories.js),
 *          plus the mutual-kNN edge list with cosine similarities. The
 *          frontend re-runs union-find over these edges as the threshold
 *          slider moves, reproducing the backend clustering exactly
 *          (same vectors, same mutual-kNN gate, same threshold semantics)
 *          with no server round-trip.
 *   GET  /graph         — data/graph.json passthrough (knowledge graph).
 *   GET  /documents     — corpus document list (id, filename, title, authors).
 *   GET  /documents/:docId/pdf — streams the original source PDF.
 *   GET  /chunks/:chunkId — one indexed chunk (text, pages, prefixLen) so the
 *          viewer can locate and highlight it in the PDF.
 *   GET  /models        — installed Ollama models + current per-role choices.
 *   POST /settings      — persist per-role model choices to .env and
 *          process.env, so pipeline spawns pick them up immediately.
 *
 * Reads DATA_DIR (default: data) like every other pipeline module — point it
 * at tests/test-output to browse test data.
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import { UMAP } from 'umap-js';

const ROOT         = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA_DIR     = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const VECTORS_PATH = path.join(DATA_DIR, 'doc_vectors.json');
const GRAPH_PATH   = path.join(DATA_DIR, 'graph.json');
const ENV_PATH     = path.join(ROOT, '.env');

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const MUTUAL_K   = parseInt(process.env.CLUSTER_MUTUAL_K || '10', 10);

// Model roles the frontend may configure. Keys are the .env variable names;
// descriptions surface in the Models tab. KG_MODEL / REASONING_MODEL are
// reserved for the LightRAG integration and may be unset.
export const MODEL_ROLES = {
  METADATA_MODEL:         'Title/author/abstract fallback when GROBID is down (extract.py)',
  EXTRACTION_MODEL:       'Other extraction tasks (extract.py)',
  QUERY_CLASSIFIER_MODEL: 'Query classification (parse_user_query.js)',
  KG_MODEL:               'Knowledge-graph construction (LightRAG, upcoming)',
  REASONING_MODEL:        'Answer synthesis over retrieved chunks (/api/chat)',
};

export const corpusRouter = express.Router();

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

// ---------------------------------------------------------------------------
// Embedding map
// ---------------------------------------------------------------------------

// Vectors are L2-normalized (generate_categories.js), so dot = cosine.
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Deterministic PRNG (mulberry32) so the UMAP layout is stable across
// requests and reloads instead of reshuffling on every fetch.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Per-axis min-max scale to [-1, 1] so the scene size is data-independent. */
function normalizeCoords(coords) {
  const dims = coords[0].length;
  for (let d = 0; d < dims; d++) {
    let min = Infinity, max = -Infinity;
    for (const c of coords) { min = Math.min(min, c[d]); max = Math.max(max, c[d]); }
    const span = max - min || 1;
    for (const c of coords) c[d] = ((c[d] - min) / span) * 2 - 1;
  }
  return coords;
}

function project(vectors, nComponents, seed) {
  const n = vectors.length;
  if (n === 1) return [nComponents === 3 ? [0, 0, 0] : [0, 0]];
  const umap = new UMAP({
    nComponents,
    // UMAP requires nNeighbors < n; 15 is the library default sweet spot.
    nNeighbors: Math.max(2, Math.min(15, n - 1)),
    minDist: 0.15,
    random: mulberry32(seed),
  });
  return normalizeCoords(umap.fit(vectors));
}

/** Mutual-kNN pairs with cosine sims — the same gate generate_categories.js
 * clusters with, so browser-side union-find over these edges reproduces the
 * backend clustering at any threshold. O(n²) sims, O(n·K) shipped. */
function mutualKnnEdges(vectors, k) {
  const n = vectors.length;
  const top = [];
  for (let i = 0; i < n; i++) {
    const sims = [];
    for (let j = 0; j < n; j++) {
      if (j !== i) sims.push([j, dot(vectors[i], vectors[j])]);
    }
    sims.sort((a, b) => b[1] - a[1]);
    top.push(new Map(sims.slice(0, k)));
  }
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (const [j, sim] of top[i]) {
      if (j > i && top[j].has(i)) edges.push({ i, j, sim: Math.round(sim * 10000) / 10000 });
    }
  }
  return edges;
}

// Projection is the expensive part — memoize on the vectors file's
// generatedAt stamp so repeat requests are free until the corpus changes.
let mapCache = null;

corpusRouter.get('/embedding-map', wrap(async (req, res) => {
  let raw;
  try {
    raw = JSON.parse(await fs.readFile(VECTORS_PATH, 'utf-8'));
  } catch {
    throw httpError(404, 'doc_vectors.json not found — run the categorize stage first');
  }
  if (!raw.docs?.length) throw httpError(404, 'doc_vectors.json is empty');

  if (mapCache?.generatedAt !== raw.generatedAt) {
    const vectors = raw.docs.map((d) => d.vector);
    const p3 = project(vectors, 3, 1337);
    const p2 = project(vectors, 2, 1337);
    mapCache = {
      generatedAt: raw.generatedAt,
      mutualK: MUTUAL_K,
      defaultThreshold: parseFloat(process.env.CLUSTER_SIMILARITY || '0.75'),
      points: raw.docs.map((d, idx) => ({
        docId:    d.docId,
        filename: d.filename,
        title:    d.title || d.filename,
        p3:       p3[idx].map((v) => Math.round(v * 1000) / 1000),
        p2:       p2[idx].map((v) => Math.round(v * 1000) / 1000),
      })),
      edges: mutualKnnEdges(vectors, MUTUAL_K),
    };
  }
  res.json(mapCache);
}));

// ---------------------------------------------------------------------------
// Knowledge graph
// ---------------------------------------------------------------------------

corpusRouter.get('/graph', wrap(async (req, res) => {
  try {
    res.json(JSON.parse(await fs.readFile(GRAPH_PATH, 'utf-8')));
  } catch {
    throw httpError(404, 'graph.json not found — run the build-graph stage first');
  }
}));

// ---------------------------------------------------------------------------
// Documents + chunks (citation deep-links)
// ---------------------------------------------------------------------------

const DOCLINGS_PATH   = path.join(DATA_DIR, 'doclings.json');
const EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.json');

// Both files are MBs; memoize each parse on its mtime.
const fileCache = new Map();
async function readJsonCached(filePath, missing) {
  const mtime = await fs.stat(filePath).then((s) => s.mtimeMs).catch(() => null);
  if (mtime === null) throw httpError(404, missing);
  const hit = fileCache.get(filePath);
  if (hit?.mtime === mtime) return hit.data;
  const data = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  fileCache.set(filePath, { mtime, data });
  return data;
}

const loadDoclings = () =>
  readJsonCached(DOCLINGS_PATH, 'doclings.json not found — run the extract stage first');

corpusRouter.get('/documents', wrap(async (req, res) => {
  const doclings = await loadDoclings();
  const docs = Object.values(doclings).map((d) => ({
    docId:    d.docId,
    filename: d.filename,
    title:    d.metadata?.title || d.filename,
    authors:  d.metadata?.authors || [],
    created:  d.metadata?.created || null,
  }));
  docs.sort((a, b) => a.title.localeCompare(b.title));
  res.json({ documents: docs });
}));

corpusRouter.get('/documents/:docId/pdf', wrap(async (req, res) => {
  const doclings = await loadDoclings();
  const entry = doclings[req.params.docId];
  if (!entry) throw httpError(404, `Unknown document "${req.params.docId}"`);
  try {
    await fs.access(entry.filePath);
  } catch {
    throw httpError(404, `Source PDF missing on disk: ${entry.filename}`);
  }
  res.type('application/pdf');
  res.sendFile(path.resolve(entry.filePath));
}));

corpusRouter.get('/chunks/:chunkId', wrap(async (req, res) => {
  const store = await readJsonCached(
    EMBEDDINGS_PATH, 'embeddings.json not found — run the embed stage first');
  const chunk = (store.chunks || []).find((c) => c.id === req.params.chunkId);
  if (!chunk) throw httpError(404, `Unknown chunk "${req.params.chunkId}"`);
  const { embedding, ...rest } = chunk;
  res.json(rest);
}));

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async function ollamaModels() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 3000);
    const resp = await fetch(`${OLLAMA_URL}/api/tags`, { signal: ctrl.signal });
    clearTimeout(timer);
    if (!resp.ok) return null;
    const body = await resp.json();
    return (body.models || []).map((m) => m.name).sort();
  } catch {
    return null; // Ollama down — the frontend falls back to free-text input
  }
}

corpusRouter.get('/models', wrap(async (req, res) => {
  const installed = await ollamaModels();
  const roles = {};
  for (const key of Object.keys(MODEL_ROLES)) roles[key] = process.env[key] || null;
  res.json({
    ollamaUp: installed !== null,
    ollamaUrl: OLLAMA_URL,
    installed: installed || [],
    roles,
    descriptions: MODEL_ROLES,
  });
}));

/**
 * Persist role→model choices. .env stays the single source of truth: existing
 * KEY= lines are replaced in place (comments and ordering preserved), missing
 * keys are appended, and process.env is updated so pipeline spawns inherit the
 * new values without a server restart. Temp-file + rename so a crash mid-write
 * can't truncate .env.
 */
corpusRouter.post('/settings', wrap(async (req, res) => {
  const updates = req.body || {};
  const keys = Object.keys(updates);
  if (keys.length === 0) throw httpError(400, 'Empty settings payload');

  for (const key of keys) {
    if (!(key in MODEL_ROLES)) throw httpError(400, `Unknown setting "${key}"`);
    const value = updates[key];
    if (typeof value !== 'string' || !value.trim() || /[\r\n]/.test(value)) {
      throw httpError(400, `Invalid value for ${key}`);
    }
  }

  let env;
  try {
    env = await fs.readFile(ENV_PATH, 'utf-8');
  } catch {
    throw httpError(500, '.env not found at project root');
  }

  const lines = env.split(/\r?\n/);
  const seen = new Set();
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^([A-Z_][A-Z0-9_]*)=/);
    if (m && keys.includes(m[1])) {
      lines[i] = `${m[1]}=${updates[m[1]].trim()}`;
      seen.add(m[1]);
    }
  }
  const missing = keys.filter((k) => !seen.has(k));
  if (missing.length) {
    if (lines[lines.length - 1]?.trim() !== '') lines.push('');
    lines.push('# Models set via the frontend Models tab');
    for (const k of missing) lines.push(`${k}=${updates[k].trim()}`);
  }

  const tmp = `${ENV_PATH}.tmp`;
  await fs.writeFile(tmp, lines.join('\n'), 'utf-8');
  await fs.rename(tmp, ENV_PATH);

  for (const k of keys) process.env[k] = updates[k].trim();

  const roles = {};
  for (const key of Object.keys(MODEL_ROLES)) roles[key] = process.env[key] || null;
  res.json({ ok: true, roles });
}));
