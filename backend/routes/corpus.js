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
function dot(vecA, vecB) {
  let sum = 0;
  for (let componentIdx = 0; componentIdx < vecA.length; componentIdx++) {
    sum += vecA[componentIdx] * vecB[componentIdx];
  }
  return sum;
}

// Deterministic PRNG (mulberry32) so the UMAP layout is stable across
// requests and reloads instead of reshuffling on every fetch.
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state |= 0; state = (state + 0x6D2B79F5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/** Per-axis min-max scale to [-1, 1] so the scene size is data-independent. */
function normalizeCoords(coords) {
  const dims = coords[0].length;
  for (let axis = 0; axis < dims; axis++) {
    let min = Infinity, max = -Infinity;
    for (const point of coords) {
      min = Math.min(min, point[axis]);
      max = Math.max(max, point[axis]);
    }
    const span = max - min || 1;
    for (const point of coords) point[axis] = ((point[axis] - min) / span) * 2 - 1;
  }
  return coords;
}

function project(vectors, nComponents, seed) {
  const docCount = vectors.length;
  if (docCount === 1) return [nComponents === 3 ? [0, 0, 0] : [0, 0]];
  const umap = new UMAP({
    nComponents,
    // UMAP requires nNeighbors < docCount; 15 is the library default sweet spot.
    nNeighbors: Math.max(2, Math.min(15, docCount - 1)),
    minDist: 0.15,
    random: mulberry32(seed),
  });
  return normalizeCoords(umap.fit(vectors));
}

/** Mutual-kNN pairs with cosine sims — the same gate generate_categories.js
 * clusters with, so browser-side union-find over these edges reproduces the
 * backend clustering at any threshold. O(n²) sims, O(n·K) shipped. */
function mutualKnnEdges(vectors, k) {
  const docCount = vectors.length;
  const nearestNeighbours = [];
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    const sims = [];
    for (let otherIdx = 0; otherIdx < docCount; otherIdx++) {
      if (otherIdx !== docIdx) sims.push([otherIdx, dot(vectors[docIdx], vectors[otherIdx])]);
    }
    sims.sort((simA, simB) => simB[1] - simA[1]);
    nearestNeighbours.push(new Map(sims.slice(0, k)));
  }
  const edges = [];
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    for (const [otherIdx, sim] of nearestNeighbours[docIdx]) {
      if (otherIdx > docIdx && nearestNeighbours[otherIdx].has(docIdx)) {
        edges.push({ i: docIdx, j: otherIdx, sim: Math.round(sim * 10000) / 10000 });
      }
    }
  }
  return edges;
}

// Projection is the expensive part — memoize on the vectors file's
// generatedAt stamp so repeat requests are free until the corpus changes.
let mapCache = null;

corpusRouter.get('/embedding-map', wrap(async (req, res) => {
  let docVectors;
  try {
    docVectors = JSON.parse(await fs.readFile(VECTORS_PATH, 'utf-8'));
  } catch {
    throw httpError(404, 'doc_vectors.json not found — run the categorize stage first');
  }
  if (!docVectors.docs?.length) throw httpError(404, 'doc_vectors.json is empty');

  if (mapCache?.generatedAt !== docVectors.generatedAt) {
    const vectors = docVectors.docs.map((doc) => doc.vector);
    const p3 = project(vectors, 3, 1337);
    const p2 = project(vectors, 2, 1337);
    mapCache = {
      generatedAt: docVectors.generatedAt,
      mutualK: MUTUAL_K,
      defaultThreshold: parseFloat(process.env.CLUSTER_SIMILARITY || '0.75'),
      points: docVectors.docs.map((doc, docIdx) => ({
        docId:    doc.docId,
        filename: doc.filename,
        title:    doc.title || doc.filename,
        p3:       p3[docIdx].map((coord) => Math.round(coord * 1000) / 1000),
        p2:       p2[docIdx].map((coord) => Math.round(coord * 1000) / 1000),
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
  const mtime = await fs.stat(filePath).then((stats) => stats.mtimeMs).catch(() => null);
  if (mtime === null) throw httpError(404, missing);
  const cached = fileCache.get(filePath);
  if (cached?.mtime === mtime) return cached.data;
  const data = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  fileCache.set(filePath, { mtime, data });
  return data;
}

const loadDoclings = () =>
  readJsonCached(DOCLINGS_PATH, 'doclings.json not found — run the extract stage first');

corpusRouter.get('/documents', wrap(async (req, res) => {
  const doclings = await loadDoclings();
  const docs = Object.values(doclings).map((doclingEntry) => ({
    docId:    doclingEntry.docId,
    filename: doclingEntry.filename,
    title:    doclingEntry.metadata?.title || doclingEntry.filename,
    authors:  doclingEntry.metadata?.authors || [],
    created:  doclingEntry.metadata?.created || null,
  }));
  docs.sort((docA, docB) => docA.title.localeCompare(docB.title));
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
  const embeddingStore = await readJsonCached(
    EMBEDDINGS_PATH, 'embeddings.json not found — run the embed stage first');
  const chunk = (embeddingStore.chunks || []).find(
    (storedChunk) => storedChunk.id === req.params.chunkId);
  if (!chunk) throw httpError(404, `Unknown chunk "${req.params.chunkId}"`);
  const { embedding, ...chunkWithoutEmbedding } = chunk;
  res.json(chunkWithoutEmbedding);
}));

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async function ollamaModels() {
  try {
    const abortController = new AbortController();
    const timeoutTimer = setTimeout(() => abortController.abort(), 3000);
    const response = await fetch(`${OLLAMA_URL}/api/tags`, { signal: abortController.signal });
    clearTimeout(timeoutTimer);
    if (!response.ok) return null;
    const body = await response.json();
    return (body.models || []).map((model) => model.name).sort();
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
  const roleKeys = Object.keys(updates);
  if (roleKeys.length === 0) throw httpError(400, 'Empty settings payload');

  for (const roleKey of roleKeys) {
    if (!(roleKey in MODEL_ROLES)) throw httpError(400, `Unknown setting "${roleKey}"`);
    const modelName = updates[roleKey];
    if (typeof modelName !== 'string' || !modelName.trim() || /[\r\n]/.test(modelName)) {
      throw httpError(400, `Invalid value for ${roleKey}`);
    }
  }

  let envContents;
  try {
    envContents = await fs.readFile(ENV_PATH, 'utf-8');
  } catch {
    throw httpError(500, '.env not found at project root');
  }

  const envLines = envContents.split(/\r?\n/);
  const replacedKeys = new Set();
  for (let lineIdx = 0; lineIdx < envLines.length; lineIdx++) {
    const keyMatch = envLines[lineIdx].match(/^([A-Z_][A-Z0-9_]*)=/);
    if (keyMatch && roleKeys.includes(keyMatch[1])) {
      envLines[lineIdx] = `${keyMatch[1]}=${updates[keyMatch[1]].trim()}`;
      replacedKeys.add(keyMatch[1]);
    }
  }
  const appendKeys = roleKeys.filter((roleKey) => !replacedKeys.has(roleKey));
  if (appendKeys.length) {
    if (envLines[envLines.length - 1]?.trim() !== '') envLines.push('');
    envLines.push('# Models set via the frontend Models tab');
    for (const roleKey of appendKeys) envLines.push(`${roleKey}=${updates[roleKey].trim()}`);
  }

  const tempPath = `${ENV_PATH}.tmp`;
  await fs.writeFile(tempPath, envLines.join('\n'), 'utf-8');
  await fs.rename(tempPath, ENV_PATH);

  for (const roleKey of roleKeys) process.env[roleKey] = updates[roleKey].trim();

  const roles = {};
  for (const key of Object.keys(MODEL_ROLES)) roles[key] = process.env[key] || null;
  res.json({ ok: true, roles });
}));
