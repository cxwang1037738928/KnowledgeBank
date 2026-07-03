/**
 * generate_categories.js
 *
 * Clusters documents by cosine similarity and writes data/categories.json.
 * The similarity threshold is passed at call time (e.g. from an API route)
 * so the frontend can control cluster granularity without restarting the server.
 *
 * Reads:
 *   data/embeddings.json  — chunk embeddings (averaged per doc for clustering)
 *
 * Writes:
 *   data/categories.json  — { threshold, categories: [{ members: [{docId, filename}] }] }
 *
 * Keyword extraction is left to heuristic.py, which runs BM-25 on cluster members.
 *
 * Run: node backend/extraction/generate_categories.js --threshold 0.75
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT            = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const EMBEDDINGS_PATH = path.join(ROOT, 'data', 'embeddings.json');
const DOCLINGS_PATH   = path.join(ROOT, 'data', 'doclings.json');
const CATEGORIES_OUT  = path.join(ROOT, 'data', 'categories.json');

// ---------------------------------------------------------------------------
// Union-Find for single-linkage clustering
// ---------------------------------------------------------------------------

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank   = new Array(n).fill(0);
  }

  find(x) {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }

  union(x, y) {
    const px = this.find(x), py = this.find(y);
    if (px === py) return;
    if      (this.rank[px] < this.rank[py]) this.parent[px] = py;
    else if (this.rank[px] > this.rank[py]) this.parent[py] = px;
    else { this.parent[py] = px; this.rank[px]++; }
  }
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

// Embeddings are L2-normalised at generation time, so dot product = cosine sim.
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// Average all chunk embeddings for a document, then re-normalise.
function buildDocVectors(chunks) {
  const accum = {};
  for (const chunk of chunks) {
    if (!chunk.embedding) continue;
    if (!accum[chunk.docId]) accum[chunk.docId] = { sum: null, count: 0 };
    const entry = accum[chunk.docId];
    if (!entry.sum) {
      entry.sum = Float64Array.from(chunk.embedding);
    } else {
      for (let i = 0; i < entry.sum.length; i++) entry.sum[i] += chunk.embedding[i];
    }
    entry.count++;
  }

  const vectors = {};
  for (const [docId, { sum, count }] of Object.entries(accum)) {
    const avg = sum.map(v => v / count);
    const mag = Math.sqrt(avg.reduce((s, v) => s + v * v, 0)) || 1;
    vectors[docId] = Array.from(avg, v => v / mag);
  }
  return vectors;
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export async function generateCategories(threshold) {
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError(`threshold must be a number in (0, 1], got: ${threshold}`);
  }

  let embedStore, doclings;

  try {
    embedStore = JSON.parse(await fs.readFile(EMBEDDINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/embeddings.json not found — run embed.js first');
  }

  try {
    doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }

  const vectors = buildDocVectors(embedStore.chunks || []);
  const docIds  = Object.keys(vectors);

  if (docIds.length === 0) {
    console.log('[generate_categories] No embedded documents to cluster.');
    const empty = { generatedAt: new Date().toISOString(), threshold, categories: [] };
    await fs.mkdir(path.dirname(CATEGORIES_OUT), { recursive: true });
    await fs.writeFile(CATEGORIES_OUT, JSON.stringify(empty, null, 2), 'utf-8');
    return empty;
  }

  // Single-linkage clustering: merge any pair whose cosine similarity >= threshold
  const uf = new UnionFind(docIds.length);
  for (let i = 0; i < docIds.length; i++) {
    for (let j = i + 1; j < docIds.length; j++) {
      if (dotProduct(vectors[docIds[i]], vectors[docIds[j]]) >= threshold) {
        uf.union(i, j);
      }
    }
  }

  // Collect members per cluster root
  const clusters = {};
  for (let i = 0; i < docIds.length; i++) {
    const root = uf.find(i);
    if (!clusters[root]) clusters[root] = [];
    clusters[root].push(docIds[i]);
  }

  const categories = Object.values(clusters).map(members => ({
    members: members.map(docId => ({
      docId,
      filename: doclings[docId]?.filename || docId,
    })),
  }));

  const output = {
    generatedAt: new Date().toISOString(),
    threshold,
    categories,
  };

  await fs.mkdir(path.dirname(CATEGORIES_OUT), { recursive: true });
  await fs.writeFile(CATEGORIES_OUT, JSON.stringify(output, null, 2), 'utf-8');

  console.log(
    `[generate_categories] ${categories.length} cluster(s) at threshold=${threshold}`,
    `(${docIds.length} docs) → ${CATEGORIES_OUT}`
  );
  return output;
}

// Run directly: node backend/extraction/generate_categories.js --threshold 0.75
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const tArg      = process.argv.indexOf('--threshold');
  const threshold = tArg !== -1
    ? parseFloat(process.argv[tArg + 1])
    : parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');

  generateCategories(threshold).catch(err => {
    console.error('[generate_categories]', err.message);
    process.exit(1);
  });
}
