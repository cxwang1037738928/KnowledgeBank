/**
 * generate_categories.js
 *
 * Clusters documents by cosine similarity of their title+abstract embeddings,
 * then calls Mistral (via Ollama) to generate a short description for each
 * cluster from its members' abstracts.
 *
 * The similarity threshold is passed at call time (e.g. from an API route)
 * so the frontend can control cluster granularity without restarting the server.
 *
 * Title+abstract vectors capture topic better than averaged full-text chunk
 * vectors: the full-text average is pulled by methodology sections, results
 * tables, and boilerplate that all papers share, whereas the abstract
 * concentrates the document's contribution in a few sentences.
 *
 * Falls back to the first 200 words of body text when both title and abstract
 * are missing (e.g. OCR-only docs where docling found no metadata).
 * If Ollama is unavailable, description is set to null and the run continues.
 *
 * Reads:  data/doclings.json
 * Writes: data/categories.json — { threshold, categories: [{ members, description }] }
 *
 * Run: node backend/extraction/generate_categories.js --threshold 0.75
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DOCLINGS_PATH  = path.join(ROOT, 'data', 'doclings.json');
const CATEGORIES_OUT = path.join(ROOT, 'data', 'categories.json');

const EMBED_MODEL          = 'Xenova/all-MiniLM-L12-v2';
const BATCH_SIZE           = 32;
const OLLAMA_URL           = process.env.OLLAMA_URL           || 'http://localhost:11434';
const CATEGORY_MODEL       = process.env.CATEGORY_MODEL       || 'ministral:3b';
const CATEGORY_ABSTRACTS_J = parseInt(process.env.CATEGORY_ABSTRACTS_J || '5', 10);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

let _extractor = null;

async function getExtractor() {
  if (!_extractor) {
    console.log(`[generate_categories] Loading ${EMBED_MODEL} ...`);
    _extractor = await pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
  }
  return _extractor;
}

async function embedBatch(texts) {
  const extractor = await getExtractor();
  const output    = await extractor(texts, { pooling: 'mean', normalize: true });
  const dims      = output.data.length / texts.length;
  return Array.from({ length: texts.length }, (_, i) =>
    Array.from(output.data.slice(i * dims, (i + 1) * dims))
  );
}

// Build the text to embed for a single document entry from doclings.json.
function metaText(entry) {
  const title    = (entry.metadata?.title    || '').trim();
  const abstract = (entry.metadata?.abstract || '').trim();
  if (title || abstract) return [title, abstract].filter(Boolean).join('\n');
  return (entry.text || '').split(/\s+/).slice(0, 200).join(' ');
}

// ---------------------------------------------------------------------------
// Category description via Mistral (Ollama)
// ---------------------------------------------------------------------------

async function describeCategoryWithOllama(abstracts) {
  if (abstracts.length === 0) return null;
  try {
    const abstractList = abstracts.map((a, i) => `[${i + 1}] ${a}`).join('\n\n');
    const prompt =
      `Given the following abstracts from academic papers in the same research cluster, ` +
      `write a concise 1–2 sentence description of what this category covers.\n\n` +
      `Abstracts:\n${abstractList}\n\n` +
      `Return ONLY the description. No explanation, no extra text.`;

    const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:   CATEGORY_MODEL,
        prompt,
        stream:  false,
        options: { temperature: 0.3 },
      }),
    });
    if (!resp.ok) return null;
    return ((await resp.json()).response || '').trim() || null;
  } catch {
    return null;
  }
}

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

// Embeddings are L2-normalised, so dot product = cosine similarity.
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export async function generateCategories(threshold) {
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError(`threshold must be a number in (0, 1], got: ${threshold}`);
  }

  let doclings;
  try {
    doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }

  const docIds = Object.keys(doclings);

  if (docIds.length === 0) {
    console.log('[generate_categories] No documents to cluster.');
    const empty = { generatedAt: new Date().toISOString(), threshold, categories: [] };
    await fs.mkdir(path.dirname(CATEGORIES_OUT), { recursive: true });
    await fs.writeFile(CATEGORIES_OUT, JSON.stringify(empty, null, 2), 'utf-8');
    return empty;
  }

  // Embed title+abstract for every document
  console.log(`[generate_categories] Embedding title+abstract for ${docIds.length} doc(s) ...`);
  const texts   = docIds.map(id => metaText(doclings[id]));
  const vectors = {};
  for (let i = 0; i < docIds.length; i += BATCH_SIZE) {
    const batchIds   = docIds.slice(i, i + BATCH_SIZE);
    const batchTexts = texts.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(batchTexts);
    embeddings.forEach((vec, j) => { vectors[batchIds[j]] = vec; });
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

  // Build categories sequentially — Ollama queues concurrent requests anyway
  console.log(`[generate_categories] Generating descriptions for ${Object.keys(clusters).length} cluster(s) via ${CATEGORY_MODEL} ...`);
  const categories = [];
  for (const members of Object.values(clusters)) {
    const abstracts = members
      .map(docId => (doclings[docId]?.metadata?.abstract || '').trim())
      .filter(Boolean)
      .slice(0, CATEGORY_ABSTRACTS_J);

    const description = await describeCategoryWithOllama(abstracts);

    categories.push({
      members: members.map(docId => ({
        docId,
        filename: doclings[docId]?.filename || docId,
      })),
      description,
    });
  }

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
