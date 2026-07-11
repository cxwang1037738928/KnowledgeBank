/**
 * generate_categories.js
 *
 * Clusters documents by cosine similarity of their title+abstract embeddings,
 * then indexes each cluster with a compact, deterministic summary — no LLM.
 *
 * Title+abstract vectors capture topic better than averaged full-text chunk
 * vectors: the full-text average is pulled by methodology sections, results
 * tables, and boilerplate that all papers share, whereas the abstract
 * concentrates the document's contribution in a few sentences.
 *
 * Falls back to the first 200 words of body text when both title and abstract
 * are missing (e.g. OCR-only docs where docling found no metadata).
 *
 * Each category records:
 *   index    — incremental cluster number (0, 1, 2, …)
 *   members  — [{ docId, filename }]
 *   keywords — top TF-IDF terms over the cluster's body text, computed with
 *              tokenise/REF_HEADINGS/normHeading from regex_utils.js (IDF is
 *              the same corpus statistic heuristic.py uses)
 *   medoid   — { docId, filename }: the member whose title+abstract vector is
 *              closest to the cluster centroid (the "file closest to the
 *              category centroid"), reusing the vectors already embedded here
 *
 * Reads:  data/doclings.json
 * Writes: data/categories.json — { threshold, generatedAt, categories: [...] }
 *
 * Run: node backend/extraction/generate_categories.js --threshold 0.75
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';
import { tokenise, REF_HEADINGS, normHeading } from './regex_utils.js';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA_DIR       = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const DOCLINGS_PATH  = path.join(DATA_DIR, 'doclings.json');
const CATEGORIES_OUT = path.join(DATA_DIR, 'categories.json');

const EMBED_MODEL = 'Xenova/all-MiniLM-L12-v2';
const BATCH_SIZE  = 32;
const KEYWORDS_N  = parseInt(process.env.KEYWORDS_N || '20', 10);

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
// TF-IDF cluster keywords. heuristic.py's cluster_keywords scores by summed
// IDF alone, which works for multi-doc clusters (shared distinctive terms
// accumulate) but degenerates for single-doc clusters — every corpus-rare term
// ties at max IDF, so the tiebreak surfaces noise. Weighting IDF by in-cluster
// term frequency (what BM25 fundamentally does) surfaces readable keywords in
// both cases: a distinctive term repeated within the cluster outranks a rare
// term seen once. IDF is the same corpus statistic used by heuristic.py.
// ---------------------------------------------------------------------------

function bodyText(entry) {
  // normHeading strips leading numbering ('7. References') so numbered
  // bibliography headings don't leak reference text into the keyword pool.
  const sections = (entry?.sections || []).filter(
    (s) => !REF_HEADINGS.has(normHeading(s.heading)) && (s.text || '').trim()
  );
  return sections.length ? sections.map((s) => s.text).join(' ') : (entry?.text || '');
}

function buildIdf(tokenisedDocs) {
  const N = tokenisedDocs.length;
  const df = new Map();
  for (const tokens of tokenisedDocs) {
    for (const t of new Set(tokens)) df.set(t, (df.get(t) || 0) + 1);
  }
  return (term) => {
    const d = df.get(term) || 0;
    return Math.log((N - d + 0.5) / (d + 0.5) + 1);
  };
}

function clusterKeywords(memberTokens, idf, n = KEYWORDS_N) {
  const tf = new Map();
  for (const tokens of memberTokens) {
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
  }
  const scores = new Map();
  for (const [t, count] of tf) scores.set(t, count * idf(t));
  // Alphabetical secondary sort key keeps ties deterministic across runs.
  return [...scores.entries()]
    .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
    .slice(0, n)
    .map(([t]) => t);
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

/**
 * The "file closest to the category centroid": the member whose title+abstract
 * vector has the highest cosine similarity to the (L2-normalised) mean of the
 * cluster's member vectors. Reuses the vectors already embedded for clustering
 * — no chunk embeddings or extra file loads needed. Single-member clusters
 * return that lone member.
 */
function clusterMedoid(memberIds, vectors) {
  if (memberIds.length === 1) return memberIds[0];

  const dim = vectors[memberIds[0]].length;
  const centroid = new Float64Array(dim);
  for (const id of memberIds) {
    const v = vectors[id];
    for (let i = 0; i < dim; i++) centroid[i] += v[i];
  }
  let mag = 0;
  for (let i = 0; i < dim; i++) mag += centroid[i] * centroid[i];
  mag = Math.sqrt(mag) || 1;
  for (let i = 0; i < dim; i++) centroid[i] /= mag;

  let best = memberIds[0], bestSim = -Infinity;
  for (const id of memberIds) {
    const sim = dotProduct(vectors[id], centroid);
    if (sim > bestSim) { bestSim = sim; best = id; }
  }
  return best;
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
  const texts = docIds.map(id => {
    const entry    = doclings[id];
    const title    = (entry.metadata?.title    || '').trim();
    const abstract = (entry.metadata?.abstract || '').trim();
    if (!title && !abstract) {
      console.warn(`[generate_categories]   no title/abstract: ${entry.filename || id} — falling back to first 200 words of body text`);
    }
    return metaText(entry);
  });
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

  // Corpus-wide IDF for cluster keywords (body text, reference sections dropped)
  const tokenised = new Map(docIds.map((d) => [d, tokenise(bodyText(doclings[d]))]));
  const idf = buildIdf([...tokenised.values()]);

  // Index each cluster with keywords + centroid medoid — no LLM
  const categories = Object.values(clusters).map((members, index) => {
    const medoidId = clusterMedoid(members, vectors);
    return {
      index,
      members: members.map(docId => ({
        docId,
        filename: doclings[docId]?.filename || docId,
      })),
      keywords: clusterKeywords(members.map(d => tokenised.get(d) || []), idf),
      medoid: {
        docId:    medoidId,
        filename: doclings[medoidId]?.filename || medoidId,
      },
    };
  });

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
