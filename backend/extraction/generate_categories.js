/**
 * generate_categories.js
 *
 * Clusters documents by cosine similarity of their title+abstract embeddings
 * (single-linkage gated by mutual-kNN — see CLUSTER_MUTUAL_K), then indexes
 * each cluster with a compact, deterministic summary — no LLM.
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
 * Writes: data/categories.json  — { threshold, generatedAt, categories: [...] }
 *         data/doc_vectors.json — { generatedAt, model, dims, docs: [{docId,
 *           filename, title, vector}] }: the title+abstract vectors this run
 *           clustered with, persisted so the frontend embedding-space view
 *           (routes/corpus.js) can project and re-cluster them at any
 *           threshold without re-embedding.
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

// Same corpus model embed.js uses — these vectors and the chunk vectors are
// compared against the same browser-side query vectors.
const EMBED_MODEL = process.env.SAPPHIRE_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L12-v2';
const BATCH_SIZE  = 32;
const KEYWORDS_N  = parseInt(process.env.KEYWORDS_N || '20', 10);
// Mutual-kNN gate: a pair may merge only if each doc is in the other's top
// MUTUAL_K nearest neighbours (plus the similarity threshold). Guards
// single-linkage against transitive chaining on large corpora.
const MUTUAL_K    = parseInt(process.env.CLUSTER_MUTUAL_K || '10', 10);

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
  return Array.from({ length: texts.length }, (_, textIdx) =>
    Array.from(output.data.slice(textIdx * dims, (textIdx + 1) * dims))
  );
}

// Text to embed for one doclings.json entry: title+abstract, else the first
// 200 body words.
function metaText(entry) {
  const title    = (entry.metadata?.title    || '').trim();
  const abstract = (entry.metadata?.abstract || '').trim();
  if (title || abstract) return [title, abstract].filter(Boolean).join('\n');
  console.warn(`[generate_categories]   no title/abstract: ${entry.filename} — falling back to first 200 words of body text`);
  return (entry.text || '').split(/\s+/).slice(0, 200).join(' ');
}

// ---------------------------------------------------------------------------
// TF-IDF cluster keywords — the single source of keyword truth (heuristic.py
// reads them from categories.json rather than recomputing). TF weighting
// matters: summed IDF alone degenerates for single-doc clusters, where every
// corpus-rare term ties at max IDF.
// ---------------------------------------------------------------------------

function bodyText(entry) {
  // normHeading strips numbering ('7. References') so numbered bibliography
  // headings don't leak reference text into the keyword pool.
  const sections = (entry?.sections || []).filter(
    (section) => !REF_HEADINGS.has(normHeading(section.heading)) && (section.text || '').trim()
  );
  return sections.length ? sections.map((section) => section.text).join(' ') : (entry?.text || '');
}

function buildIdf(tokenisedDocs) {
  const docCount = tokenisedDocs.length;
  const docFreq = new Map();
  for (const tokens of tokenisedDocs) {
    for (const token of new Set(tokens)) docFreq.set(token, (docFreq.get(token) || 0) + 1);
  }
  return (term) => {
    const termDocFreq = docFreq.get(term) || 0;
    return Math.log((docCount - termDocFreq + 0.5) / (termDocFreq + 0.5) + 1);
  };
}

function clusterKeywords(memberTokens, idf, keywordCount = KEYWORDS_N) {
  const termFreq = new Map();
  for (const tokens of memberTokens) {
    for (const token of tokens) termFreq.set(token, (termFreq.get(token) || 0) + 1);
  }
  const scores = new Map();
  for (const [term, count] of termFreq) scores.set(term, count * idf(term));
  // Alphabetical secondary sort key keeps ties deterministic across runs.
  return [...scores.entries()]
    .sort((entryA, entryB) => (entryB[1] - entryA[1]) || entryA[0].localeCompare(entryB[0]))
    .slice(0, keywordCount)
    .map(([term]) => term);
}

// ---------------------------------------------------------------------------
// Union-Find for single-linkage clustering
// ---------------------------------------------------------------------------

class UnionFind {
  constructor(size) {
    this.parent = Array.from({ length: size }, (_, nodeIdx) => nodeIdx);
    this.rank   = new Array(size).fill(0);
  }

  find(node) {
    if (this.parent[node] !== node) this.parent[node] = this.find(this.parent[node]);
    return this.parent[node];
  }

  union(nodeA, nodeB) {
    const rootA = this.find(nodeA), rootB = this.find(nodeB);
    if (rootA === rootB) return;
    if      (this.rank[rootA] < this.rank[rootB]) this.parent[rootA] = rootB;
    else if (this.rank[rootA] > this.rank[rootB]) this.parent[rootB] = rootA;
    else { this.parent[rootB] = rootA; this.rank[rootA]++; }
  }
}

// Embeddings are L2-normalised, so dot product = cosine similarity.
function dotProduct(vecA, vecB) {
  let sum = 0;
  for (let componentIdx = 0; componentIdx < vecA.length; componentIdx++) {
    sum += vecA[componentIdx] * vecB[componentIdx];
  }
  return sum;
}

/** Member whose vector is closest to the L2-normalised cluster centroid. */
function clusterMedoid(memberIds, vectors) {
  if (memberIds.length === 1) return memberIds[0];

  const dims = vectors[memberIds[0]].length;
  const centroid = new Float64Array(dims);
  for (const memberId of memberIds) {
    const memberVector = vectors[memberId];
    for (let dim = 0; dim < dims; dim++) centroid[dim] += memberVector[dim];
  }
  let magnitude = 0;
  for (let dim = 0; dim < dims; dim++) magnitude += centroid[dim] * centroid[dim];
  magnitude = Math.sqrt(magnitude) || 1;
  for (let dim = 0; dim < dims; dim++) centroid[dim] /= magnitude;

  let closestId = memberIds[0], bestSim = -Infinity;
  for (const memberId of memberIds) {
    const sim = dotProduct(vectors[memberId], centroid);
    if (sim > bestSim) { bestSim = sim; closestId = memberId; }
  }
  return closestId;
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export async function generateCategories(threshold, dataDir = DATA_DIR) {
  if (typeof threshold !== 'number' || isNaN(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError(`threshold must be a number in (0, 1], got: ${threshold}`);
  }
  const doclingsPath  = path.join(dataDir, 'doclings.json');
  const categoriesOut = path.join(dataDir, 'categories.json');
  const vectorsOut    = path.join(dataDir, 'doc_vectors.json');

  let doclings;
  try {
    doclings = JSON.parse(await fs.readFile(doclingsPath, 'utf-8'));
  } catch {
    throw new Error(`${doclingsPath} not found — run extract.py first`);
  }

  const docIds = Object.keys(doclings);

  if (docIds.length === 0) {
    console.log('[generate_categories] No documents to cluster.');
    const empty = { generatedAt: new Date().toISOString(), threshold, categories: [] };
    await fs.mkdir(path.dirname(categoriesOut), { recursive: true });
    await fs.writeFile(categoriesOut, JSON.stringify(empty, null, 2), 'utf-8');
    return empty;
  }

  console.log(`[generate_categories] Embedding title+abstract for ${docIds.length} doc(s) ...`);
  const texts = docIds.map(docId => metaText(doclings[docId]));
  const vectors = {};
  for (let batchStart = 0; batchStart < docIds.length; batchStart += BATCH_SIZE) {
    const batchIds   = docIds.slice(batchStart, batchStart + BATCH_SIZE);
    const batchTexts = texts.slice(batchStart, batchStart + BATCH_SIZE);
    const embeddings = await embedBatch(batchTexts);
    embeddings.forEach((vector, idxInBatch) => { vectors[batchIds[idxInBatch]] = vector; });
  }

  // Single-linkage gated by mutual-kNN: merge only when sim >= threshold AND
  // each doc is in the other's top MUTUAL_K neighbours — plain single-linkage
  // transitively chains unrelated docs into one blob at corpus scale.
  const docCount = docIds.length;
  const neighbours = []; // per doc: Map(otherIdx -> sim) of its top MUTUAL_K
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    const sims = [];
    for (let otherIdx = 0; otherIdx < docCount; otherIdx++) {
      if (otherIdx !== docIdx) {
        sims.push([otherIdx, dotProduct(vectors[docIds[docIdx]], vectors[docIds[otherIdx]])]);
      }
    }
    sims.sort((simA, simB) => simB[1] - simA[1]);
    neighbours.push(new Map(sims.slice(0, MUTUAL_K)));
  }

  const unionFind = new UnionFind(docCount);
  for (let docIdx = 0; docIdx < docCount; docIdx++) {
    for (const [otherIdx, sim] of neighbours[docIdx]) {
      if (otherIdx > docIdx && sim >= threshold && neighbours[otherIdx].has(docIdx)) {
        unionFind.union(docIdx, otherIdx);
      }
    }
  }

  const clusters = {};
  for (let docIdx = 0; docIdx < docIds.length; docIdx++) {
    const clusterRoot = unionFind.find(docIdx);
    if (!clusters[clusterRoot]) clusters[clusterRoot] = [];
    clusters[clusterRoot].push(docIds[docIdx]);
  }

  const tokenised = new Map(docIds.map((docId) => [docId, tokenise(bodyText(doclings[docId]))]));
  const idf = buildIdf([...tokenised.values()]);

  const categories = Object.values(clusters).map((members, index) => {
    const medoidId = clusterMedoid(members, vectors);
    return {
      index,
      members: members.map(docId => ({
        docId,
        filename: doclings[docId]?.filename || docId,
      })),
      keywords: clusterKeywords(members.map(docId => tokenised.get(docId) || []), idf),
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

  await fs.mkdir(path.dirname(categoriesOut), { recursive: true });
  await fs.writeFile(categoriesOut, JSON.stringify(output, null, 2), 'utf-8');

  // Persisted so the frontend embedding view can re-cluster at any threshold
  // without re-embedding.
  const docVectors = {
    generatedAt: output.generatedAt,
    model:       EMBED_MODEL,
    dims:        vectors[docIds[0]].length,
    docs: docIds.map(docId => ({
      docId,
      filename: doclings[docId]?.filename || docId,
      title:    (doclings[docId]?.metadata?.title || '').trim() || null,
      vector:   vectors[docId],
    })),
  };
  await fs.writeFile(vectorsOut, JSON.stringify(docVectors), 'utf-8');

  console.log(
    `[generate_categories] ${categories.length} cluster(s) at threshold=${threshold}`,
    `(${docIds.length} docs) → ${categoriesOut}`
  );
  return output;
}

// Run directly: node backend/extraction/generate_categories.js --threshold 0.75
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const thresholdArgIdx = process.argv.indexOf('--threshold');
  const threshold = thresholdArgIdx !== -1
    ? parseFloat(process.argv[thresholdArgIdx + 1])
    : parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');

  generateCategories(threshold).catch(err => {
    console.error('[generate_categories]', err.message);
    process.exit(1);
  });
}
