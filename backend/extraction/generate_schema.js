/**
 * generate_schema.js — Stage 5: cluster documents and generate category schemas
 *
 * Algorithm:
 *   1. Load document-level embeddings from data/embeddings.json.
 *      One embedding per document is derived by averaging its chunk embeddings.
 *   2. Cluster by pairwise cosine similarity >= SIMILARITY_THRESHOLD (0.75)
 *      using single-linkage union-find — any two docs with sim >= threshold
 *      end up in the same cluster.
 *   3. For each cluster:
 *      a. Find the centroid document (smallest mean distance to all others).
 *      b. Compute top-J keywords via BM-25-weighted TF across the cluster.
 *      c. Compute I least-common (highest intra-cluster IDF) words per doc.
 *      d. Call Phi-4 via Ollama to generate a base schema for the centroid.
 *      e. Call Phi-4 to extend/populate the schema for every non-centroid doc.
 *
 * Writes:
 *   data/categories.json  — cluster membership, keywords, per-doc rare words
 *   data/schemas.json     — schemas keyed by docId
 *
 * Run: node backend/extraction/generate_schema.js
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const EMBEDDINGS_PATH = path.join(ROOT, 'data', 'embeddings.json');
const DOCLINGS_PATH   = path.join(ROOT, 'data', 'doclings.json');
const CATEGORIES_OUT  = path.join(ROOT, 'data', 'categories.json');
const SCHEMAS_OUT     = path.join(ROOT, 'data', 'schemas.json');

const OLLAMA_URL        = process.env.OLLAMA_URL  || 'http://localhost:11434';
const SCHEMA_MODEL      = process.env.SCHEMA_MODEL || 'phi4';
const SIMILARITY_THRESHOLD = parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');
const TOP_J_KEYWORDS    = 5;
const TOP_I_RARE        = 5;
const BM25_K1 = 1.5;
const BM25_B  = 0.75;

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function cosineSim(a, b) {
  return dot(a, b); // vectors from MiniLM are already L2-normalised
}

function meanVec(vecs) {
  if (vecs.length === 0) return [];
  const dim = vecs[0].length;
  const result = new Float64Array(dim);
  for (const v of vecs) for (let i = 0; i < dim; i++) result[i] += v[i];
  for (let i = 0; i < dim; i++) result[i] /= vecs.length;
  return Array.from(result);
}

// ---------------------------------------------------------------------------
// Union-Find (for single-linkage clustering)
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
    if (this.rank[px] < this.rank[py]) this.parent[px] = py;
    else if (this.rank[px] > this.rank[py]) this.parent[py] = px;
    else { this.parent[py] = px; this.rank[px]++; }
  }
}

// ---------------------------------------------------------------------------
// BM-25 helpers (self-contained, no external library)
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
  'from','as','is','was','are','were','be','been','have','has','had','do',
  'does','did','will','would','could','should','this','that','it','its','not',
  'no','so','if','than','then','we','you','he','she','they','their','our',
]);

function tokenise(text) {
  return text.toLowerCase().match(/[a-z]+/g)?.filter((t) => !STOPWORDS.has(t) && t.length > 2) ?? [];
}

function bm25Scores(docs) {
  const N  = docs.length;
  const dl = docs.map((d) => d.length);
  const avgdl = dl.reduce((a, b) => a + b, 0) / (N || 1);
  const df = new Map();
  for (const tokens of docs) {
    for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const idf = (t) => Math.log((N - (df.get(t) ?? 0) + 0.5) / ((df.get(t) ?? 0) + 0.5) + 1);

  return docs.map((tokens, di) => {
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const scores = new Map();
    for (const [t, count] of tf) {
      const tfn = count * (BM25_K1 + 1) / (count + BM25_K1 * (1 - BM25_B + BM25_B * dl[di] / avgdl));
      scores.set(t, idf(t) * tfn);
    }
    return scores;
  });
}

function topKeywords(docScoresMaps, j) {
  const totals = new Map();
  for (const scoreMap of docScoresMaps) {
    for (const [t, s] of scoreMap) totals.set(t, (totals.get(t) ?? 0) + s);
  }
  return [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, j)
    .map(([term, score]) => ({ term, score: parseFloat(score.toFixed(4)) }));
}

function rareWords(docTokens, allDocTokensInCluster, i) {
  const N = allDocTokensInCluster.length;
  const df = new Map();
  for (const tokens of allDocTokensInCluster) {
    for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const idf = (t) => Math.log((N + 1) / ((df.get(t) ?? 0) + 1));
  const uniqueInDoc = new Set(docTokens);
  return [...uniqueInDoc]
    .map((t) => ({ term: t, idf: parseFloat(idf(t).toFixed(4)) }))
    .sort((a, b) => b.idf - a.idf)
    .slice(0, i);
}

// ---------------------------------------------------------------------------
// Ollama API
// ---------------------------------------------------------------------------

async function ollamaGenerate(prompt, model = SCHEMA_MODEL) {
  const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, stream: false }),
  });
  if (!resp.ok) throw new Error(`Ollama error ${resp.status}: ${await resp.text()}`);
  const body = await resp.json();
  return body.response ?? '';
}

// ---------------------------------------------------------------------------
// Schema generation prompts
// ---------------------------------------------------------------------------

async function generateBaseSchema(docEntry, keywords) {
  const keywordList = keywords.map((k) => k.term).join(', ');
  const preview     = (docEntry.text || '').slice(0, 2000);
  const prompt = `You are a data schema designer.

Document title: ${docEntry.metadata?.title || docEntry.filename}
Category keywords: ${keywordList}
Document excerpt (first 2000 chars):
${preview}

Task: Produce a concise JSON schema that captures the key structured information types present in this category of documents.
The schema should use JSON Schema draft-07 format with "type", "properties", and "required" fields.
Output ONLY the raw JSON schema object, no markdown fences, no explanation.`;

  const raw = await ollamaGenerate(prompt);
  try {
    return JSON.parse(raw.trim());
  } catch {
    // If Phi-4 wraps the JSON in markdown fences, strip them
    const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
    return match ? JSON.parse(match[1].trim()) : { raw };
  }
}

async function extendSchema(baseSchema, docEntry, rareWordList) {
  const preview     = (docEntry.text || '').slice(0, 1500);
  const rareStr     = rareWordList.map((r) => r.term).join(', ');
  const prompt = `You are a data schema designer.

Base schema (from category centroid document):
${JSON.stringify(baseSchema, null, 2)}

Child document title: ${docEntry.metadata?.title || docEntry.filename}
Rare words specific to this document: ${rareStr}
Document excerpt:
${preview}

Task: Extend or specialise the base schema for this specific document.
Add any new properties that are specific to this document but missing from the base schema.
Keep all existing properties. Output ONLY the raw JSON schema object, no markdown fences.`;

  const raw = await ollamaGenerate(prompt);
  try {
    return JSON.parse(raw.trim());
  } catch {
    const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
    return match ? JSON.parse(match[1].trim()) : { raw };
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export async function generateSchemas() {
  const [embStore, doclings] = await Promise.all([
    fs.readFile(EMBEDDINGS_PATH, 'utf-8').then(JSON.parse),
    fs.readFile(DOCLINGS_PATH,   'utf-8').then(JSON.parse),
  ]);

  // Build one embedding per document by averaging its chunk embeddings
  const chunksByDoc = new Map();
  for (const chunk of embStore.chunks) {
    if (!chunksByDoc.has(chunk.docId)) chunksByDoc.set(chunk.docId, []);
    chunksByDoc.get(chunk.docId).push(chunk.embedding);
  }

  const docIds    = [...chunksByDoc.keys()];
  const docVecs   = docIds.map((id) => meanVec(chunksByDoc.get(id)));

  if (docIds.length === 0) {
    console.log('[generate_schema] No embedded documents found.');
    return;
  }

  // ---------------------------------------------------------------------------
  // Cluster by pairwise cosine similarity
  // ---------------------------------------------------------------------------
  const uf = new UnionFind(docIds.length);
  for (let i = 0; i < docIds.length; i++) {
    for (let j = i + 1; j < docIds.length; j++) {
      if (cosineSim(docVecs[i], docVecs[j]) >= SIMILARITY_THRESHOLD) {
        uf.union(i, j);
      }
    }
  }

  const clusterMap = new Map();
  for (let i = 0; i < docIds.length; i++) {
    const root = uf.find(i);
    if (!clusterMap.has(root)) clusterMap.set(root, []);
    clusterMap.get(root).push(i);
  }

  const clusters = [...clusterMap.values()];
  console.log(`[generate_schema] ${clusters.length} cluster(s) from ${docIds.length} documents`);

  // ---------------------------------------------------------------------------
  // Per-cluster analysis
  // ---------------------------------------------------------------------------
  const categoriesOutput = { generatedAt: new Date().toISOString(), categories: [] };
  const schemasOutput    = {};

  for (let ci = 0; ci < clusters.length; ci++) {
    const memberIdxs = clusters[ci];
    const memberDocIds = memberIdxs.map((i) => docIds[i]);
    const memberVecs   = memberIdxs.map((i) => docVecs[i]);
    const clusterLabel = `cluster_${ci}`;

    // Find centroid: doc with minimum mean distance to all other members
    const centroidIdx = (() => {
      let best = 0, bestScore = -Infinity;
      for (let a = 0; a < memberIdxs.length; a++) {
        const mean = memberIdxs.reduce((s, _, b) => s + cosineSim(memberVecs[a], memberVecs[b]), 0) / memberIdxs.length;
        if (mean > bestScore) { bestScore = mean; best = a; }
      }
      return best;
    })();
    const centroidDocId = memberDocIds[centroidIdx];

    // BM-25 keyword analysis across the cluster
    const memberTexts  = memberDocIds.map((id) => doclings[id]?.text || '');
    const memberTokens = memberTexts.map(tokenise);
    const scoresMaps   = bm25Scores(memberTokens);
    const keywords     = topKeywords(scoresMaps, TOP_J_KEYWORDS);

    // Per-doc rare words
    const memberRareWords = memberDocIds.map((_, mi) =>
      rareWords(memberTokens[mi], memberTokens, TOP_I_RARE)
    );

    const categoryEntry = {
      id:         clusterLabel,
      centroidDocId,
      topKeywords: keywords,
      members: memberDocIds.map((id, mi) => ({
        docId:     id,
        filename:  doclings[id]?.filename || id,
        isCentroid: id === centroidDocId,
        rareWords: memberRareWords[mi],
      })),
    };
    categoriesOutput.categories.push(categoryEntry);

    // Schema generation via Phi-4
    console.log(`[generate_schema] Cluster ${ci + 1}/${clusters.length}: ${memberDocIds.length} doc(s), generating schemas ...`);

    const centroidEntry = doclings[centroidDocId];
    if (!centroidEntry) continue;

    let baseSchema;
    try {
      baseSchema = await generateBaseSchema(centroidEntry, keywords);
      schemasOutput[centroidDocId] = { clusterId: clusterLabel, isCentroid: true, schema: baseSchema };
    } catch (err) {
      console.warn(`[generate_schema]   Base schema failed for ${centroidDocId}: ${err.message}`);
      continue;
    }

    for (const [mi, docId] of memberDocIds.entries()) {
      if (docId === centroidDocId) continue;
      const entry = doclings[docId];
      if (!entry) continue;
      try {
        const extended = await extendSchema(baseSchema, entry, memberRareWords[mi]);
        schemasOutput[docId] = { clusterId: clusterLabel, isCentroid: false, schema: extended };
      } catch (err) {
        console.warn(`[generate_schema]   Extension failed for ${docId}: ${err.message}`);
      }
    }
  }

  await fs.mkdir(path.dirname(CATEGORIES_OUT), { recursive: true });
  await fs.writeFile(CATEGORIES_OUT, JSON.stringify(categoriesOutput, null, 2));
  await fs.writeFile(SCHEMAS_OUT,    JSON.stringify(schemasOutput, null, 2));
  console.log(`[generate_schema] Wrote ${CATEGORIES_OUT}`);
  console.log(`[generate_schema] Wrote ${SCHEMAS_OUT}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  generateSchemas().catch((err) => {
    console.error('[generate_schema]', err.message);
    process.exit(1);
  });
}
