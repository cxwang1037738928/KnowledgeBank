/**
 * embed.js  embed all documents
 *
 * Reads data/doclings.json, chunks each document's text, generates 384-dim
 * embeddings with Xenova/all-MiniLM-L12-v2, and writes the results to
 * data/embeddings.json (overwrites — this is the authoritative embedding
 * store for the whole pipeline).
 *
 * Run: node backend/extraction/embed.js
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline } from '@xenova/transformers';
import { chunkDocument } from './chunker.js';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA_DIR       = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const DOCLINGS_PATH  = path.join(DATA_DIR, 'doclings.json');
const EMBEDDINGS_OUT = path.join(DATA_DIR, 'embeddings.json');

const MODEL      = 'Xenova/all-MiniLM-L12-v2';
const DIMENSIONS = 384; // all-MiniLM-L12-v2 outputs 384-dim vectors
const CHUNK_SIZE    = parseInt(process.env.CHUNK_SIZE    || '180', 10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || '30',  10);
const BATCH_SIZE    = 32;

// ---------------------------------------------------------------------------
// Embedder (lazy singleton)
// ---------------------------------------------------------------------------

let _extractor = null;

async function getExtractor() {
  if (!_extractor) {
    console.log(`[embed] Loading ${MODEL} ...`);
    _extractor = await pipeline('feature-extraction', MODEL, { quantized: true });
  }
  return _extractor;
}

async function embedBatch(texts) {
  const extractor = await getExtractor();
  const output = await extractor(texts, { pooling: 'mean', normalize: true });
  const dims = output.data.length / texts.length;
  const results = [];
  for (let i = 0; i < texts.length; i++) {
    results.push(Array.from(output.data.slice(i * dims, (i + 1) * dims)));
  }
  return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export async function embedAll({ force = false } = {}) {
  let doclings;
  try {
    const raw = await fs.readFile(DOCLINGS_PATH, 'utf-8');
    doclings = JSON.parse(raw);
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }

  const docIds = Object.keys(doclings);
  if (docIds.length === 0) {
    console.log('[embed] doclings.json is empty — nothing to embed.');
    return;
  }

  let existing = { chunks: [], metadata: {} };
  if (!force) {
    try {
      existing = JSON.parse(await fs.readFile(EMBEDDINGS_OUT, 'utf-8'));
    } catch { /* first run */ }
  }

  // A doc counts as embedded only if its chunks are newer than its extraction
  // — presence alone would keep stale embeddings after a re-extract.
  const newestChunkAt = new Map();
  for (const c of existing.chunks) {
    const t = Date.parse(c.ingestedAt) || 0;
    if (t > (newestChunkAt.get(c.docId) || 0)) newestChunkAt.set(c.docId, t);
  }
  const isFresh = (id) => {
    if (!newestChunkAt.has(id)) return false;
    const extractedAt = Date.parse(doclings[id].extractedAt) || 0;
    return newestChunkAt.get(id) >= extractedAt;
  };
  const toProcess = force ? docIds : docIds.filter((id) => !isFresh(id));

  if (toProcess.length === 0) {
    console.log('[embed] All documents already embedded. Use --force to re-embed.');
    return;
  }

  console.log(`[embed] Embedding ${toProcess.length} document(s) ...`);

  const newChunks = [];
  for (const docId of toProcess) {
    const entry = doclings[docId];
    const text  = entry.text || '';
    if (!text.trim()) {
      console.warn(`[embed]   Skipping ${entry.filename} — no text`);
      continue;
    }

    const rawChunks = chunkDocument(entry, {
      chunkSize: CHUNK_SIZE,
      overlap:   CHUNK_OVERLAP,
    });
    const embeddings = [];
    for (let i = 0; i < rawChunks.length; i += BATCH_SIZE) {
      const batch = rawChunks.slice(i, i + BATCH_SIZE).map((c) => c.text);
      embeddings.push(...await embedBatch(batch));
    }

    rawChunks.forEach((chunk, i) => {
      newChunks.push({
        id:           `${docId}_${i}`,
        docId,
        filename:     entry.filename,
        pageNumber:   null,
        chunkIndex:   i,
        heading:      chunk.heading,
        sectionIndex: chunk.sectionIndex,
        chunkType:    chunk.type,
        text:         chunk.text,
        embedding:    embeddings[i],
        ingestedAt:   new Date().toISOString(),
      });
    });

    console.log(`[embed]   ${entry.filename} — ${rawChunks.length} chunks`);
  }

  // Merge: drop old chunks for re-processed docs, append new ones
  const reprocessedIds = new Set(toProcess);
  const kept = existing.chunks.filter((c) => !reprocessedIds.has(c.docId));
  const allChunks = [...kept, ...newChunks];

  const store = {
    chunks: allChunks,
    metadata: {
      model:      MODEL,
      dimensions: DIMENSIONS,
      created:    existing.metadata?.created || new Date().toISOString(),
      updated:    new Date().toISOString(),
      totalDocs:  new Set(allChunks.map((c) => c.docId)).size,
    },
  };

  await fs.mkdir(path.dirname(EMBEDDINGS_OUT), { recursive: true });
  await fs.writeFile(EMBEDDINGS_OUT, JSON.stringify(store, null, 2), 'utf-8');
  console.log(`[embed] Wrote ${allChunks.length} chunks (${store.metadata.totalDocs} docs) → ${EMBEDDINGS_OUT}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const force = process.argv.includes('--force');
  embedAll({ force }).catch((err) => {
    console.error('[embed]', err.message);
    process.exit(1);
  });
}