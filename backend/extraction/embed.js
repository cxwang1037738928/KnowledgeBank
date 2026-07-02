/**
 * embed.js — Stage 4 (part 1): embed all documents
 *
 * Reads data/doclings.json, chunks each document's text, generates 768-dim
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
import { chunkText } from '../../src/ingestion/chunker.js';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DOCLINGS_PATH  = path.join(ROOT, 'data', 'doclings.json');
const EMBEDDINGS_OUT = path.join(ROOT, 'data', 'embeddings.json');

const MODEL      = 'Xenova/all-MiniLM-L12-v2';
const DIMENSIONS = 768;
const CHUNK_SIZE    = parseInt(process.env.CHUNK_SIZE    || '500',  10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || '50',   10);
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

  // Load existing store to support incremental updates
  let existing = { chunks: [], metadata: {} };
  if (!force) {
    try {
      existing = JSON.parse(await fs.readFile(EMBEDDINGS_OUT, 'utf-8'));
    } catch { /* first run */ }
  }

  const existingDocIds = new Set(existing.chunks.map((c) => c.docId));
  const toProcess = force ? docIds : docIds.filter((id) => !existingDocIds.has(id));

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

    const rawChunks = chunkText(text, CHUNK_SIZE, CHUNK_OVERLAP);
    const embeddings = [];
    for (let i = 0; i < rawChunks.length; i += BATCH_SIZE) {
      const batch = rawChunks.slice(i, i + BATCH_SIZE);
      embeddings.push(...await embedBatch(batch));
    }

    rawChunks.forEach((chunkText_, i) => {
      newChunks.push({
        id:         `${docId}_${i}`,
        docId,
        filename:   entry.filename,
        pageNumber: null,
        chunkIndex: i,
        text:       chunkText_,
        embedding:  embeddings[i],
        ingestedAt: new Date().toISOString(),
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
