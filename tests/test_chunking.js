/**
 * test_chunking.js — inspect chunking quality from an existing embedding store
 *
 * Reads tests/test-output/embeddings.json and prints per-chunk statistics.
 * Does NOT re-embed — run test_embed.js first.
 *
 * Run:  node tests/test_chunking.js
 *
 * Prerequisite: tests/test-output/embeddings.json (produced by test_embed.js)
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

const start = Date.now();

const store  = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'embeddings.json'), 'utf-8'));
const chunks = store.chunks || [];

const chunksByDoc = new Map();
for (const chunk of chunks) {
  if (!chunksByDoc.has(chunk.docId)) chunksByDoc.set(chunk.docId, []);
  chunksByDoc.get(chunk.docId).push(chunk);
}

const wordCounts  = chunks.map((chunk) => (chunk.text.match(/\S+/g) || []).length);
const avgWords    = wordCounts.reduce((total, wordCount) => total + wordCount, 0)
                    / Math.max(chunks.length, 1);
const oversized   = wordCounts.filter((wordCount) => wordCount > 220).length;
const tableChunks = chunks.filter((chunk) => chunk.chunkType === 'table').length;
const withHeading = chunks.filter((chunk) => chunk.heading && chunk.heading !== 'Table').length;

console.log('[test_chunking] Chunking stats:');
console.log(`  ${chunks.length} chunks across ${chunksByDoc.size} docs (model ${store.metadata.model}, ${store.metadata.dimensions}-dim)`);
console.log(`  types: ${chunks.length - tableChunks} text, ${tableChunks} table`);
console.log(`  heading coverage: ${withHeading}/${chunks.length} (${(100 * withHeading / Math.max(chunks.length, 1)).toFixed(0)}%)`);
console.log(`  avg ${avgWords.toFixed(0)} words/chunk`);
for (const [, docChunks] of chunksByDoc) {
  console.log(`    ${docChunks[0].filename}: ${docChunks.length} chunks`);
}
if (oversized > 0) {
  console.warn(`  WARNING: ${oversized} chunk(s) exceed ~220 words — MiniLM truncates ~256 word-piece`);
  console.warn('           tokens, so their tails are NOT embedded. Lower CHUNK_SIZE (~200) in .env.');
}

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_chunking            : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s.`);
