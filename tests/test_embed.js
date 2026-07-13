/**
 * test_embed.js — pipeline stage 3: embed documents into chunks + vectors
 *
 * Embeds all documents using structural chunking. Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_embed.js
 *
 * Prerequisite: run test_extract.js first (needs tests/test-output/doclings.json).
 *
 * Outputs:
 *   tests/test-output/embeddings.json  — full embedding store (chunks + vectors)
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const { embedAll } = await import('../backend/extraction/embed.js');

const start = Date.now();
console.log('[test_embed] Embedding documents (--force, structural chunking) ...\n');
await embedAll({ force: true });

const store  = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'embeddings.json'), 'utf-8'));
const chunks = store.chunks || [];
const chunksByDoc = new Map();
for (const chunk of chunks) {
  if (!chunksByDoc.has(chunk.docId)) chunksByDoc.set(chunk.docId, []);
  chunksByDoc.get(chunk.docId).push(chunk);
}

console.log('\n[test_embed] Embedding complete:');
console.log(`  model: ${store.metadata.model}  (${store.metadata.dimensions}-dim)`);
console.log(`  ${chunks.length} chunks across ${chunksByDoc.size} docs`);
for (const [, docChunks] of chunksByDoc) {
  console.log(`    ${docChunks[0].filename}: ${docChunks.length} chunks`);
}

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_embed               : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_chunking.js to inspect chunk quality or test_generate_categories.js next.`);
