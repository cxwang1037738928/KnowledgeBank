/**
 * test_categories.js — pipeline stages 3+4: embed (structural chunking) + cluster
 *
 * Embeds all documents then clusters them into categories at the given
 * cosine similarity threshold. Copies the resulting categories.json into
 * tests/test-output/ and prints chunking stats + a cluster summary.
 *
 * Run:  node tests/test_categories.js [--threshold 0.75]
 *
 * Prerequisite: run test_extract.js first so data/doclings.json is populated.
 *
 * CHANGED to match the current pipeline: embed.js now uses chunkDocument()
 * (structure-aware — section boundaries, heading prefixes, standalone table
 * chunks) instead of flat chunkText(), and stores heading / sectionIndex /
 * chunkType per chunk. This test inspects those fields:
 *   - chunk type breakdown (text vs table)
 *   - heading coverage — % of chunks that carry a section heading; low
 *     coverage means docling found little structure and most docs fell
 *     back to sliding-window chunking
 *   - oversized-chunk warning — all-MiniLM-L12-v2 truncates ~256 word-piece
 *     tokens (~200 words); chunks beyond that embed only their head
 *
 * Outputs:
 *   data/embeddings.json                — full embedding store (pipeline artifact)
 *   data/categories.json                — categories (pipeline artifact)
 *   tests/test-output/categories.json   — copy for inspection
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT        = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_OUTPUT = path.join(ROOT, 'tests', 'test-output');

await fs.mkdir(TEST_OUTPUT, { recursive: true });

const { embedAll }           = await import('../backend/extraction/embed.js');
const { generateCategories } = await import('../backend/extraction/generate_categories.js');

const tArg      = process.argv.indexOf('--threshold');
const threshold = tArg !== -1
  ? parseFloat(process.argv[tArg + 1])
  : parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');

// ---- Embed -----------------------------------------------------------------

console.log('[test_categories] Embedding documents (--force, structural chunking) ...\n');
await embedAll({ force: true });

// ---- Chunking stats ----------------------------------------------------------

const store  = JSON.parse(await fs.readFile(path.join(ROOT, 'data', 'embeddings.json'), 'utf-8'));
const chunks = store.chunks || [];

const byDoc = new Map();
for (const c of chunks) {
  if (!byDoc.has(c.docId)) byDoc.set(c.docId, []);
  byDoc.get(c.docId).push(c);
}

const wordCounts   = chunks.map((c) => (c.text.match(/\S+/g) || []).length);
const avgWords     = wordCounts.reduce((s, w) => s + w, 0) / Math.max(chunks.length, 1);
const oversized    = wordCounts.filter((w) => w > 220).length;
const tableChunks  = chunks.filter((c) => c.chunkType === 'table').length;
const withHeading  = chunks.filter((c) => c.heading && c.heading !== 'Table').length;

console.log('\n[test_categories] Chunking stats:');
console.log(`  ${chunks.length} chunks across ${byDoc.size} docs (model ${store.metadata.model}, ${store.metadata.dimensions}-dim)`);
console.log(`  types: ${chunks.length - tableChunks} text, ${tableChunks} table`);
console.log(`  heading coverage: ${withHeading}/${chunks.length} (${(100 * withHeading / Math.max(chunks.length, 1)).toFixed(0)}%)`);
console.log(`  avg ${avgWords.toFixed(0)} words/chunk`);
for (const [docId, docChunks] of byDoc) {
  console.log(`    ${docChunks[0].filename}: ${docChunks.length} chunks`);
}
if (oversized > 0) {
  console.warn(`  WARNING: ${oversized} chunk(s) exceed ~220 words — MiniLM truncates ~256 word-piece`);
  console.warn('           tokens, so their tails are NOT embedded. Lower CHUNK_SIZE (~200) in .env.');
}

// ---- Generate categories ---------------------------------------------------

console.log('\n[test_categories] Clustering at threshold=' + threshold + ' ...\n');
const result = await generateCategories(threshold);

// ---- Copy to test-output ---------------------------------------------------

const dest = path.join(TEST_OUTPUT, 'categories.json');
await fs.copyFile(path.join(ROOT, 'data', 'categories.json'), dest);
console.log('\n[test_categories] categories.json → tests/test-output/categories.json');

// ---- Print summary ---------------------------------------------------------

console.log(`\n${result.categories.length} cluster(s) at threshold=${threshold}:\n`);
for (const [i, cat] of result.categories.entries()) {
  const members  = cat.members.map((m) => m.filename).join(', ');
  console.log(`  Cluster ${i + 1} (${cat.members.length} doc${cat.members.length !== 1 ? 's' : ''}): ${members}`);
}

console.log('\nDone. Run test_downstream.js next.');