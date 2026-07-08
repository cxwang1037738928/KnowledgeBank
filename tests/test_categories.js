/**
 * test_categories.js — pipeline stages 3+4: embed (structural chunking) + cluster
 *
 * Embeds all documents then clusters them into categories at the given
 * cosine similarity threshold. All outputs go to tests/test-output/.
 *
 * Run:  node tests/test_categories.js [--threshold 0.75]
 *
 * Prerequisite: run test_extract.js first so tests/test-output/doclings.json
 * is populated.
 *
 * Outputs:
 *   tests/test-output/embeddings.json   — full embedding store
 *   tests/test-output/categories.json   — categories with cluster descriptions
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

// Must be set before imports — embed.js and generate_categories.js read DATA_DIR
// at module load time.
process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

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

const store  = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'embeddings.json'), 'utf-8'));
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

// ---- Print summary ---------------------------------------------------------

console.log(`\n${result.categories.length} cluster(s) at threshold=${threshold}:\n`);
for (const [i, cat] of result.categories.entries()) {
  const members = cat.members.map((m) => m.filename).join(', ');
  const desc    = cat.description ? `\n    "${cat.description}"` : '';
  console.log(`  Cluster ${i + 1} (${cat.members.length} doc${cat.members.length !== 1 ? 's' : ''}): ${members}${desc}`);
}

console.log('\nDone. Run test_downstream.js next.');
