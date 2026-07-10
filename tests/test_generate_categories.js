/**
 * test_generate_categories.js — pipeline stage 4: cluster docs into categories
 *
 * Clusters embedded documents by cosine similarity and generates category
 * descriptions via LLM. Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_generate_categories.js [--threshold 0.75]
 *
 * Prerequisite: tests/test-output/embeddings.json (produced by test_embed.js)
 *
 * Outputs:
 *   tests/test-output/categories.json  — clusters with descriptions
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const { generateCategories } = await import('../backend/extraction/generate_categories.js');

const tArg      = process.argv.indexOf('--threshold');
const threshold = tArg !== -1
  ? parseFloat(process.argv[tArg + 1])
  : parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');

const start = Date.now();
console.log('[test_generate_categories] Clustering at threshold=' + threshold + ' ...\n');
const result = await generateCategories(threshold);

console.log(`\n[test_generate_categories] ${result.categories.length} cluster(s) at threshold=${threshold}:\n`);
for (const [i, cat] of result.categories.entries()) {
  const members = cat.members.map((m) => m.filename).join(', ');
  const desc    = cat.description ? `\n    "${cat.description}"` : '';
  console.log(`  Cluster ${i + 1} (${cat.members.length} doc${cat.members.length !== 1 ? 's' : ''}): ${members}${desc}`);
}

const elapsed = ((Date.now() - start) / 1000).toFixed(2);
const ts      = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${ts}] test_generate_categories : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_heuristic.js next.`);
