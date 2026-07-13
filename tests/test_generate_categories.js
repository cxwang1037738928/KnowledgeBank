/**
 * test_generate_categories.js — pipeline stage 4: cluster docs into categories
 *
 * Clusters embedded documents by cosine similarity and indexes each cluster
 * with its top BM25 keywords and centroid medoid (no LLM). Outputs go to
 * tests/test-output/.
 *
 * Run:  node tests/test_generate_categories.js [--threshold 0.75]
 *
 * Prerequisite: tests/test-output/doclings.json (produced by test_extract.js)
 *
 * Outputs:
 *   tests/test-output/categories.json  — { index, members, keywords, medoid }
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

const thresholdArgIdx = process.argv.indexOf('--threshold');
const threshold = thresholdArgIdx !== -1
  ? parseFloat(process.argv[thresholdArgIdx + 1])
  : parseFloat(process.env.CLUSTER_SIMILARITY || '0.75');

const start = Date.now();
console.log('[test_generate_categories] Clustering at threshold=' + threshold + ' ...\n');
const result = await generateCategories(threshold);

console.log(`\n[test_generate_categories] ${result.categories.length} cluster(s) at threshold=${threshold}:\n`);
for (const category of result.categories) {
  const memberNames = category.members.map((member) => member.filename).join(', ');
  console.log(`  Category ${category.index} (${category.members.length} doc${category.members.length !== 1 ? 's' : ''}): ${memberNames}`);
  console.log(`    keywords: ${category.keywords.slice(0, 8).join(', ')}`);
  console.log(`    medoid:   ${category.medoid.filename}`);
}

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_generate_categories : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_heuristic.js next.`);
