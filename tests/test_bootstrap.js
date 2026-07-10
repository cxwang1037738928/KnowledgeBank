/**
 * test_bootstrap.js — pipeline stage 6: generate synthetic bootstrap queries
 *
 * Generates per-category synthetic queries via the bootstrap LLM.
 * Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_bootstrap.js [--per-category 8]
 *
 * Prerequisite: tests/test-output/categories.json, heuristic_output.json
 *   (produced by test_generate_categories.js, test_heuristic.js)
 * Needs Ollama running with BOOTSTRAP_MODEL (default ministral-3:3b-instruct-2512-q4_K_M).
 *
 * Outputs:
 *   tests/test-output/bootstrap_queries.json
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const { bootstrapQueries } = await import('../backend/extraction/bootstrap_queries.js');

const argv        = process.argv;
const j           = argv.indexOf('--per-category');
const perCategory = j !== -1 ? parseInt(argv[j + 1], 10) : 8;

const start = Date.now();
console.log(`[test_bootstrap] Generating bootstrap queries (${perCategory} per category, one LLM call each) ...\n`);
const boot = await bootstrapQueries({ perCategory });

for (const cat of boot.categories) {
  console.log(`  Category ${cat.categoryIndex + 1} — ${cat.keywords.slice(0, 5).join(', ')}:`);
  for (const q of cat.queries.slice(0, 3)) {
    console.log(`    [${q.type}] ${q.query}`);
  }
  if (cat.queries.length > 3) console.log(`    ... ${cat.queries.length - 3} more`);
  if (cat.queries.length === 0) console.warn('    WARNING: no queries generated for this category');
}

const elapsed = ((Date.now() - start) / 1000).toFixed(2);
const ts      = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${ts}] test_bootstrap           : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_build_graph.js next.`);
