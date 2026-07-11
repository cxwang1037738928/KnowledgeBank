/**
 * test_heuristic.js — pipeline stage 5: BM25 + citation PageRank top-k selection
 *
 * Spawns heuristic.py to rank documents and build the citation graph.
 * Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_heuristic.js [--k 5]
 *
 * Prerequisite: tests/test-output/doclings.json, embeddings.json, categories.json
 *   (produced by test_extract.js, test_embed.js, test_generate_categories.js)
 * Needs Ollama running with CITATION_MODEL (default ministral-3:3b-instruct-2512-q4_K_M).
 *
 * Outputs:
 *   tests/test-output/heuristic_output.json
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'heuristic.py');
const PYTHON       = process.env.PYTHON || 'python';

const argv = process.argv;
const i    = argv.indexOf('--k');
const k    = i !== -1 ? argv[i + 1] : '5';

const start = Date.now();
console.log(`[test_heuristic] Spawning heuristic.py --k ${k} (${PYTHON}; GROBID parsedReferences, Ollama only as legacy fallback) ...\n`);

await new Promise((resolve, reject) => {
  const proc = spawn(PYTHON, [HEURISTIC_PY, '--k', k], { stdio: 'inherit', cwd: ROOT });
  proc.on('close', (code) => code !== 0
    ? reject(new Error(`heuristic.py exited with code ${code}`))
    : resolve());
  proc.on('error', (err) => reject(new Error(`Failed to spawn heuristic.py: ${err.message}`)));
});

const heuristic = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'heuristic_output.json'), 'utf-8'));

console.log('\n[test_heuristic] Top-k breakdown (final = 0.25·bm25 + 0.75·pagerank):');
for (const d of heuristic.topK) {
  console.log(
    `  ${d.finalScore.toFixed(4)}  ${d.filename}` +
    `  (repr=${d.bm25Representativeness} novelty=${d.bm25Novelty} pr=${d.pagerankScore})`
  );
}
console.log(`  ${heuristic.edges.length} citation edge(s) across the corpus`);
if (heuristic.edges.length === 0) {
  console.warn('  WARNING: zero citation edges — PageRank is uniform, so ranking is');
  console.warn('           effectively BM25-only. Usual causes: missing metadata.title/');
  console.warn('           authors (see test_extract.js coverage summary), empty');
  console.warn('           references arrays, or citation model parsing failures.');
}

const elapsed = ((Date.now() - start) / 1000).toFixed(2);
const ts      = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${ts}] test_heuristic           : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. Run test_bootstrap.js next.`);
