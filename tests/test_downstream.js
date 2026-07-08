/**
 * test_downstream.js — pipeline stages 5–7: heuristic ranking → bootstrap
 * queries → knowledge graph
 *
 * All outputs go to tests/test-output/ — no writes to data/.
 *
 *   5. heuristic.py        — BM25 (representativeness + novelty) + citation
 *                            PageRank top-k selection. Needs Ollama running
 *                            (Phi-4 parses reference strings).
 *   6. bootstrap_queries.js — per-category synthetic queries via Ministral-3b.
 *   7. build_graph.js       — knowledge graph over the top-k documents.
 *
 * Run:  node tests/test_downstream.js [--k 5] [--per-category 8] [--skip-bootstrap]
 *
 * Prerequisite: run test_categories.js first (needs tests/test-output/doclings.json,
 * embeddings.json, categories.json). Ollama must be serving the
 * model named in CITATION_MODEL (default phi4) and BOOTSTRAP_MODEL (default ministral:3b).
 *
 * Outputs (all in tests/test-output/):
 *   heuristic_output.json, bootstrap_queries.json, graph.json
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

// Must be set before imports and before spawning Python — all pipeline modules
// read DATA_DIR at load time.
process.env.DATA_DIR = TEST_DATA;

const HEURISTIC_PY = path.join(ROOT, 'backend', 'extraction', 'heuristic.py');

await fs.mkdir(TEST_DATA, { recursive: true });

const argv = process.argv;
const flag = (name, fallback) => {
  const i = argv.indexOf(name);
  return i !== -1 ? argv[i + 1] : fallback;
};
const k             = flag('--k', '5');
const perCategory   = parseInt(flag('--per-category', '8'), 10);
const skipBootstrap = argv.includes('--skip-bootstrap');

const PYTHON = process.env.PYTHON || 'python';

// ---- 5. heuristic.py ---------------------------------------------------------

console.log(`[test_downstream] Spawning heuristic.py --k ${k} (${PYTHON}, Phi-4 citation parsing — needs Ollama) ...\n`);

await new Promise((resolve, reject) => {
  const proc = spawn(PYTHON, [HEURISTIC_PY, '--k', k], { stdio: 'inherit', cwd: ROOT });
  proc.on('close', (code) => code !== 0
    ? reject(new Error(`heuristic.py exited with code ${code}`))
    : resolve());
  proc.on('error', (err) => reject(new Error(`Failed to spawn heuristic.py: ${err.message}`)));
});

const heuristic = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'heuristic_output.json'), 'utf-8'));

console.log('\n[test_downstream] Top-k breakdown (final = 0.25·bm25 + 0.75·pagerank):');
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
  console.warn('           references arrays, or Phi-4 parsing failures.');
}

// ---- 6. bootstrap_queries.js ---------------------------------------------------

if (skipBootstrap) {
  console.log('\n[test_downstream] Skipping bootstrap queries (--skip-bootstrap).');
} else {
  console.log('\n[test_downstream] Generating bootstrap queries (one Ministral-3b call per category) ...\n');
  const { bootstrapQueries } = await import('../backend/extraction/bootstrap_queries.js');
  const boot = await bootstrapQueries({ perCategory });

  for (const cat of boot.categories) {
    console.log(`  Category ${cat.categoryIndex + 1} — ${cat.keywords.slice(0, 5).join(', ')}:`);
    for (const q of cat.queries.slice(0, 3)) {
      console.log(`    [${q.type}] ${q.query}`);
    }
    if (cat.queries.length > 3) console.log(`    ... ${cat.queries.length - 3} more`);
    if (cat.queries.length === 0) console.warn('    WARNING: no queries generated for this category');
  }
}

// ---- 7. build_graph.js ---------------------------------------------------------

console.log('\n[test_downstream] Building knowledge graph ...\n');
const { buildGraph } = await import('../backend/extraction/build_graph.js');
const graph = await buildGraph();

const docNodes     = graph.nodes.filter((n) => n.type === 'document').length;
const sectionNodes = graph.nodes.filter((n) => n.type === 'section').length;
const citeEdges    = graph.edges.filter((e) => e.type === 'cites').length;
const sectionEdges = graph.edges.filter((e) => e.type === 'has_section').length;

console.log('[test_downstream] Graph summary:');
console.log(`  nodes: ${docNodes} document, ${sectionNodes} section`);
console.log(`  edges: ${citeEdges} cites, ${sectionEdges} has_section`);

console.log('\nDone. All outputs in tests/test-output/.');
