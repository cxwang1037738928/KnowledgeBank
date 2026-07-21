/**
 * test_kg_graph.js — pipeline stage 6: kg-gen knowledge graph
 *
 * Spawns kg_graph.py to build the entity/relation graph over the extracted
 * text. Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_kg_graph.js
 *
 * Prerequisite: tests/test-output/doclings.json (produced by test_extract.js)
 * Also needs Ollama running with KG_MODEL pulled.
 *
 * Outputs:
 *   tests/test-output/graph.json
 *   tests/test-output/kg_view.html
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

const KG_GRAPH_PY = path.join(ROOT, 'backend', 'extraction', 'kg_graph.py');
const PYTHON      = process.env.PYTHON || 'python';

const start = Date.now();
console.log(`[test_kg_graph] Spawning kg_graph.py (${PYTHON}; model ${process.env.KG_MODEL}) ...\n`);

await new Promise((resolve, reject) => {
  const proc = spawn(PYTHON, [KG_GRAPH_PY], { stdio: 'inherit', cwd: ROOT });
  proc.on('close', (exitCode) => exitCode !== 0
    ? reject(new Error(`kg_graph.py exited with code ${exitCode}`))
    : resolve());
  proc.on('error', (err) => reject(new Error(`Failed to spawn kg_graph.py: ${err.message}`)));
});

const graph = JSON.parse(await fs.readFile(path.join(TEST_DATA, 'graph.json'), 'utf-8'));
const viewHtml = await fs.stat(path.join(TEST_DATA, 'kg_view.html'));

console.log('\n[test_kg_graph] Graph summary:');
console.log(`  ${graph.entities.length} entities, ${graph.relations.length} relations, ` +
            `${graph.edges.length} relation types`);
console.log(`  over ${graph.sourceDocIds.length} document(s)`);
console.log(`  kg_view.html: ${viewHtml.size} bytes`);
for (const [subject, predicate, object] of graph.relations.slice(0, 5)) {
  console.log(`    ${subject} —[${predicate}]→ ${object}`);
}
if (graph.relations.length === 0) {
  console.warn('  WARNING: zero relations — the model returned no triples. Usual causes:');
  console.warn('           Ollama down, KG_MODEL not pulled, or KG_CHARS_PER_DOC too small.');
}

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_kg_graph            : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. All outputs in tests/test-output/.`);
