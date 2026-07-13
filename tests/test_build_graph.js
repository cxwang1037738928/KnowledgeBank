/**
 * test_build_graph.js — pipeline stage 7: build knowledge graph
 *
 * Builds the knowledge graph over the top-k documents.
 * Outputs go to tests/test-output/.
 *
 * Run:  node tests/test_build_graph.js
 *
 * Prerequisite: tests/test-output/doclings.json, heuristic_output.json
 *   (produced by test_extract.js, test_heuristic.js)
 *
 * Outputs:
 *   tests/test-output/graph.json
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');

process.env.DATA_DIR = TEST_DATA;

await fs.mkdir(TEST_DATA, { recursive: true });

const { buildGraph } = await import('../backend/extraction/sapphire/build_graph.js');

const start = Date.now();
console.log('[test_build_graph] Building knowledge graph ...\n');
const graph = await buildGraph();

const docNodes     = graph.nodes.filter((node) => node.type === 'document').length;
const sectionNodes = graph.nodes.filter((node) => node.type === 'section').length;
const citeEdges    = graph.edges.filter((edge) => edge.type === 'cites').length;
const sectionEdges = graph.edges.filter((edge) => edge.type === 'has_section').length;

console.log('[test_build_graph] Graph summary:');
console.log(`  nodes: ${docNodes} document, ${sectionNodes} section`);
console.log(`  edges: ${citeEdges} cites, ${sectionEdges} has_section`);

const elapsed   = ((Date.now() - start) / 1000).toFixed(2);
const timestamp = new Date().toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
await fs.appendFile(
  path.join(ROOT, 'tests', 'test_log.txt'),
  `[${timestamp}] test_build_graph         : ${elapsed}s\n`,
  'utf-8',
);
console.log(`\nDone in ${elapsed}s. All outputs in tests/test-output/.`);
