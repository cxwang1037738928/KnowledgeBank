/**
 * build_graph.js — Stage 3: knowledge graph for the top-k documents
 *
 * Reads:
 *   data/doclings.json          — extracted document content
 *   data/heuristic_output.json  — top-k doc IDs + citation edges
 *
 * Writes:
 *   data/graph.json — plain JSON adjacency structure:
 *     nodes: document nodes + section nodes derived from document structure
 *     edges: citation edges (from heuristic connectivity) + section-membership edges
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DOCLINGS_PATH  = path.join(ROOT, 'data', 'doclings.json');
const HEURISTIC_PATH = path.join(ROOT, 'data', 'heuristic_output.json');
const GRAPH_PATH     = path.join(ROOT, 'data', 'graph.json');

// ---------------------------------------------------------------------------
// Node / edge builders
// ---------------------------------------------------------------------------

function documentNode(docId, entry) {
  return {
    id:       `doc:${docId}`,
    type:     'document',
    docId,
    label:    entry.metadata?.title || entry.filename,
    filename: entry.filename,
  };
}

function sectionNode(docId, sectionIndex, section) {
  return {
    id:      `section:${docId}:${sectionIndex}`,
    type:    'section',
    docId,
    label:   section.heading || `Section ${sectionIndex + 1}`,
    preview: section.text.slice(0, 200),
  };
}

function sectionEdge(docId, sectionIndex) {
  return {
    source: `doc:${docId}`,
    target: `section:${docId}:${sectionIndex}`,
    type:   'has_section',
  };
}

function citationEdge(sourceDocId, targetDocId) {
  return {
    source: `doc:${sourceDocId}`,
    target: `doc:${targetDocId}`,
    type:   'cites',
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function readJSON(filePath) {
  const raw = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(raw);
}

export async function buildGraph() {
  let doclings, heuristic;

  try {
    doclings = await readJSON(DOCLINGS_PATH);
  } catch {
    throw new Error(`data/doclings.json not found — run extract.py first`);
  }

  try {
    heuristic = await readJSON(HEURISTIC_PATH);
  } catch {
    throw new Error(`data/heuristic_output.json not found — run heuristic.py first`);
  }

  const topKIds = new Set(heuristic.topK.map((d) => d.docId));
  const citationEdges = heuristic.edges.filter(
    (e) => topKIds.has(e.source) || topKIds.has(e.target)
  );

  const nodes = [];
  const edges = [];

  // Add document nodes and their section sub-nodes for all top-k docs
  for (const docId of topKIds) {
    const entry = doclings[docId];
    if (!entry) continue;

    nodes.push(documentNode(docId, entry));

    const sections = entry.sections || [];
    sections.forEach((section, i) => {
      if (section.heading || section.text.length > 50) {
        nodes.push(sectionNode(docId, i, section));
        edges.push(sectionEdge(docId, i));
      }
    });
  }

  // Add citation edges (may connect to non-top-k docs as stubs)
  for (const { source, target } of citationEdges) {
    // Ensure the target doc node exists as at least a stub
    const targetNodeId = `doc:${target}`;
    if (!nodes.find((n) => n.id === targetNodeId) && doclings[target]) {
      nodes.push(documentNode(target, doclings[target]));
    }
    edges.push(citationEdge(source, target));
  }

  const graph = {
    createdAt: new Date().toISOString(),
    topKDocIds: [...topKIds],
    nodes,
    edges,
  };

  await fs.mkdir(path.dirname(GRAPH_PATH), { recursive: true });
  await fs.writeFile(GRAPH_PATH, JSON.stringify(graph, null, 2), 'utf-8');

  console.log(`[build_graph] ${nodes.length} nodes, ${edges.length} edges → ${GRAPH_PATH}`);
  return graph;
}

// Run directly: node backend/extraction/build_graph.js
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  buildGraph().catch((err) => {
    console.error('[build_graph]', err.message);
    process.exit(1);
  });
}
