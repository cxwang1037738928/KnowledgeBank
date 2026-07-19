/**
 * build_graph.js knowledge graph for the top-k documents
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
import { REF_HEADINGS, normHeading } from '../regex_utils.js';

const ROOT           = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const DATA_DIR       = path.resolve(ROOT, process.env.DATA_DIR || 'data');

// ---------------------------------------------------------------------------
// Node / edge builders
// ---------------------------------------------------------------------------

function documentNode(docId, doclingEntry) {
  return {
    id:       `doc:${docId}`,
    type:     'document',
    docId,
    label:    doclingEntry.metadata?.title || doclingEntry.filename,
    filename: doclingEntry.filename,
    created:  doclingEntry.metadata?.created || null,   // {year, month|null} | null
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
  const fileContents = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(fileContents);
}

export async function buildGraph(dataDir = DATA_DIR) {
  const doclingsPath  = path.join(dataDir, 'doclings.json');
  const heuristicPath = path.join(dataDir, 'heuristic_output.json');
  const graphPath     = path.join(dataDir, 'graph.json');
  let doclings, heuristicOutput;

  try {
    doclings = await readJSON(doclingsPath);
  } catch {
    throw new Error(`${doclingsPath} not found — run extract.py first`);
  }

  try {
    heuristicOutput = await readJSON(heuristicPath);
  } catch {
    throw new Error(`${heuristicPath} not found — run heuristic.py first`);
  }

  const topKIds = new Set(heuristicOutput.topK.map((topDoc) => topDoc.docId));
  const citationEdges = heuristicOutput.edges.filter(
    (edge) => topKIds.has(edge.source) || topKIds.has(edge.target)
  );

  const nodes = [];
  const edges = [];

  // Add document nodes and their section sub-nodes for all top-k docs
  for (const docId of topKIds) {
    const doclingEntry = doclings[docId];
    if (!doclingEntry) continue;

    nodes.push(documentNode(docId, doclingEntry));

    const sections = doclingEntry.sections || [];
    sections.forEach((section, sectionIdx) => {
      // Bibliographies aren't knowledge nodes: their "content" is raw
      // citation strings, which pollute the graph and any downstream
      // embedding/extraction over section nodes.
      if (REF_HEADINGS.has(normHeading(section.heading || ''))) return;
      if (section.heading || section.text.length > 50) {
        nodes.push(sectionNode(docId, sectionIdx, section));
        edges.push(sectionEdge(docId, sectionIdx));
      }
    });
  }

  // Add citation edges (may connect to non-top-k docs as stubs)
  for (const { source, target } of citationEdges) {
    // Ensure the target doc node exists as at least a stub
    const targetNodeId = `doc:${target}`;
    if (!nodes.find((node) => node.id === targetNodeId) && doclings[target]) {
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

  await fs.mkdir(path.dirname(graphPath), { recursive: true });
  await fs.writeFile(graphPath, JSON.stringify(graph, null, 2), 'utf-8');

  console.log(`[build_graph] ${nodes.length} nodes, ${edges.length} edges → ${graphPath}`);
  return graph;
}

// Run directly: node backend/extraction/sapphire/build_graph.js
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  buildGraph().catch((err) => {
    console.error('[build_graph]', err.message);
    process.exit(1);
  });
}
