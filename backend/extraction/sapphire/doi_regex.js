/**
 * doi_regex.js — find each paper's DOI with regular expressions and log it
 * into the document's metadata.
 *
 * Runs as a post-extract step: extract.py (docling + GROBID) writes
 * data/doclings.json, then annotateDois() scans each document's text and
 * stamps metadata.doi. Only the head of the document is searched — the
 * paper's own DOI sits on the first page, while the references section is
 * full of OTHER papers' DOIs that must not be picked up.
 *
 * Exports:
 *   findDoi(text)   — first DOI in a string, or null
 *   annotateDois()  — read doclings.json, set metadata.doi on every entry,
 *                     write it back; returns {docId: doi|null}
 *
 * Run directly: node backend/extraction/sapphire/doi_regex.js
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

// Crossref-recommended pattern: matches 98%+ of DOIs issued since 2000.
// 10.<4-9 digit registrant>/<suffix of allowed characters>.
export const DOI_RE = /\b10\.\d{4,9}\/[-._;()/:a-zA-Z0-9]+/g;

/**
 * First DOI found in `text`, or null. Trailing punctuation that regularly
 * glues onto DOIs in extracted text (sentence periods, closing parens) is
 * stripped from the match.
 */
export function findDoi(text) {
  if (!text) return null;
  const match = text.match(DOI_RE);
  if (!match) return null;
  return match[0].replace(/[.,;)\]]+$/, '') || null;
}

const ROOT          = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const DATA_DIR      = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const DOCLINGS_PATH = path.join(DATA_DIR, 'doclings.json');

// How much of the document head to search. The paper's own DOI appears on
// the first page (header, footer, or copyright block); anything deeper is
// increasingly likely to be a cited work's DOI.
const HEAD_CHARS = 5000;

/**
 * Annotate every doclings.json entry with metadata.doi (null when no DOI is
 * found in the document head). Existing non-null DOIs are left untouched so
 * re-runs are idempotent.
 */
export async function annotateDois() {
  let doclings;
  try {
    doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }

  const found = {};
  for (const [docId, entry] of Object.entries(doclings)) {
    entry.metadata = entry.metadata || {};
    if (!entry.metadata.doi) {
      entry.metadata.doi = findDoi((entry.text || '').slice(0, HEAD_CHARS));
    }
    found[docId] = entry.metadata.doi;
  }

  await fs.writeFile(DOCLINGS_PATH, JSON.stringify(doclings, null, 2), 'utf-8');

  const hits = Object.values(found).filter(Boolean).length;
  console.log(`[doi_regex] ${hits}/${Object.keys(found).length} documents have a DOI → ${DOCLINGS_PATH}`);
  return found;
}

// Run directly: node backend/extraction/sapphire/doi_regex.js
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  annotateDois().catch((err) => {
    console.error('[doi_regex]', err.message);
    process.exit(1);
  });
}
