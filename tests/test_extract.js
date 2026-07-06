/**
 * test_extract.js — pipeline stage 2: docling extraction
 *
 * Spawns extract.py --force on all documents in data/documents.json, then
 * writes one .txt file per document into tests/test-output/text/.
 * Run: node tests/test_extract.js
 *
 * Prerequisite: run test_cleaning.js first so data/documents.json and
 * data/enhanced/ are populated.
 *
 * CHANGED to match the current pipeline — two downstream stages now depend
 * on fields this test previously ignored, so it surfaces them:
 *   - metadata.title / metadata.authors — heuristic.py's citation matching
 *     (build_connectivity) requires BOTH a title and an author match to
 *     create an edge. Docs missing these produce zero incoming edges and a
 *     dead PageRank term. This test prints a corpus-wide coverage summary
 *     and warns loudly when coverage is low.
 *   - sections — chunker.js chunks along section boundaries and prefixes
 *     headings. Each .txt now includes the section outline so you can
 *     eyeball whether docling recovered real structure or one giant blob
 *     (blob docs fall back to plain sliding-window chunking).
 *
 * Outputs:
 *   tests/test-output/text/<docId>_<name>.txt — metadata header + section
 *     outline + full extracted text
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const ROOT          = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_OUTPUT   = path.join(ROOT, 'tests', 'test-output', 'text');
const DOCLINGS_PATH = path.join(ROOT, 'data', 'doclings.json');
const EXTRACT_PY    = path.join(ROOT, 'backend', 'extraction', 'extract.py');

await fs.mkdir(TEST_OUTPUT, { recursive: true });

// ---- Run extract.py --------------------------------------------------------

console.log('[test_extract] Spawning extract.py --force ...\n');

await new Promise((resolve, reject) => {
  const proc = spawn('python', [EXTRACT_PY, '--force'], {
    stdio: 'inherit',
    cwd: ROOT,
  });
  proc.on('close', (code) => {
    if (code !== 0) reject(new Error(`extract.py exited with code ${code}`));
    else resolve();
  });
  proc.on('error', (err) => reject(new Error(`Failed to spawn extract.py: ${err.message}`)));
});

// ---- Write per-doc .txt files ----------------------------------------------

let doclings;
try {
  doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
} catch {
  console.error('[test_extract] data/doclings.json not found — extract.py may have failed.');
  process.exit(1);
}

const entries = Object.values(doclings);
if (entries.length === 0) {
  console.log('\n[test_extract] doclings.json is empty — nothing to write.');
  process.exit(0);
}

console.log(`\n[test_extract] Writing ${entries.length} .txt file(s) to tests/test-output/text/\n`);

for (const entry of entries) {
  const baseName = path.basename(entry.filename, '.pdf');
  const outFile  = `${entry.docId}_${baseName}.txt`;
  const outPath  = path.join(TEST_OUTPUT, outFile);

  const meta    = entry.metadata || {};
  const outline = (entry.sections || [])
    .map((s, i) => `  [${i}] ${s.heading || '(no heading)'} — ${(s.text || '').split(/\s+/).length} words`)
    .join('\n');

  const lines = [
    `File:       ${entry.filename}`,
    `DocId:      ${entry.docId}`,
    `Extracted:  ${entry.extractedAt}`,
    `Title:      ${meta.title || '(MISSING — citation matching cannot target this doc)'}`,
    `Authors:    ${(meta.authors || []).join('; ') || '(MISSING — citation matching cannot target this doc)'}`,
    `Abstract:   ${meta.abstract ? `${meta.abstract.split(/\s+/).length} words` : '(missing)'}`,
    `Sections:   ${entry.sections.length}`,
    `Tables:     ${entry.tables.length}`,
    `References: ${entry.references.length}`,
    '',
    'Section outline:',
    outline || '  (none — chunker will fall back to plain sliding-window chunking)',
    '',
    '='.repeat(72),
    '',
    entry.text,
  ];

  await fs.writeFile(outPath, lines.join('\n'), 'utf-8');
  console.log(`  → ${outFile}  (${entry.text.length} chars, ${entry.sections.length} sections, ${entry.references.length} refs)`);
}

// ---- Metadata coverage summary ----------------------------------------------
// heuristic.py needs title AND authors on the TARGET side of every citation
// edge; references on the SOURCE side. Low coverage here = sparse/empty
// citation graph = uniform PageRank downstream.

const withTitle   = entries.filter((e) => e.metadata?.title).length;
const withAuthors = entries.filter((e) => (e.metadata?.authors || []).length > 0).length;
const withRefs    = entries.filter((e) => (e.references || []).length > 0).length;

console.log('\n[test_extract] Metadata coverage (needed by heuristic.py citation matching):');
console.log(`  title:      ${withTitle}/${entries.length}`);
console.log(`  authors:    ${withAuthors}/${entries.length}`);
console.log(`  references: ${withRefs}/${entries.length}`);
if (withTitle < entries.length || withAuthors < entries.length) {
  console.warn('  WARNING: docs without title/authors can never RECEIVE a citation edge —');
  console.warn('           PageRank degrades toward uniform. Check _extract_metadata in extract.py.');
}

console.log('\nDone. Run test_categories.js next.');