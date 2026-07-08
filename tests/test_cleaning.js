/**
 * test_cleaning.js — pipeline stage 1: ingest + enhance
 *
 * Exercises clean_pdf.js and enhance_pdf.js on every PDF in tests/test-input/.
 * All outputs go to tests/test-output/ — no writes to data/.
 *
 * Run: node tests/test_cleaning.js
 *
 * Outputs:
 *   tests/test-output/documents.json          — ingest metadata (read by test_extract.js)
 *   tests/test-output/enhanced/<docId>.json   — page-type report (read by extract.py)
 *   tests/test-output/enhanced/<docId>/page_*.png
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT      = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_DATA = path.join(ROOT, 'tests', 'test-output');
const TEST_INPUT = path.join(ROOT, 'tests', 'test-input');

// Must be set before imports — clean_pdf.js and enhance_pdf.js read these at
// module load time.
process.env.DATA_DIR           = TEST_DATA;
process.env.DOCUMENTS_META_PATH = path.join(TEST_DATA, 'documents.json');
process.env.ENHANCED_DIR        = path.join(TEST_DATA, 'enhanced');

const { ingestDocument }  = await import('../backend/parser/cleaning/clean_pdf.js');
const { processDocument } = await import('../backend/parser/cleaning/enhance_pdf.js');

await fs.mkdir(path.join(TEST_DATA, 'enhanced'), { recursive: true });

// ---- Find input PDFs -------------------------------------------------------

let entries;
try {
  entries = await fs.readdir(TEST_INPUT);
} catch {
  console.error(`test-input directory not found: ${TEST_INPUT}`);
  process.exit(1);
}

const pdfs = entries.filter((e) => e.toLowerCase().endsWith('.pdf'));
if (pdfs.length === 0) {
  console.log('No PDFs found in tests/test-input/ — add some and re-run.');
  process.exit(0);
}

console.log(`Found ${pdfs.length} PDF(s) in tests/test-input/\n`);

// ---- Process each PDF ------------------------------------------------------

for (const filename of pdfs) {
  const filePath = path.join(TEST_INPUT, filename);
  console.log(`=== ${filename} ===`);

  // 1. Ingest
  const ingestResult = await ingestDocument(filePath, { enqueue: false });

  let docId;
  if (ingestResult.error) {
    // Duplicate — extract the existing docId from the error message and continue
    const match = ingestResult.error.match(/doc_id "([0-9a-f]+)"/);
    if (match) {
      docId = match[1];
      console.log(`  [ingest]  Already ingested — docId=${docId} (skipping re-ingest)`);
    } else {
      console.log(`  [ingest]  ERROR: ${ingestResult.error}`);
      console.log();
      continue;
    }
  } else {
    docId = ingestResult.docId;
    console.log(`  [ingest]  docId=${docId}  status=${ingestResult.status}`);
  }

  // 2. Enhance (rasterize + denoise + deskew + binarize)
  console.log(`  [enhance] Processing at 300 DPI — this takes a moment per page...`);
  try {
    const report = await processDocument(filePath, { docId, dpi: 300 });

    const scanned = report.pages.filter((p) => p.pageType !== 'digital').length;
    const route   = report.pages.length && scanned / report.pages.length < 0.3 ? 'digital' : 'ocr';
    console.log(`  [enhance] ${report.numPages} page(s) done — ${scanned} scanned/mixed → extract.py will use the ${route.toUpperCase()} converter`);

    for (const page of report.pages) {
      console.log(
        `    page ${String(page.pageNumber).padStart(2)}: ` +
        `type=${page.pageType.padEnd(7)}  ` +
        `deskew=${String(page.enhancement.deskewAngle).padStart(5)}°  ` +
        `otsu=${String(page.enhancement.binarizationThreshold).padStart(3)}  ` +
        `${page.dimensions.width}x${page.dimensions.height}px`
      );
    }

    console.log(`  [enhance] Report  → tests/test-output/enhanced/${docId}.json`);
    console.log(`  [enhance] Images  → tests/test-output/enhanced/${docId}/page_*.png`);
  } catch (err) {
    console.log(`  [enhance] ERROR: ${err.message}`);
  }

  console.log();
}

console.log('Done. Run test_extract.js next.');
