/**
 * test_cleaning.js — pipeline stage 1: ingest + enhance
 *
 * Exercises clean_pdf.js and enhance_pdf.js on every PDF in tests/test-input/.
 * Run: node tests/test_cleaning.js
 *
 * CHANGED to match the current pipeline: enhancement output now goes to
 * data/enhanced/ — NOT tests/test-output/enhanced/. extract.py's
 * _choose_converter() reads data/enhanced/<docId>.json to decide between
 * the digital and OCR converters; when the reports lived under
 * tests/test-output/, extract.py never found them and silently routed every
 * document through the (much slower) OCR pipeline. The per-doc JSON reports
 * are additionally COPIED into tests/test-output/enhanced/ for inspection.
 *
 * Outputs:
 *   data/documents.json                     — ingest metadata (needed by extract.py)
 *   data/enhanced/<docId>.json + page PNGs  — pipeline artifacts (read by extract.py)
 *   tests/test-output/enhanced/<docId>.json — report copies for inspection
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT         = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TEST_INPUT   = path.join(ROOT, 'tests', 'test-input');
const TEST_OUTPUT  = path.join(ROOT, 'tests', 'test-output');
const ENHANCED_DIR = path.join(ROOT, 'data', 'enhanced');

// Must be set before enhance_pdf.js is imported — it reads ENHANCED_DIR at
// module load time. Points at data/enhanced/ so extract.py can find reports.
process.env.ENHANCED_DIR = ENHANCED_DIR;

const { ingestDocument }  = await import('../backend/parser/cleaning/clean_pdf.js');
const { processDocument } = await import('../backend/parser/cleaning/enhance_pdf.js');

await fs.mkdir(ENHANCED_DIR, { recursive: true });
await fs.mkdir(path.join(TEST_OUTPUT, 'enhanced'), { recursive: true });

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
  const ingestResult = await ingestDocument(filePath);

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

    // Copy the JSON report into test-output for inspection; the original
    // stays in data/enhanced/ where extract.py expects it.
    const reportSrc = path.join(ENHANCED_DIR, `${docId}.json`);
    try {
      await fs.copyFile(reportSrc, path.join(TEST_OUTPUT, 'enhanced', `${docId}.json`));
    } catch { /* report path may differ if enhance_pdf writes elsewhere */ }

    console.log(`  [enhance] Report  → data/enhanced/${docId}.json (copy in tests/test-output/enhanced/)`);
    console.log(`  [enhance] Images  → data/enhanced/${docId}/page_*.png`);
  } catch (err) {
    console.log(`  [enhance] ERROR: ${err.message}`);
  }

  console.log();
}

console.log('Done. Run test_extract.js next.');