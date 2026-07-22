/**
 * clean_pdf.js — PDF ingestion: validate → hash-derived docId → persist to
 * data/documents.json. Crawler-agnostic: any PDF enters the pipeline here.
 */

import 'dotenv/config';
import fs from 'fs/promises';
import { createReadStream } from 'fs';
import crypto from 'crypto';
import path from 'path';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MAX_FILE_SIZE = parseInt(process.env.MAX_PDF_SIZE_MB || '50', 10) * 1024 * 1024;
const DOCUMENTS_DIR = path.resolve(process.env.DOCUMENTS_DIR || './documents');
const META_PATH = path.resolve(process.env.DOCUMENTS_META_PATH || './data/documents.json');

// Status values a document moves through
export const STATUS = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
};

// ---------------------------------------------------------------------------
// Metadata store  (data/documents.json)
// ---------------------------------------------------------------------------

async function readMeta() {
  try {
    const raw = await fs.readFile(META_PATH, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return { documents: {} };
  }
}

async function writeMeta(store) {
  await fs.mkdir(path.dirname(META_PATH), { recursive: true });
  // Write-to-temp + rename: a crash mid-write can never leave a truncated
  // documents.json behind (rename is atomic on the same volume).
  const tempPath = `${META_PATH}.tmp`;
  await fs.writeFile(tempPath, JSON.stringify(store, null, 2), 'utf-8');
  await fs.rename(tempPath, META_PATH);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Streams a file through SHA-256 — never loads the whole file into memory.
 * @param {string} filePath
 * @returns {Promise<string>} hex digest
 */
function computeHash(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

/**
 * Reads the first 4 bytes to confirm the file starts with %PDF.
 * @param {string} filePath
 * @returns {Promise<boolean>}
 */
async function hasPDFMagicBytes(filePath) {
  let handle;
  try {
    handle = await fs.open(filePath, 'r');
    const headerBytes = Buffer.alloc(4);
    await handle.read(headerBytes, 0, 4, 0);
    return headerBytes.toString('ascii') === '%PDF';
  } catch {
    return false;
  } finally {
    await handle?.close();
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Validates a PDF (existence, extension, size, magic bytes, duplicate hash)
 * without modifying any state.
 * @param {string} filePath - absolute path to the file
 * @returns {Promise<{ valid: boolean, error?: string, hash?: string, fileSize?: number }>}
 */
export async function validatePDF(filePath) {
  try {
    await fs.access(filePath, fs.constants.R_OK);
  } catch {
    return { valid: false, error: 'File not found or not readable.' };
  }

  if (path.extname(filePath).toLowerCase() !== '.pdf') {
    return { valid: false, error: 'File does not have a .pdf extension.' };
  }

  const stat = await fs.stat(filePath);
  if (stat.size === 0) {
    return { valid: false, error: 'File is empty.' };
  }
  if (stat.size > MAX_FILE_SIZE) {
    const sizeMb = (stat.size / 1024 / 1024).toFixed(1);
    return { valid: false, error: `File size ${sizeMb} MB exceeds the ${process.env.MAX_PDF_SIZE_MB || 50} MB limit.` };
  }

  const isRealPDF = await hasPDFMagicBytes(filePath);
  if (!isRealPDF) {
    return { valid: false, error: 'File does not appear to be a valid PDF (bad magic bytes).' };
  }

  const hash = await computeHash(filePath);
  const store = await readMeta();
  const duplicate = Object.values(store.documents).find((doc) => doc.sha256 === hash);
  if (duplicate) {
    return {
      valid: false,
      error: `Duplicate file — already ingested as doc_id "${duplicate.docId}" (${duplicate.filename}).`,
      hash,
      fileSize: stat.size,
    };
  }

  return { valid: true, hash, fileSize: stat.size };
}

/**
 * Page count via pdf-parse (doubles as a deeper corruption check). The PDF
 * Info dictionary's Title/Author are junk-prone and unused — authoritative
 * metadata comes from extraction. Null when the PDF cannot be parsed.
 * @param {string} filePath
 * @returns {Promise<{ pageCount: number } | null>}
 */
export async function extractPDFMeta(filePath) {
  try {
    const buffer = await fs.readFile(filePath);
    const result = await pdfParse(buffer, { max: 0 }); // metadata only, skip full text
    return { pageCount: result.numpages };
  } catch {
    return null;
  }
}

/**
 * Ingest one PDF: validate → docId from SHA-256 (stable across re-uploads) →
 * persist to documents.json.
 * @param {string} filePath - absolute path to the PDF
 * @returns {Promise<{ docId: string, status: string } | { error: string }>}
 */
export async function ingestDocument(filePath) {
  const validation = await validatePDF(filePath);
  if (!validation.valid) {
    return { error: validation.error };
  }

  const pdfMeta = await extractPDFMeta(filePath);
  if (!pdfMeta) {
    return { error: 'PDF appears corrupted — could not extract page metadata.' };
  }

  const docId = validation.hash.slice(0, 16);
  const now = new Date().toISOString();

  /** @type {DocRecord} */
  const record = {
    docId,
    filename: path.basename(filePath),
    filePath,
    fileSize: validation.fileSize,
    sha256: validation.hash,
    pageCount: pdfMeta.pageCount,
    uploadedAt: now,
    status: STATUS.PENDING,
    statusUpdatedAt: now,
    statusHistory: [{ status: STATUS.PENDING, at: now }],
    error: null,
  };

  const store = await readMeta();
  store.documents[docId] = record;
  await writeMeta(store);

  return { docId, status: STATUS.PENDING };
}

/**
 * Updates the status of a document in documents.json as it moves through
 * processing stages.
 *
 * @param {string} docId
 * @param {string} status - one of STATUS.*
 * @param {{ error?: string }} [extra]
 * @returns {Promise<DocRecord | null>}
 */
export async function updateDocumentStatus(docId, status, extra = {}) {
  const store = await readMeta();
  const record = store.documents[docId];
  if (!record) return null;

  const now = new Date().toISOString();
  record.status = status;
  record.statusUpdatedAt = now;
  record.statusHistory.push({ status, at: now });
  if (extra.error !== undefined) record.error = extra.error;

  await writeMeta(store);
  return record;
}

/**
 * Returns the full record for a single document, or null if not found.
 * @param {string} docId
 * @returns {Promise<DocRecord | null>}
 */
export async function getDocumentStatus(docId) {
  const store = await readMeta();
  return store.documents[docId] ?? null;
}

/**
 * Returns all document records, optionally filtered by status.
 * @param {{ status?: string }} [filter]
 * @returns {Promise<DocRecord[]>}
 */
export async function listDocuments(filter = {}) {
  const store = await readMeta();
  let docs = Object.values(store.documents);
  if (filter.status) {
    docs = docs.filter((doc) => doc.status === filter.status);
  }
  return docs.sort((docA, docB) => docB.uploadedAt.localeCompare(docA.uploadedAt));
}

/**
 * Scans DOCUMENTS_DIR, ingests every PDF not already tracked, and returns a
 * per-file result summary.
 *
 * @returns {Promise<Array<{ filename: string, docId?: string, status?: string, error?: string }>>}
 */
export async function scanAndIngestDirectory() {
  let entries;
  try {
    entries = await fs.readdir(DOCUMENTS_DIR);
  } catch {
    return [{ filename: DOCUMENTS_DIR, error: 'Documents directory not found or not readable.' }];
  }

  const pdfs = entries.filter((entry) => path.extname(entry).toLowerCase() === '.pdf');
  if (pdfs.length === 0) return [];

  // Sequential on purpose: ingestDocument rewrites documents.json whole, so
  // concurrent ingests are a lost-update race.
  const results = [];
  for (const filename of pdfs) {
    try {
      const value = await ingestDocument(path.join(DOCUMENTS_DIR, filename));
      results.push({ filename, ...value });
    } catch (err) {
      results.push({ filename, error: err?.message ?? 'Unknown error' });
    }
  }
  return results;
}

/**
 * @typedef {object} DocRecord
 * @property {string}   docId
 * @property {string}   filename
 * @property {string}   filePath
 * @property {number}   fileSize
 * @property {string}   sha256
 * @property {number}   pageCount
 * @property {string}   uploadedAt
 * @property {string}   status        - pending | processing | completed | failed
 * @property {string}   statusUpdatedAt
 * @property {Array<{status: string, at: string}>} statusHistory
 * @property {string|null} error
 */
