import 'dotenv/config';
import fs from 'fs/promises';
import { createReadStream } from 'fs';
import crypto from 'crypto';
import path from 'path';
import { Queue } from 'bullmq';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MAX_FILE_SIZE = parseInt(process.env.MAX_PDF_SIZE_MB || '50', 10) * 1024 * 1024;
const DOCUMENTS_DIR = path.resolve(process.env.DOCUMENTS_DIR || './documents');
const META_PATH = path.resolve(process.env.DOCUMENTS_META_PATH || './data/documents.json');

const REDIS_CONN = {
  host: process.env.REDIS_HOST || '127.0.0.1',
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
  ...(process.env.REDIS_PASSWORD ? { password: process.env.REDIS_PASSWORD } : {}),
  // Don't retry — fail fast so the catch block in ingestDocument can close the
  // queue before ioredis spams ECONNREFUSED errors across all subsequent docs.
  retryStrategy: () => null,
};

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
  await fs.writeFile(META_PATH, JSON.stringify(store, null, 2), 'utf-8');
}

// ---------------------------------------------------------------------------
// Task queue
// ---------------------------------------------------------------------------

let _queue = null;

function getPDFQueue() {
  if (!_queue) {
    _queue = new Queue('pdf-processing', { connection: REDIS_CONN });
  }
  return _queue;
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
    const buf = Buffer.alloc(4);
    await handle.read(buf, 0, 4, 0);
    return buf.toString('ascii') === '%PDF';
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
 * Validates a PDF file without modifying any state.
 *
 * Checks:
 *  - File exists and is readable
 *  - Extension is .pdf
 *  - File size is within MAX_FILE_SIZE
 *  - Magic bytes confirm it is a real PDF (not a renamed file)
 *  - SHA-256 hash is not already present in the metadata store (duplicate guard)
 *
 * @param {string} filePath - absolute path to the file
 * @returns {Promise<{ valid: boolean, error?: string, hash?: string, fileSize?: number }>}
 */
export async function validatePDF(filePath) {
  // Existence + readability
  try {
    await fs.access(filePath, fs.constants.R_OK);
  } catch {
    return { valid: false, error: 'File not found or not readable.' };
  }

  // Extension
  if (path.extname(filePath).toLowerCase() !== '.pdf') {
    return { valid: false, error: 'File does not have a .pdf extension.' };
  }

  // File size
  const stat = await fs.stat(filePath);
  if (stat.size === 0) {
    return { valid: false, error: 'File is empty.' };
  }
  if (stat.size > MAX_FILE_SIZE) {
    const mb = (stat.size / 1024 / 1024).toFixed(1);
    return { valid: false, error: `File size ${mb} MB exceeds the ${process.env.MAX_PDF_SIZE_MB || 50} MB limit.` };
  }

  // Magic bytes
  const isRealPDF = await hasPDFMagicBytes(filePath);
  if (!isRealPDF) {
    return { valid: false, error: 'File does not appear to be a valid PDF (bad magic bytes).' };
  }

  // Duplicate detection
  const hash = await computeHash(filePath);
  const store = await readMeta();
  const duplicate = Object.values(store.documents).find((d) => d.sha256 === hash);
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
 * Extracts PDF metadata using pdf-parse.
 * Returns null if the PDF cannot be parsed (treat as corrupted).
 *
 * @param {string} filePath
 * @returns {Promise<{ pageCount: number, title: string|null, author: string|null } | null>}
 */
export async function extractPDFMeta(filePath) {
  try {
    const buffer = await fs.readFile(filePath);
    const result = await pdfParse(buffer, { max: 0 }); // max:0 = parse metadata only, skip full text
    return {
      pageCount: result.numpages,
      title: result.info?.Title || null,
      author: result.info?.Author || null,
    };
  } catch {
    return null;
  }
}

/**
 * Full ingestion pipeline for a single PDF:
 *   validate → assign doc_id → extract metadata → persist to documents.json → enqueue
 *
 * The doc_id is derived from the file's SHA-256 hash so it is stable — the same
 * physical file always produces the same ID regardless of filename or upload time.
 *
 * @param {string} filePath - absolute path to the PDF
 * @returns {Promise<{ docId: string, status: string } | { error: string }>}
 */
export async function ingestDocument(filePath, { enqueue = true } = {}) {
  // 1. Validate
  const validation = await validatePDF(filePath);
  if (!validation.valid) {
    return { error: validation.error };
  }

  // 2. Extract PDF metadata (also serves as a deeper corruption check)
  const pdfMeta = await extractPDFMeta(filePath);
  if (!pdfMeta) {
    return { error: 'PDF appears corrupted — could not extract page metadata.' };
  }

  // 3. Build the doc record
  const docId = validation.hash.slice(0, 16); // first 16 hex chars of SHA-256
  const now = new Date().toISOString();

  /** @type {DocRecord} */
  const record = {
    docId,
    filename: path.basename(filePath),
    filePath,
    fileSize: validation.fileSize,
    sha256: validation.hash,
    pageCount: pdfMeta.pageCount,
    title: pdfMeta.title,
    author: pdfMeta.author,
    uploadedAt: now,
    status: STATUS.PENDING,
    statusUpdatedAt: now,
    statusHistory: [{ status: STATUS.PENDING, at: now }],
    error: null,
  };

  // 4. Persist to documents.json
  const store = await readMeta();
  store.documents[docId] = record;
  await writeMeta(store);

  // 5. Enqueue — skipped when enqueue:false (e.g. tests driving the pipeline directly)
  if (enqueue) {
    try {
      const queue = getPDFQueue();
      await queue.add('parse', { docId, filePath }, { jobId: docId, removeOnComplete: 100, removeOnFail: 200 });
    } catch (err) {
      console.warn(`[clean_pdf] Could not enqueue doc_id="${docId}": ${err.message}. Is Redis running?`);
      if (_queue) { await _queue.close().catch(() => {}); _queue = null; }
    }
  }

  return { docId, status: STATUS.PENDING };
}

/**
 * Updates the status of a document in documents.json.
 * Used by the queue worker as it moves through processing stages.
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
    docs = docs.filter((d) => d.status === filter.status);
  }
  return docs.sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));
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

  const pdfs = entries.filter((f) => path.extname(f).toLowerCase() === '.pdf');
  if (pdfs.length === 0) return [];

  const results = await Promise.allSettled(
    pdfs.map((filename) => ingestDocument(path.join(DOCUMENTS_DIR, filename)))
  );

  return results.map((r, i) =>
    r.status === 'fulfilled'
      ? { filename: pdfs[i], ...r.value }
      : { filename: pdfs[i], error: r.reason?.message ?? 'Unknown error' }
  );
}

/**
 * @typedef {object} DocRecord
 * @property {string}   docId
 * @property {string}   filename
 * @property {string}   filePath
 * @property {number}   fileSize
 * @property {string}   sha256
 * @property {number}   pageCount
 * @property {string|null} title
 * @property {string|null} author
 * @property {string}   uploadedAt
 * @property {string}   status        - pending | processing | completed | failed
 * @property {string}   statusUpdatedAt
 * @property {Array<{status: string, at: string}>} statusHistory
 * @property {string|null} error
 */
