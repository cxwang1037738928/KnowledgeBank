import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import { createCanvas } from '@napi-rs/canvas';
import sharp from 'sharp';

// #TODO: make this run in parallel for multiple pages, and make deskew function lazier(more efficient)
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// pdfjs needs these asset dirs to render embedded/standard fonts and CJK text.
// Paths must use forward slashes with a trailing slash regardless of OS —
// pdfjs parses them as URL-like strings, not native fs paths.
const PDFJS_ROOT = path.dirname(fileURLToPath(import.meta.resolve('pdfjs-dist/package.json')));
const STANDARD_FONT_DATA_URL = path.join(PDFJS_ROOT, 'standard_fonts').replace(/\\/g, '/') + '/';
const CMAP_URL = path.join(PDFJS_ROOT, 'cmaps').replace(/\\/g, '/') + '/';

const DEFAULT_DPI = parseInt(process.env.RASTER_DPI || '300', 10);
const ENHANCED_DIR = path.resolve(process.env.ENHANCED_DIR || './data/enhanced');

// Chars per square point (page area in PDF points) above which a page is
// considered to have a real digital text layer rather than sparse/no text.
// Empirical cutoff — tune against your corpus rather than trusting blindly.
const TEXT_DENSITY_THRESHOLD = 0.0008;

export const PAGE_TYPE = {
  DIGITAL: 'digital',
  SCANNED: 'scanned',
  MIXED: 'mixed', // scanned image that already carries an OCR text layer
};

// ---------------------------------------------------------------------------
// 2.1 — Page type detection (digital vs scanned)
// ---------------------------------------------------------------------------

/**
 * Loads a PDF via pdfjs-dist for page-level inspection and rendering.
 * @param {string} pdfPath
 * @returns {Promise<import('pdfjs-dist').PDFDocumentProxy>}
 */
export async function loadDocument(pdfPath) {
  const data = new Uint8Array(await fs.readFile(pdfPath));
  return pdfjsLib.getDocument({
    data,
    standardFontDataUrl: STANDARD_FONT_DATA_URL,
    cMapUrl: CMAP_URL,
    cMapPacked: true,
    verbosity: pdfjsLib.VerbosityLevel.ERRORS,
  }).promise;
}

/**
 * Classifies a single page as digital, scanned, or mixed by combining the
 * extracted text density with whether the page draws any image XObject.
 * (We check for image *presence*, not full-page coverage — recovering the
 * exact transform matrix per paint op from pdfjs's operator list would add
 * real complexity for a marginal accuracy gain.)
 *
 * @param {import('pdfjs-dist').PDFDocumentProxy} doc
 * @param {number} pageNumber - 1-indexed
 * @returns {Promise<{ pageNumber: number, type: string, textDensity: number, hasImage: boolean, charCount: number }>}
 */
export async function classifyPageType(doc, pageNumber, { textDensityThreshold = TEXT_DENSITY_THRESHOLD } = {}) {
  const page = await doc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 1 });
  const areaPts = viewport.width * viewport.height;

  const textContent = await page.getTextContent();
  const text = textContent.items.map((item) => item.str).join('');
  const textDensity = text.length / areaPts;

  const opList = await page.getOperatorList();
  const hasImage = opList.fnArray.includes(pdfjsLib.OPS.paintImageXObject);

  let type;
  if (hasImage && textDensity < textDensityThreshold) type = PAGE_TYPE.SCANNED;
  else if (hasImage) type = PAGE_TYPE.MIXED;
  else if (textDensity >= textDensityThreshold) type = PAGE_TYPE.DIGITAL;
  else type = PAGE_TYPE.SCANNED;

  return { pageNumber, type, textDensity: parseFloat(textDensity.toFixed(6)), hasImage, charCount: text.length };
}

/**
 * Classifies every page in a document.
 * @param {string} pdfPath
 * @returns {Promise<Array<ReturnType<typeof classifyPageType>>>}
 */
export async function classifyAllPages(pdfPath) {
  const doc = await loadDocument(pdfPath);
  const results = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    results.push(await classifyPageType(doc, pageNum));
  }
  return results;
}

// ---------------------------------------------------------------------------
// 2.2 — Rasterization
// ---------------------------------------------------------------------------

/**
 * Rasterizes one PDF page to a PNG buffer at the given DPI.
 * @param {import('pdfjs-dist').PDFDocumentProxy} doc
 * @param {number} pageNumber - 1-indexed
 * @param {number} dpi
 * @returns {Promise<{ buffer: Buffer, width: number, height: number }>}
 */
export async function rasterizePage(doc, pageNumber, dpi = DEFAULT_DPI) {
  const page = await doc.getPage(pageNumber);
  const scale = dpi / 72; // PDF points are 1/72 inch
  const viewport = page.getViewport({ scale });

  const width = Math.ceil(viewport.width);
  const height = Math.ceil(viewport.height);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Explicit white fill — some PDFs render transparent regions otherwise,
  // which is the wrong default background for an OCR-bound scan.
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  await page.render({ canvasContext: ctx, viewport }).promise;

  return { buffer: canvas.toBuffer('image/png'), width, height };
}

/**
 * Rasterizes every page of a document.
 * @param {string} pdfPath
 * @param {number} dpi
 * @returns {Promise<Array<{ pageNumber: number, buffer: Buffer, width: number, height: number }>>}
 */
export async function rasterizeDocument(pdfPath, dpi = DEFAULT_DPI) {
  const doc = await loadDocument(pdfPath);
  const pages = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    pages.push({ pageNumber: pageNum, ...(await rasterizePage(doc, pageNum, dpi)) });
  }
  return pages;
}

// ---------------------------------------------------------------------------
// Raw pixel helpers
// ---------------------------------------------------------------------------

async function toGrayscaleRaw(buffer) {
  const { data, info } = await sharp(buffer).grayscale().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

function computeHistogram(grayData) {
  const hist = new Array(256).fill(0);
  for (let pixelIdx = 0; pixelIdx < grayData.length; pixelIdx++) hist[grayData[pixelIdx]]++;
  return hist;
}

/** Otsu's method: finds the threshold that maximizes between-class variance. */
function otsuThreshold(hist, total) {
  let sum = 0;
  for (let level = 0; level < 256; level++) sum += level * hist[level];

  let sumB = 0;
  let weightB = 0;
  let maxVariance = 0;
  let threshold = 0;

  for (let level = 0; level < 256; level++) {
    weightB += hist[level];
    if (weightB === 0) continue;
    const weightF = total - weightB;
    if (weightF === 0) break;

    sumB += level * hist[level];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const variance = weightB * weightF * (meanB - meanF) ** 2;

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = level;
    }
  }
  return threshold;
}

// ---------------------------------------------------------------------------
// 2.3 — Enhancement pipeline
// ---------------------------------------------------------------------------

/** Median filter — removes salt-and-pepper scanner noise. */
export async function denoise(buffer, { size = 3 } = {}) {
  return sharp(buffer).median(size).toBuffer();
}

/** Stretches the image's luminance histogram to use the full dynamic range. */
export async function normalizeContrast(buffer) {
  return sharp(buffer).normalize().toBuffer();
}

/**
 * Estimates page skew via the classic projection-profile method: rotate by
 * candidate angles and pick the one that maximizes variance of per-row ink
 * counts (text baselines line up into sharp peaks/valleys at the correct angle).
 *
 * @param {Buffer} buffer
 * @returns {Promise<number>} estimated skew angle in degrees
 */
export async function estimateSkewAngle(buffer, { maxAngle = 10, coarseStep = 1, fineStep = 0.1, searchWidth = 1000 } = {}) {
  const small = await sharp(buffer).grayscale().resize({ width: searchWidth, withoutEnlargement: true }).toBuffer();

  async function varianceAtAngle(angle) {
    const { data, info } = await sharp(small)
      .rotate(angle, { background: '#ffffff' })
      .raw()
      .toBuffer({ resolveWithObject: true });

    const rowSums = new Float64Array(info.height);
    for (let y = 0; y < info.height; y++) {
      let sum = 0;
      const rowStart = y * info.width;
      for (let x = 0; x < info.width; x++) {
        if (data[rowStart + x] < 128) sum++;
      }
      rowSums[y] = sum;
    }

    const mean = rowSums.reduce((total, rowSum) => total + rowSum, 0) / rowSums.length;
    return rowSums.reduce((total, rowSum) => total + (rowSum - mean) ** 2, 0) / rowSums.length;
  }

  let bestAngle = 0;
  let bestVariance = -Infinity;
  for (let angle = -maxAngle; angle <= maxAngle; angle += coarseStep) {
    const variance = await varianceAtAngle(angle);
    if (variance > bestVariance) {
      bestVariance = variance;
      bestAngle = angle;
    }
  }

  // No ink content (blank/fully-white page) — nothing to align.
  if (bestVariance <= 0) return 0;

  for (let angle = bestAngle - coarseStep; angle <= bestAngle + coarseStep; angle += fineStep) {
    const variance = await varianceAtAngle(angle);
    if (variance > bestVariance) {
      bestVariance = variance;
      bestAngle = angle;
    }
  }

  return parseFloat(bestAngle.toFixed(2));
}

/**
 * Corrects page rotation/tilt from scanning.
 * @param {Buffer} buffer
 * @returns {Promise<{ buffer: Buffer, angle: number }>}
 */
export async function deskew(buffer, opts = {}) {
  const angle = await estimateSkewAngle(buffer, opts);
  if (Math.abs(angle) < 0.05) return { buffer, angle: 0 };
  const rotated = await sharp(buffer).rotate(angle, { background: '#ffffff' }).toBuffer();
  return { buffer: rotated, angle };
}

/**
 * Converts to black-and-white using an Otsu-derived global threshold —
 * adapts to each page's contrast instead of a fixed cutoff.
 * @param {Buffer} buffer
 * @returns {Promise<{ buffer: Buffer, threshold: number }>}
 */
export async function binarize(buffer) {
  const { data } = await toGrayscaleRaw(buffer);
  const threshold = otsuThreshold(computeHistogram(data), data.length);
  const binarized = await sharp(buffer).threshold(threshold).toBuffer();
  return { buffer: binarized, threshold };
}

/**
 * Runs the full per-page enhancement pipeline in the standard order:
 * denoise → normalize contrast → deskew → binarize. Binarization is last
 * since OCR wants a final clean B&W image and rotating a binary image
 * (instead of grayscale) would introduce jagged edges.
 *
 * @param {Buffer} buffer - rasterized page image (color or grayscale)
 * @returns {Promise<{ buffer: Buffer, angle: number, threshold: number }>}
 */
export async function enhancePage(buffer, opts = {}) {
  const denoised = await denoise(buffer, opts.denoise);
  const normalized = await normalizeContrast(denoised);
  const { buffer: deskewed, angle } = await deskew(normalized, opts.deskew);
  const { buffer: binarized, threshold } = await binarize(deskewed);
  return { buffer: binarized, angle, threshold };
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/**
 * Runs the full Stage 2 pipeline for a single page:
 * classify → rasterize → enhance. Optionally persists the enhanced PNG.
 *
 * @param {import('pdfjs-dist').PDFDocumentProxy} doc
 * @param {number} pageNumber
 * @param {{ dpi?: number, saveDir?: string|null }} [opts]
 */
/**
 * Returns true if the buffer is >99% white pixels — indicates pdfjs failed
 * to decode the page content (most commonly: JBig2-compressed images).
 */
async function isBlankPage(buffer) {
  const { data, info } = await sharp(buffer).grayscale().raw().toBuffer({ resolveWithObject: true });
  const white = data.reduce((count, pixel) => count + (pixel > 250 ? 1 : 0), 0);
  return white / (info.width * info.height) > 0.99;
}

export async function processPage(doc, pageNumber, { dpi = DEFAULT_DPI, saveDir = null } = {}) {
  const pageType = await classifyPageType(doc, pageNumber);
  const { buffer: colorBuffer, width, height } = await rasterizePage(doc, pageNumber, dpi);
  const blank = await isBlankPage(colorBuffer);
  const enhanced = await enhancePage(colorBuffer);

  if (saveDir) {
    await fs.mkdir(saveDir, { recursive: true });
    await fs.writeFile(path.join(saveDir, `page_${pageNumber}.png`), enhanced.buffer);
  }

  return {
    pageNumber,
    pageType: pageType.type,
    textDensity: pageType.textDensity,
    dimensions: { width, height, dpi },
    enhancement: { deskewAngle: enhanced.angle, binarizationThreshold: enhanced.threshold },
    blank,
  };
}

/**
 * Runs processPage() across an entire document. Pages are processed
 * sequentially — rasterizing at 300 DPI is memory-heavy and a long PDF
 * processed page-by-page keeps peak memory bounded.
 *
 * @param {string} pdfPath
 * @param {{ docId?: string, dpi?: number }} [opts]
 */
export async function processDocument(pdfPath, { docId, dpi = DEFAULT_DPI } = {}) {
  const doc = await loadDocument(pdfPath);
  const saveDir = docId ? path.join(ENHANCED_DIR, docId) : null;

  const pages = [];
  for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
    pages.push(await processPage(doc, pageNum, { dpi, saveDir }));
  }

  const report = { docId: docId ?? null, pdfPath, numPages: doc.numPages, dpi, processedAt: new Date().toISOString(), pages };

  const blankPages = pages.filter(page => page.blank).map(page => page.pageNumber);
  if (blankPages.length > 0) {
    console.warn(
      `[enhance] ${path.basename(pdfPath)}: ${blankPages.length}/${doc.numPages} page(s) rendered blank` +
      ` (pages ${blankPages.join(', ')}) — likely JBig2-compressed images pdfjs cannot decode.` +
      ` docling will extract these pages directly from the PDF.`
    );
  }

  if (docId) {
    await fs.mkdir(ENHANCED_DIR, { recursive: true });
    await fs.writeFile(path.join(ENHANCED_DIR, `${docId}.json`), JSON.stringify(report, null, 2));
  }

  return report;
}
