/**
 * search_doi.js — enrich document metadata from the Crossref REST API by DOI.
 *
 * When a document has a DOI (stamped by doi_regex.js), Crossref is the
 * authoritative source for its title, authors, abstract, reference list, and
 * cited-by count — cleaner than anything recovered from the PDF. When the DOI
 * is absent or Crossref has no match, we resort to the previous values
 * (GROBID / docling) already on the document.
 *
 * Politeness: every request joins Crossref's "polite pool" by advertising a
 * contact email (CROSSREF_MAILTO) both as a ?mailto= query parameter and in
 * the User-Agent header, per Crossref's etiquette guidelines.
 *
 * Caching: results are cached at cache/meta.json, keyed by DOI, so a given DOI
 * is fetched at most once across all runs. Each cached object holds:
 *   { title, author, abstract, created, reference, citedBy, fetchedAt }
 * (created = {year, month|null} from `issued`; DOIs cached before this field
 * existed need a cache entry delete to pick it up.)
 * A DOI Crossref does not know is cached as null (a "known miss") so it is not
 * retried on every run. Transient network/HTTP errors are NOT cached.
 *
 * Exports:
 *   crossrefByDoi(doi)  — live fetch + normalize; null on definitive miss
 *   getMeta(doi)        — cache-aware single lookup (reads/writes meta.json)
 *   enrichDoclings()    — apply Crossref data to every DOI'd doc in doclings.json
 *
 * Run directly: node backend/extraction/sapphire/search_doi.js
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

/**
 * Strip JATS/XML markup from a Crossref abstract; null when empty. Crossref
 * abstracts typically start with a literal 'Abstract' heading element — after
 * tag stripping that word leaks in as a prefix, so it is removed too.
 */
export function stripJats(x) {
  if (!x) return null;
  const text = x
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^abstract[\s:.]+/i, '');
  return text || null;
}

const ROOT          = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const DATA_DIR      = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const DOCLINGS_PATH = path.join(DATA_DIR, 'doclings.json');
const CACHE_DIR     = path.join(ROOT, 'cache');
const META_PATH     = path.join(CACHE_DIR, 'meta.json');

const MAILTO          = process.env.CROSSREF_MAILTO || 'ericwang030@gmail.com';
const USER_AGENT      = `OpenCrawl/1.0 (mailto:${MAILTO})`;
const TIMEOUT_MS      = parseInt(process.env.CROSSREF_TIMEOUT_MS || '20000', 10);
const POLITE_DELAY_MS = parseInt(process.env.CROSSREF_DELAY_MS || '250', 10);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Crossref response → normalized metadata
// ---------------------------------------------------------------------------

/** Map a Crossref `message` (work) object to our cached metadata shape. */
function normalizeWork(m) {
  const title = (Array.isArray(m.title) ? m.title[0] : m.title || '').trim() || null;

  const author = (m.author || [])
    .map((a) => [a.given, a.family].filter(Boolean).join(' ').trim())
    .filter(Boolean);

  const reference = (m.reference || []).map((r) => ({
    // Crossref reference entries frequently carry a DOI but no article title;
    // keep both so DOI-based matching stays possible even when title is blank.
    title:   (r['article-title'] || r['volume-title'] || '').trim(),
    authors: r.author ? [String(r.author).trim()] : [],
    doi:     r.DOI || null,
    raw:     (r.unstructured || '').trim(),
  }));

  const citedBy = Number.isFinite(m['is-referenced-by-count'])
    ? m['is-referenced-by-count']
    : null;

  // Publication date from `issued` date-parts [year, month, day]. Same
  // future-date cap as extract.py's text-scan fallback, for consistency.
  let created = null;
  const parts = m.issued?.['date-parts']?.[0];
  if (Array.isArray(parts) && Number.isFinite(parts[0])) {
    const year  = parts[0];
    const month = Number.isFinite(parts[1]) ? parts[1] : null;
    const now = new Date();
    const future = year > now.getFullYear()
      || (year === now.getFullYear() && month !== null && month > now.getMonth() + 1);
    if (!future) created = { year, month };
  }

  return {
    title,
    author,
    abstract: stripJats(m.abstract),
    created,
    reference,
    citedBy,
    fetchedAt: new Date().toISOString(),
  };
}

/**
 * Fetch one work from Crossref by DOI. Returns the normalized object, or null
 * when Crossref definitively has no such DOI (404 / empty message). Throws on
 * transient failures (timeout, 5xx, network) so callers don't cache a miss.
 */
export async function crossrefByDoi(doi) {
  const url = `https://api.crossref.org/works/${encodeURIComponent(doi)}?mailto=${encodeURIComponent(MAILTO)}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let resp;
  try {
    resp = await fetch(url, {
      headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  if (resp.status === 404) return null;                 // definitive miss
  if (!resp.ok) throw new Error(`Crossref HTTP ${resp.status} for ${doi}`);

  const body = await resp.json();
  if (!body || !body.message) return null;
  return normalizeWork(body.message);
}

// ---------------------------------------------------------------------------
// Cache (cache/meta.json)
// ---------------------------------------------------------------------------

async function loadCache() {
  try {
    const raw = await fs.readFile(META_PATH, 'utf-8');
    return raw.trim() ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

async function saveCache(cache) {
  await fs.mkdir(CACHE_DIR, { recursive: true });
  // Temp + rename so a crash mid-write can't truncate the cache file.
  const tmp = `${META_PATH}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(cache, null, 2), 'utf-8');
  await fs.rename(tmp, META_PATH);
}

/**
 * Cache-aware single-DOI lookup. Returns the cached/fetched metadata object,
 * or null (a known or fresh miss). Transient errors return null WITHOUT
 * caching, so the DOI is retried on a later run.
 */
export async function getMeta(doi) {
  if (!doi) return null;
  const cache = await loadCache();
  if (Object.prototype.hasOwnProperty.call(cache, doi)) return cache[doi];

  let meta;
  try {
    meta = await crossrefByDoi(doi);
  } catch (err) {
    console.warn(`[search_doi] ${doi}: ${err.message} — not cached`);
    return null;
  }
  cache[doi] = meta;            // object, or null for a known miss
  await saveCache(cache);
  return meta;
}

// ---------------------------------------------------------------------------
// Batch enrichment of doclings.json
// ---------------------------------------------------------------------------

/**
 * For every document in doclings.json that has metadata.doi, overlay Crossref
 * data: title / authors / abstract / citedBy replace the PDF-derived values
 * when Crossref provides them (otherwise the previous value is kept), and the
 * Crossref reference list is attached as `crossrefReferences`. The GROBID
 * `parsedReferences` used for title-based citation matching are left intact,
 * since Crossref references often lack article titles.
 */
export async function enrichDoclings() {
  let doclings;
  try {
    doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }

  const cache = await loadCache();
  let hits = 0;
  let misses = 0;
  let withDoi = 0;

  for (const entry of Object.values(doclings)) {
    const doi = entry?.metadata?.doi;
    if (!doi) continue;
    withDoi++;

    let meta;
    if (Object.prototype.hasOwnProperty.call(cache, doi)) {
      meta = cache[doi];                       // cache hit (object or known miss)
    } else {
      try {
        meta = await crossrefByDoi(doi);
      } catch (err) {
        console.warn(`[search_doi] ${doi}: ${err.message} — leaving previous metadata`);
        continue;
      }
      cache[doi] = meta;
      await saveCache(cache);                  // persist incrementally
      await sleep(POLITE_DELAY_MS);            // be polite between live calls
    }

    if (!meta) { misses++; continue; }         // no Crossref match → keep previous
    hits++;

    entry.metadata = entry.metadata || {};
    if (meta.title)                        entry.metadata.title    = meta.title;
    if (meta.author && meta.author.length) entry.metadata.authors  = meta.author;
    if (meta.abstract)                     entry.metadata.abstract = meta.abstract;
    if (meta.created?.year)                entry.metadata.created  = meta.created;
    if (meta.citedBy != null)              entry.metadata.citedBy  = meta.citedBy;
    if (meta.reference && meta.reference.length) entry.crossrefReferences = meta.reference;
  }

  await fs.writeFile(DOCLINGS_PATH, JSON.stringify(doclings, null, 2), 'utf-8');
  console.log(`[search_doi] Crossref: ${hits} matched, ${misses} unmatched of ${withDoi} DOI'd doc(s) → ${DOCLINGS_PATH}`);
  return doclings;
}

// Run directly: node backend/extraction/sapphire/search_doi.js
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  enrichDoclings().catch((err) => {
    console.error('[search_doi]', err.message);
    process.exit(1);
  });
}
