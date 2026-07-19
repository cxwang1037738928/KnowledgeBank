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
export function stripJats(jatsMarkup) {
  if (!jatsMarkup) return null;
  const plainText = jatsMarkup
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^abstract[\s:.]+/i, '');
  return plainText || null;
}

const ROOT          = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const DATA_DIR      = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const CACHE_DIR     = path.join(ROOT, 'cache');
const META_PATH     = path.join(CACHE_DIR, 'meta.json');

const MAILTO          = process.env.CROSSREF_MAILTO || 'ericwang030@gmail.com';
const USER_AGENT      = `OpenCrawl/1.0 (mailto:${MAILTO})`;
const TIMEOUT_MS      = parseInt(process.env.CROSSREF_TIMEOUT_MS || '20000', 10);
const POLITE_DELAY_MS = parseInt(process.env.CROSSREF_DELAY_MS || '250', 10);

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

// ---------------------------------------------------------------------------
// Crossref response → normalized metadata
// ---------------------------------------------------------------------------

/** Map a Crossref `message` (work) object to our cached metadata shape. */
function normalizeWork(work) {
  const title = (Array.isArray(work.title) ? work.title[0] : work.title || '').trim() || null;

  const author = (work.author || [])
    .map((crossrefAuthor) => [crossrefAuthor.given, crossrefAuthor.family].filter(Boolean).join(' ').trim())
    .filter(Boolean);

  const reference = (work.reference || []).map((crossrefReference) => ({
    // Crossref reference entries frequently carry a DOI but no article title;
    // keep both so DOI-based matching stays possible even when title is blank.
    title:   (crossrefReference['article-title'] || crossrefReference['volume-title'] || '').trim(),
    authors: crossrefReference.author ? [String(crossrefReference.author).trim()] : [],
    doi:     crossrefReference.DOI || null,
    raw:     (crossrefReference.unstructured || '').trim(),
  }));

  const citedBy = Number.isFinite(work['is-referenced-by-count'])
    ? work['is-referenced-by-count']
    : null;

  // Publication date from `issued` date-parts [year, month, day]. Same
  // future-date cap as extract.py's text-scan fallback, for consistency.
  let created = null;
  const issuedDateParts = work.issued?.['date-parts']?.[0];
  if (Array.isArray(issuedDateParts) && Number.isFinite(issuedDateParts[0])) {
    const year  = issuedDateParts[0];
    const month = Number.isFinite(issuedDateParts[1]) ? issuedDateParts[1] : null;
    const now = new Date();
    const isFutureDate = year > now.getFullYear()
      || (year === now.getFullYear() && month !== null && month > now.getMonth() + 1);
    if (!isFutureDate) created = { year, month };
  }

  return {
    title,
    author,
    abstract: stripJats(work.abstract),
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
  const crossrefUrl = `https://api.crossref.org/works/${encodeURIComponent(doi)}?mailto=${encodeURIComponent(MAILTO)}`;
  const abortController = new AbortController();
  const timeoutTimer = setTimeout(() => abortController.abort(), TIMEOUT_MS);
  let response;
  try {
    response = await fetch(crossrefUrl, {
      headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
      signal: abortController.signal,
    });
  } finally {
    clearTimeout(timeoutTimer);
  }

  if (response.status === 404) return null;             // definitive miss
  if (!response.ok) throw new Error(`Crossref HTTP ${response.status} for ${doi}`);

  const body = await response.json();
  if (!body || !body.message) return null;
  return normalizeWork(body.message);
}

// ---------------------------------------------------------------------------
// Cache (cache/meta.json)
// ---------------------------------------------------------------------------

async function loadCache() {
  try {
    const cacheJson = await fs.readFile(META_PATH, 'utf-8');
    return cacheJson.trim() ? JSON.parse(cacheJson) : {};
  } catch {
    return {};
  }
}

async function saveCache(metaByDoi) {
  await fs.mkdir(CACHE_DIR, { recursive: true });
  // Temp + rename so a crash mid-write can't truncate the cache file.
  const tempPath = `${META_PATH}.tmp`;
  await fs.writeFile(tempPath, JSON.stringify(metaByDoi, null, 2), 'utf-8');
  await fs.rename(tempPath, META_PATH);
}

// ---------------------------------------------------------------------------
// Batch enrichment of doclings.json
// ---------------------------------------------------------------------------

/**
 * Overlay Crossref data on every DOI'd doclings.json entry. GROBID
 * parsedReferences stay intact — Crossref references often lack article
 * titles, which the citation matcher needs.
 */
export async function enrichDoclings(dataDir = DATA_DIR) {
  const doclingsPath = path.join(dataDir, 'doclings.json');
  let doclings;
  try {
    doclings = JSON.parse(await fs.readFile(doclingsPath, 'utf-8'));
  } catch {
    throw new Error(`${doclingsPath} not found — run extract.py first`);
  }

  const metaByDoi = await loadCache();
  let matchedCount = 0;
  let unmatchedCount = 0;
  let doiDocCount = 0;

  for (const doclingEntry of Object.values(doclings)) {
    const doi = doclingEntry?.metadata?.doi;
    if (!doi) continue;
    doiDocCount++;

    let crossrefMeta;
    if (Object.prototype.hasOwnProperty.call(metaByDoi, doi)) {
      crossrefMeta = metaByDoi[doi];           // cache hit (object or known miss)
    } else {
      try {
        crossrefMeta = await crossrefByDoi(doi);
      } catch (err) {
        console.warn(`[search_doi] ${doi}: ${err.message} — leaving previous metadata`);
        continue;
      }
      metaByDoi[doi] = crossrefMeta;
      await saveCache(metaByDoi);              // persist incrementally
      await sleep(POLITE_DELAY_MS);            // be polite between live calls
    }

    if (!crossrefMeta) { unmatchedCount++; continue; }   // no match → keep previous
    matchedCount++;

    doclingEntry.metadata = doclingEntry.metadata || {};
    if (crossrefMeta.title)                                    doclingEntry.metadata.title    = crossrefMeta.title;
    if (crossrefMeta.author && crossrefMeta.author.length)     doclingEntry.metadata.authors  = crossrefMeta.author;
    if (crossrefMeta.abstract)                                 doclingEntry.metadata.abstract = crossrefMeta.abstract;
    if (crossrefMeta.created?.year)                            doclingEntry.metadata.created  = crossrefMeta.created;
    if (crossrefMeta.citedBy != null)                          doclingEntry.metadata.citedBy  = crossrefMeta.citedBy;
    if (crossrefMeta.reference && crossrefMeta.reference.length) doclingEntry.crossrefReferences = crossrefMeta.reference;
  }

  await fs.writeFile(doclingsPath, JSON.stringify(doclings, null, 2), 'utf-8');
  console.log(`[search_doi] Crossref: ${matchedCount} matched, ${unmatchedCount} unmatched of ${doiDocCount} DOI'd doc(s) → ${doclingsPath}`);
  return doclings;
}

// Run directly: node backend/extraction/sapphire/search_doi.js
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  enrichDoclings().catch((err) => {
    console.error('[search_doi]', err.message);
    process.exit(1);
  });
}
