/**
 * regex_utils.js — shared regex patterns and the small pure helpers built on
 * them, for the JS side of the pipeline.
 *
 * Consumers:
 *   doi_regex.js          — DOI_RE / findDoi
 *   search_doi.js         — stripJats
 *   bootstrap_queries.js  — STOPWORDS / tokenise / REF_HEADINGS / normHeading
 *
 * Several patterns deliberately mirror Python counterparts (extract.py /
 * heuristic_utils.py) — Python cannot import from this file, so the two
 * copies must be kept in sync by hand:
 *   normHeading   ↔ extract.py _norm_heading / heuristic.py _norm_heading
 *   REF_HEADINGS  ↔ extract.py _REF_SECTION_HEADINGS / heuristic.py _REF_HEADINGS
 *   tokenise      ↔ heuristic_utils.py tokenise (same stopword list)
 */

// ---------------------------------------------------------------------------
// DOI
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Section headings
// ---------------------------------------------------------------------------

export const REF_HEADINGS = new Set([
  'references', 'bibliography', 'works cited', 'literature cited', 'citations',
]);

/**
 * Normalize a section heading for matching: lowercase, strip leading
 * numbering ('7. References', 'VII) References') and trailing punctuation.
 * Mirror of extract.py's _norm_heading — without it, numbered bibliography
 * headings slip past exact-string membership checks.
 */
export function normHeading(text) {
  return (text || '')
    .toLowerCase()
    .trim()
    .replace(/^[\divxlc]+[.)]?\s+/, '')
    .replace(/[\s.:]+$/, '');
}

// ---------------------------------------------------------------------------
// Tokenisation (mirrors heuristic_utils.py so keywords agree across stages)
// ---------------------------------------------------------------------------

export const STOPWORDS = new Set([
  'the','a','an','and','or','but','in','on','at','to','for','of','with',
  'by','from','as','is','was','are','were','be','been','being','have',
  'has','had','do','does','did','will','would','could','should','may',
  'might','this','that','these','those','it','its','i','we','you','he',
  'she','they','their','our','us','not','no','so','if','than','then',
]);

export function tokenise(text) {
  const tokens = (text.toLowerCase().match(/[a-z]+/g) || []);
  return tokens.filter((t) => !STOPWORDS.has(t) && t.length > 2);
}

// ---------------------------------------------------------------------------
// Crossref / JATS
// ---------------------------------------------------------------------------

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
