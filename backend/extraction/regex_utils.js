/**
 * regex_utils.js — shared regex patterns and the small pure helpers built on
 * them, usable by every crawler (sapphire/ruby/topaz alike).
 *
 * Consumers:
 *   generate_categories.js       — STOPWORDS / tokenise / REF_HEADINGS / normHeading
 *   sapphire/build_graph.js      — REF_HEADINGS / normHeading
 *
 * Academic-only patterns (DOI matching, JATS stripping) live with their sole
 * consumers in sapphire/ (doi_regex.js, search_doi.js).
 *
 * Several patterns deliberately mirror Python counterparts (sapphire/extract.py
 * / sapphire/heuristic_utils.py) — Python cannot import from this file, so the
 * two copies must be kept in sync by hand:
 *   normHeading   ↔ extract.py _norm_heading / heuristic.py _norm_heading
 *   REF_HEADINGS  ↔ extract.py _REF_SECTION_HEADINGS / heuristic.py _REF_HEADINGS
 *   tokenise      ↔ heuristic_utils.py tokenise (same stopword list)
 */

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
  return tokens.filter((token) => !STOPWORDS.has(token) && token.length > 2);
}
