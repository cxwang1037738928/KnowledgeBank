/**
 * chunker.js — structure-aware chunking for DoclingDocument extracts
 *
 * Two entry points:
 *
 *   chunkDocument(entry, opts)  — PREFERRED. Takes one doclings.json entry
 *     (the output of extract.py) and chunks along the document's own
 *     structure: section boundaries are never crossed, every chunk carries
 *     its section heading as an embedded prefix (so the embedding "knows"
 *     where the text came from), tiny adjacent sections are merged so you
 *     don't get fragment chunks, and tables are emitted as standalone
 *     chunks instead of being shredded mid-row.
 *
 *   chunkText(text, chunkSize, overlap) — fallback for plain text with no
 *     structure (keeps the signature embed.js already uses). Sentence-aware
 *     sliding window: chunks end on sentence boundaries where possible and
 *     overlap by whole sentences, not by an arbitrary character cut.
 *
 * Sizes are in WORDS, not characters. Note the embedding model
 * (all-MiniLM-L12-v2) truncates input around 256 word-piece tokens, i.e.
 * roughly 180–200 English words. CHUNK_SIZE much above ~200 means the tail
 * of every chunk is silently thrown away at embed time.
 */

const DEFAULTS = {
  chunkSize: process.env.CHUNK_SIZE ? parseInt(process.env.CHUNK_SIZE) : 180,
  overlap:   process.env.CHUNK_OVERLAP ? parseInt(process.env.CHUNK_OVERLAP) : 20,
  minSectionMerge: 60,
  maxTableWords: 300
};

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

const wordCount = (s) => (s.match(/\S+/g) || []).length;

/**
 * Split text into sentences. Deliberately conservative: splits on
 * .!? followed by whitespace + capital/opening char, and avoids splitting
 * on common abbreviations and initials ("et al.", "Fig.", "e.g.", "J. Smith")
 * that are everywhere in academic prose.
 */
function splitSentences(text) {
  const ABBREV = /(?:et al|e\.g|i\.e|cf|vs|fig|figs|eq|eqs|sec|ref|refs|no|vol|pp|dr|mr|ms|prof|jr|sr|approx)\.$/i;
  const parts = [];
  let start = 0;
  const re = /[.!?]+["')\]]*\s+(?=[A-Z0-9("'\[])/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const candidate = text.slice(start, m.index + m[0].length);
    const trimmed = candidate.trimEnd();
    // Don't split right after an abbreviation or a single-letter initial
    if (ABBREV.test(trimmed) || /\b[A-Z]\.$/.test(trimmed)) continue;
    if (trimmed) parts.push(trimmed);
    start = m.index + m[0].length;
  }
  const tail = text.slice(start).trim();
  if (tail) parts.push(tail);
  return parts.length ? parts : (text.trim() ? [text.trim()] : []);
}

/**
 * Sentence-aware sliding window over plain text.
 * Returns string[] — same shape embed.js already expects.
 */
export function chunkText(text, chunkSize = DEFAULTS.chunkSize, overlap = DEFAULTS.overlap) {
  if (!text || !text.trim()) return [];
  const sentences = splitSentences(text);
  const chunks = [];
  let current = [];   // sentences in the chunk being built
  let currentWords = 0;

  const flush = () => {
    if (!current.length) return;
    chunks.push(current.join(' '));
    // Carry trailing sentences forward as overlap
    let carried = [];
    let carriedWords = 0;
    for (let i = current.length - 1; i >= 0 && carriedWords < overlap; i--) {
      carried.unshift(current[i]);
      carriedWords += wordCount(current[i]);
    }
    // Overlap must never be the whole chunk, or we'd loop forever
    if (carriedWords >= currentWords) carried = [];
    current = carried;
    currentWords = current.reduce((s, sent) => s + wordCount(sent), 0);
  };

  for (const sentence of sentences) {
    const w = wordCount(sentence);
    if (w >= chunkSize) {
      // Pathological sentence (equations, mangled OCR): hard-split by words
      flush();
      const words = sentence.split(/\s+/);
      for (let i = 0; i < words.length; i += chunkSize - overlap) {
        chunks.push(words.slice(i, i + chunkSize).join(' '));
      }
      current = [];
      currentWords = 0;
      continue;
    }
    if (currentWords + w > chunkSize && currentWords > 0) flush();
    // After flush the carry is kept as overlap; if overlap + sentence still
    // overflows (large sentence near the chunk limit), drop the carry so the
    // final chunk stays within chunkSize.
    if (currentWords + w > chunkSize) { current = []; currentWords = 0; }
    current.push(sentence);
    currentWords += w;
  }
  if (current.length) chunks.push(current.join(' '));
  return chunks;
}

// ---------------------------------------------------------------------------
// Structure-aware chunking over a doclings.json entry
// ---------------------------------------------------------------------------

/**
 * chunkDocument(entry, opts) → [{ text, heading, sectionIndex, type }]
 *
 * entry is one value from doclings.json:
 *   { text, markdown, sections: [{heading, text}], tables: [str], metadata }
 *
 * Strategy:
 *   1. Merge runs of tiny sections (< minSectionMerge words) into one unit so
 *      front-matter fragments don't become their own near-empty chunks.
 *   2. Within each unit, sentence-chunk the body and prefix every chunk with
 *      its heading path — the heading text participates in the embedding,
 *      which measurably helps retrieval for queries phrased like headings
 *      ("related work on X", "evaluation methodology").
 *   3. Emit each table as ONE chunk (truncated if huge). Splitting a
 *      markdown table mid-row produces chunks that embed as noise.
 *   4. If the entry has no usable sections, fall back to chunkText on
 *      entry.text so nothing silently drops out of the index.
 */
export function chunkDocument(entry, opts = {}) {
  const { chunkSize, overlap, minSectionMerge, maxTableWords } = { ...DEFAULTS, ...opts };
  const sections = (entry.sections || []).filter((s) => (s.text || '').trim() || (s.heading || '').trim());

  // Fallback: no structure available (e.g. OCR doc where docling found no headers)
  if (sections.length === 0) {
    return chunkText(entry.text || '', chunkSize, overlap).map((text, i) => ({
      text,
      heading: null,
      sectionIndex: null,
      type: 'text',
    }));
  }

  // --- 1. merge tiny adjacent sections -------------------------------------
  const units = []; // { heading, text, sectionIndex }
  for (let i = 0; i < sections.length; i++) {
    const sec = sections[i];
    const body = (sec.text || '').trim();
    const heading = (sec.heading || '').trim();
    const words = wordCount(body);

    const last = units[units.length - 1];
    if (last && wordCount(last.text) < minSectionMerge && words < minSectionMerge) {
      // Fold this tiny section into the previous tiny one, keeping its
      // heading inline so no structural signal is lost.
      last.text = [last.text, heading ? `${heading}. ${body}` : body]
        .filter(Boolean).join(' ');
    } else {
      units.push({ heading, text: body, sectionIndex: i });
    }
  }

  // --- 2. chunk each unit, prefixing the heading ----------------------------
  const title = entry.metadata?.title || '';
  const out = [];
  for (const unit of units) {
    // Heading prefix: "Title — Heading\n" gives the embedder document +
    // section context without eating much of the token budget.
    const prefixParts = [title, unit.heading].filter(Boolean);
    const prefix = prefixParts.length ? `${prefixParts.join(' — ')}\n` : '';
    const prefixWords = wordCount(prefix);

    const bodyChunks = chunkText(unit.text, Math.max(chunkSize - prefixWords, 50), overlap);
    if (bodyChunks.length === 0 && unit.heading) {
      // Heading-only section (e.g. "Appendix") — still worth a stub? No: skip.
      continue;
    }
    for (const body of bodyChunks) {
      out.push({
        text: prefix + body,
        heading: unit.heading || null,
        sectionIndex: unit.sectionIndex,
        type: 'text',
      });
    }
  }

  // --- 3. tables as standalone chunks ---------------------------------------
  (entry.tables || []).forEach((tbl, i) => {
    const t = (tbl || '').trim();
    if (!t) return;
    const words = t.split(/\s+/);
    const truncated = words.length > maxTableWords
      ? words.slice(0, maxTableWords).join(' ') + ' …'
      : t;
    out.push({
      text: (title ? `${title} — Table\n` : '') + truncated,
      heading: 'Table',
      sectionIndex: null,
      type: 'table',
    });
  });

  return out;
}

export default { chunkText, chunkDocument };