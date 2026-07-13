/**
 * retriever.js — RAG retrieval + answer synthesis for /api/chat
 *
 * The query embedding arrives from the BROWSER (the frontend embeds the
 * user's question with the same MiniLM model the corpus was embedded with),
 * so retrieval here is pure math over embeddings.json — no model loads.
 *
 * Scoring: (cosine × category boost) + lexical BM25 blend.
 *   - category boost: if a query token matches a keyword of a category
 *     (categories.json), every chunk of every doc in that category gets a 5%
 *     boost per matching keyword, multiplicative: 1.05^matches.
 *   - BM25 over chunk text, normalized to the query's best-scoring chunk and
 *     weighted by LEXICAL_WEIGHT. Pure-semantic ranking fails meta-queries
 *     ("which document mentions quantum chemicals?") — the question framing
 *     dominates the embedding while the literal term pins the exact chunk.
 *
 * Answering: Ollama chat with REASONING_MODEL from .env; retrieved chunks
 * are injected as context in the system prompt. The response is streamed so
 * the timeout can key on inactivity instead of total time — a small model on
 * CPU needs ~110s for a long answer, which is slow but not hung.
 *
 * Exports:
 *   retrieve(queryEmbedding, queryText, {topK}) — ranked chunks
 *   answer(messages, chunks)                    — Ollama completion string
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { tokenise } from '../extraction/regex_utils.js';

const ROOT            = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA_DIR        = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.json');
const CATEGORIES_PATH = path.join(DATA_DIR, 'categories.json');

const OLLAMA_URL     = process.env.OLLAMA_URL || 'http://localhost:11434';
const KEYWORD_BOOST  = 1.05;   // per matched category keyword, multiplicative
const LEXICAL_WEIGHT = parseFloat(process.env.RETRIEVER_LEXICAL_WEIGHT || '0.3');
const BM25_K1        = 1.5;
const BM25_B         = 0.75;
const DEFAULT_TOP_K  = parseInt(process.env.RETRIEVER_TOP_K || '8', 10);
// Relevance floor: keep only chunks scoring within this fraction of the best
// one. Without it every query returns exactly topK chunks, so a question with
// one real hit also ships 7 near-misses the model has to explain away.
const SCORE_FLOOR = parseFloat(process.env.RETRIEVER_SCORE_FLOOR || '0.5');
// Inactivity timeout, NOT a wall-clock deadline: we stream from Ollama and
// reset this on every token. A long answer is more work, not a hang — a fixed
// deadline killed perfectly healthy 110s generations.
const OLLAMA_IDLE_TIMEOUT = parseInt(process.env.OLLAMA_IDLE_TIMEOUT_MS || '60000', 10);

// ---------------------------------------------------------------------------
// Corpus cache (reloaded when the underlying file timestamps change)
// ---------------------------------------------------------------------------

let _cache = null;

async function loadCorpus() {
  let embeddingStore;
  try {
    embeddingStore = JSON.parse(await fs.readFile(EMBEDDINGS_PATH, 'utf-8'));
  } catch {
    const err = new Error('embeddings.json not found — run the embed stage first');
    err.status = 503;
    throw err;
  }

  let categories = [];
  try {
    categories = JSON.parse(await fs.readFile(CATEGORIES_PATH, 'utf-8')).categories || [];
  } catch { /* no categories → no keyword boost */ }

  return {
    updated: embeddingStore.metadata?.updated,
    dims: embeddingStore.metadata?.dimensions,
    chunks: embeddingStore.chunks,
    categories,
    lexical: buildLexicalIndex(embeddingStore.chunks),
  };
}

/** Per-chunk term frequencies + document frequencies for BM25. */
function buildLexicalIndex(chunks) {
  const docFreq = new Map();
  const chunkTermFreqs = chunks.map((chunk) => {
    const termFreq = new Map();
    for (const token of tokenise(chunk.text || '')) termFreq.set(token, (termFreq.get(token) || 0) + 1);
    for (const token of termFreq.keys()) docFreq.set(token, (docFreq.get(token) || 0) + 1);
    return termFreq;
  });
  const avgChunkLength = chunkTermFreqs.reduce(
    (total, termFreq) => total + termFreq.size, 0) / (chunkTermFreqs.length || 1);
  return { docFreq, chunkTermFreqs, avgChunkLength, chunkCount: chunks.length };
}

/**
 * BM25 score of every chunk against the query tokens, with two twists:
 *   - idf is SQUARED so rare terms dominate. Meta-queries ("which document
 *     mentions quantum chemicals") carry several medium-rarity words
 *     (document, part, mentions) whose classic-BM25 sum outweighs the one
 *     df=1 term the user actually cares about.
 *   - unmatched plural query tokens fall back to their singular ("chemicals"
 *     → "chemical") since the corpus index is unstemmed.
 */
function bm25Scores(queryTokens, { docFreq, chunkTermFreqs, avgChunkLength, chunkCount }) {
  const scores = new Float64Array(chunkTermFreqs.length);
  const matchedTerms = new Set();
  for (const token of queryTokens) {
    if (docFreq.has(token)) matchedTerms.add(token);
    else if (token.endsWith('s') && docFreq.has(token.slice(0, -1))) {
      matchedTerms.add(token.slice(0, -1));
    }
  }
  for (const term of matchedTerms) {
    const termDocFreq = docFreq.get(term);
    const idf = Math.log(1 + (chunkCount - termDocFreq + 0.5) / (termDocFreq + 0.5)) ** 2;
    for (let chunkIdx = 0; chunkIdx < chunkTermFreqs.length; chunkIdx++) {
      const termFreq = chunkTermFreqs[chunkIdx].get(term);
      if (!termFreq) continue;
      scores[chunkIdx] += idf * (termFreq * (BM25_K1 + 1)) /
        (termFreq + BM25_K1 * (1 - BM25_B + BM25_B
          * (chunkTermFreqs[chunkIdx].size / avgChunkLength)));
    }
  }
  return scores;
}

async function getCorpus() {
  const updated = await fs.stat(EMBEDDINGS_PATH).then((s) => s.mtimeMs).catch(() => 0);
  if (!_cache || _cache.stamp !== updated) {
    _cache = { stamp: updated, data: await loadCorpus() };
  }
  return _cache.data;
}

// ---------------------------------------------------------------------------
// Retrieval
// ---------------------------------------------------------------------------

// Embeddings are L2-normalized, so dot = cosine.
function dot(vecA, vecB) {
  let sum = 0;
  for (let componentIdx = 0; componentIdx < vecA.length; componentIdx++) {
    sum += vecA[componentIdx] * vecB[componentIdx];
  }
  return sum;
}

/**
 * Per-doc multiplicative boost from category keyword matches:
 * 1.05^(query tokens ∩ category keywords) for every doc in that category.
 */
function keywordBoosts(queryText, categories) {
  const queryTokens = new Set(tokenise(queryText || ''));
  const boostByDocId = new Map();
  for (const category of categories) {
    const matchCount = (category.keywords || [])
      .filter((keyword) => queryTokens.has(keyword)).length;
    if (matchCount === 0) continue;
    const boostFactor = KEYWORD_BOOST ** matchCount;
    for (const member of category.members || []) boostByDocId.set(member.docId, boostFactor);
  }
  return boostByDocId;
}

/**
 * Rank corpus chunks against a query.
 * @param {number[]} queryEmbedding — L2-normalized, same model/dims as the corpus
 * @param {string}   queryText      — raw question, for keyword matching
 * @returns top-K [{docId, filename, heading, text, sim, boost, score}]
 */
export async function retrieve(queryEmbedding, queryText, { topK = DEFAULT_TOP_K } = {}) {
  const corpus = await getCorpus();
  if (corpus.dims && queryEmbedding.length !== corpus.dims) {
    const err = new Error(`query embedding has ${queryEmbedding.length} dims; corpus uses ${corpus.dims}`);
    err.status = 400;
    throw err;
  }

  const boostByDocId = keywordBoosts(queryText, corpus.categories);
  // Normalize BM25 to the query's best chunk so the blend weight is stable
  // across queries of different rarity.
  const bm25 = bm25Scores(tokenise(queryText || ''), corpus.lexical);
  const bm25Max = Math.max(...bm25) || 1;

  const scored = corpus.chunks.map((chunk, chunkIdx) => {
    const sim   = dot(queryEmbedding, chunk.embedding);
    const boost = boostByDocId.get(chunk.docId) || 1;
    const lex   = bm25[chunkIdx] / bm25Max;
    return { chunk, sim, boost, lex, score: sim * boost + LEXICAL_WEIGHT * lex };
  });
  scored.sort((a, b) => b.score - a.score);

  // Drop the long tail below the floor; the top chunk always survives.
  const cutoff = Math.max(scored[0]?.score * SCORE_FLOOR || 0, 0);
  const kept = scored.slice(0, topK).filter((c, idx) => idx === 0 || c.score >= cutoff);

  return kept.map(({ chunk, sim, boost, lex, score }) => ({
    chunkId:  chunk.id,
    docId:    chunk.docId,
    filename: chunk.filename,
    heading:  chunk.heading,
    pages:    chunk.pages ?? null,
    text:     chunk.text,
    sim:      Math.round(sim * 10000) / 10000,
    boost:    Math.round(boost * 10000) / 10000,
    lex:      Math.round(lex * 10000) / 10000,
    score:    Math.round(score * 10000) / 10000,
  }));
}

// ---------------------------------------------------------------------------
// Answer synthesis (Ollama, REASONING_MODEL)
// ---------------------------------------------------------------------------

/**
 * Answer the conversation using the retrieved chunks as grounding context.
 * @param {{role: string, content: string}[]} messages — chat history, last = question
 * @param {Awaited<ReturnType<typeof retrieve>>} chunks
 * @returns {Promise<{reply: string, model: string}>}
 */
export async function answer(messages, chunks) {
  const model = (process.env.REASONING_MODEL || '').trim();
  if (!model) {
    const err = new Error('REASONING_MODEL is not set — pick one in the Models tab');
    err.status = 503;
    throw err;
  }

  const context = chunks
    .map((chunk, chunkIdx) =>
      `[${chunkIdx + 1}] ${chunk.filename}${chunk.heading ? ` — ${chunk.heading}` : ''}\n${chunk.text}`)
    .join('\n\n');

  // The citation format is load-bearing: the frontend turns [n] into a link
  // into the PDF, so a model that writes [n1] or names the file inline
  // produces an answer with no working citations. Be blunt about it.
  const system =
    'You answer questions about a document corpus.\n\n' +
    'Ground every claim in the context excerpts below. Cite each claim with the ' +
    'excerpt\'s number in square brackets — exactly like [1] or [2], or [1, 3] for ' +
    'several. Write the number alone: never [n1], never [source 1], never the ' +
    'filename. If the context does not contain the answer, say so plainly.\n\n' +
    'Context:\n' + context;

  // Stream so the timeout can be based on inactivity rather than total time:
  // a slow model writing a long answer is working, not hung.
  const abortController = new AbortController();
  let idleTimer;
  const armIdleTimer = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => abortController.abort(), OLLAMA_IDLE_TIMEOUT);
  };

  let response;
  armIdleTimer();
  try {
    response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortController.signal,
      body: JSON.stringify({
        model,
        stream: true,
        messages: [{ role: 'system', content: system }, ...messages],
        options: { temperature: 0.2 },
      }),
    });
  } catch (err) {
    clearTimeout(idleTimer);
    const stalled = err.name === 'AbortError';
    const httpError = new Error(stalled
      ? `Ollama at ${OLLAMA_URL} sent nothing for ${OLLAMA_IDLE_TIMEOUT / 1000}s`
      : `Ollama unreachable at ${OLLAMA_URL} (${err.message})`);
    httpError.status = 502;
    throw httpError;
  }

  if (!response.ok) {
    clearTimeout(idleTimer);
    const detail = await response.text().catch(() => '');
    const httpError = new Error(
      `Ollama HTTP ${response.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`);
    httpError.status = 502;
    throw httpError;
  }

  // NDJSON: one JSON object per line, each carrying a token in message.content.
  let reply = '';
  let pendingLine = '';
  try {
    const decoder = new TextDecoder();
    for await (const streamBytes of response.body) {
      armIdleTimer();                          // progress — restart the idle clock
      pendingLine += decoder.decode(streamBytes, { stream: true });
      const lines = pendingLine.split('\n');
      pendingLine = lines.pop() ?? '';         // keep the trailing partial line
      for (const line of lines) {
        if (!line.trim()) continue;
        let streamEvent;
        try { streamEvent = JSON.parse(line); } catch { continue; }
        if (streamEvent.error) throw new Error(streamEvent.error);
        reply += streamEvent.message?.content || '';
      }
    }
  } catch (err) {
    const stalled = err.name === 'AbortError';
    const httpError = new Error(stalled
      ? `Ollama stalled — no output for ${OLLAMA_IDLE_TIMEOUT / 1000}s`
      : `Ollama stream failed: ${err.message}`);
    httpError.status = 502;
    throw httpError;
  } finally {
    clearTimeout(idleTimer);
  }

  // Repair before returning: the model's numbering is a guess, not an index.
  return { reply: repairCitations(normalizeCitations(reply.trim(), chunks.length), chunks), model };
}

// ---------------------------------------------------------------------------
// Citation repair
// ---------------------------------------------------------------------------

const CITE_MARKER   = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;
const QUOTED_SPAN   = /["“”']([^"“”']{25,})["“”']/g;
const VERBATIM_RUN  = 8;     // words of literal overlap that pin a claim to a chunk
const MIN_OVERLAP   = 0.5;   // else the claim is ungrounded and the marker is dropped
const CLAIM_WINDOW  = 400;   // chars each side of a marker searched for its quote

// Space-free comparison, same trick as the PDF viewer: immune to how each side
// hyphenates, spaces, or line-breaks the text.
const spaceless = (text) => tokenise(text || '').join('');

/**
 * Which retrieved chunk does this claim actually come from?
 * A literal run of VERBATIM_RUN words outranks any amount of loose word overlap.
 */
function bestChunkFor(claim, chunkIndex) {
  const claimTokens = tokenise(claim);
  if (claimTokens.length < 4) return { chunkIdx: -1, score: 0 };

  let best = { chunkIdx: -1, score: 0 };
  chunkIndex.forEach((indexed, chunkIdx) => {
    let verbatim = 0;
    for (let start = 0; start + VERBATIM_RUN <= claimTokens.length; start++) {
      if (indexed.spaceless.includes(claimTokens.slice(start, start + VERBATIM_RUN).join(''))) {
        verbatim = 1;
        break;
      }
    }
    const overlap = claimTokens.filter((token) => indexed.tokens.has(token)).length
      / claimTokens.length;
    const score = verbatim + overlap;
    if (score > best.score) best = { chunkIdx, score };
  });
  return best;
}

/** Quoted spans with their positions: "..." and blockquote ("> ...") lines. */
function quotedSpans(text) {
  const spans = [...text.matchAll(QUOTED_SPAN)].map((quote) => ({
    text: quote[1],
    start: quote.index,
    end: quote.index + quote[0].length,
  }));

  let lineStart = 0;
  for (const line of text.split('\n')) {
    if (/^\s*>\s?\S/.test(line)) {
      spans.push({
        text: line.replace(/^\s*>\s?/, ''),
        start: lineStart,
        end: lineStart + line.length,
      });
    }
    lineStart += line.length + 1;
  }
  return spans;
}

/** Chars between a span and a marker; 0 when the marker sits inside the span. */
function gapTo(span, markerStart, markerEnd) {
  if (span.end < markerStart) return markerStart - span.end;
  if (span.start > markerEnd) return span.start - markerEnd;
  return 0;
}

/** The sentence a marker sits in — the claim when nothing nearby is quoted. */
function sentenceAround(text, markerStart, markerEnd) {
  const before = text.slice(Math.max(0, markerStart - CLAIM_WINDOW), markerStart);
  const after  = text.slice(markerEnd, markerEnd + CLAIM_WINDOW);
  return (before.split(/(?<=[.!?])\s|\n/).pop() || '')
    + (after.split(/(?<=[.!?])\s|\n/)[0] || '');
}

/**
 * Re-point every [n] at the chunk whose text the claim actually came from, and
 * delete markers no chunk supports. The model quotes one excerpt and cites
 * another (and copies the source paper's own [12]-style refs verbatim), so a
 * number in the reply cannot be trusted as an index into `chunks`.
 */
function repairCitations(text, chunks) {
  if (!chunks.length) return text;
  const chunkIndex = chunks.map((chunk) => ({
    spaceless: spaceless(chunk.text),
    tokens: new Set(tokenise(chunk.text || '')),
  }));
  const spans = quotedSpans(text);

  let repaired = '';
  let cursor = 0;
  for (const marker of text.matchAll(CITE_MARKER)) {
    const markerStart = marker.index;
    const markerEnd = markerStart + marker[0].length;

    // Cite the NEAREST quote, not the best-scoring one in the neighbourhood —
    // otherwise a strong quote two paragraphs away steals its neighbour's marker.
    const quote = spans
      .map((span) => ({ span, gap: gapTo(span, markerStart, markerEnd) }))
      .filter((candidate) => candidate.gap <= CLAIM_WINDOW)
      .sort((a, b) => a.gap - b.gap)[0];

    let target = -1;
    if (quote) {
      // A quote claims to be verbatim, so it must literally appear in a chunk.
      // If none contains it, the model made it up — drop the marker entirely.
      const best = bestChunkFor(quote.span.text, chunkIndex);
      if (best.score >= 1) target = best.chunkIdx;
    } else {
      // Paraphrase: no literal text to match, so fall back to word overlap.
      const best = bestChunkFor(sentenceAround(text, markerStart, markerEnd), chunkIndex);
      if (best.score >= MIN_OVERLAP) target = best.chunkIdx;
    }

    repaired += text.slice(cursor, markerStart);
    if (target >= 0) repaired += `[${target + 1}]`;
    cursor = markerEnd;
  }
  repaired += text.slice(cursor);

  // Tidy what dropped markers left behind: doubled spaces, space before punctuation.
  return repaired.replace(/ {2,}/g, ' ').replace(/ ([.,;:)])/g, '$1');
}

/**
 * Coerce the citation markers small models actually emit into the [n] form
 * the frontend links on: [n1], [N1], [#1], [source 1], [1](...) → [1].
 * Markers pointing past the excerpt count are left alone (they'd link nowhere).
 */
function normalizeCitations(text, excerptCount) {
  return text
    .replace(/\[\s*(?:n|N|#|source|excerpt|ref)\s*[.:]?\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g, '[$1]')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]\([^)]*\)/g, '[$1]')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (marker, markerNumbers) => {
      const validNumbers = markerNumbers.split(',')
        .map((number) => parseInt(number, 10))
        .filter((number) => number >= 1 && number <= excerptCount);
      return validNumbers.length ? `[${validNumbers.join(', ')}]` : marker;
    });
}
