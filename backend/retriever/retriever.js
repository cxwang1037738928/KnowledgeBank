/**
 * retriever.js — RAG retrieval + answer synthesis for /api/chats/:chatId/chat
 *
 * The query embedding arrives from the BROWSER (the frontend embeds the
 * user's question with the same MiniLM model the corpus was embedded with),
 * so retrieval here is pure math over the collection's Chunk rows — no model loads.
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
 *   retrieve(collection, queryEmbedding, queryText, {topK}) — ranked chunks
 *   answer(messages, chunks)                          — Ollama completion string
 */

import 'dotenv/config';
import { tokenise } from '../extraction/regex_utils.js';
import { getPrompt } from '../prompts.js';
import { embedTexts } from './embedder.js';
import { prisma } from '../db.js';

const OLLAMA_URL     = process.env.OLLAMA_URL || 'http://localhost:11434';
// per matched category keyword, multiplicative
const KEYWORD_BOOST  = parseFloat(process.env.RETRIEVER_KEYWORD_BOOST || '1.05');
const LEXICAL_WEIGHT = parseFloat(process.env.RETRIEVER_LEXICAL_WEIGHT || '0.3');
const BM25_K1        = parseFloat(process.env.RETRIEVER_BM25_K1 || '1.5');   // term-frequency saturation
const BM25_B         = parseFloat(process.env.RETRIEVER_BM25_B  || '0.75');  // length normalization strength
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
// Corpus cache (per collection, invalidated when corpusUpdatedAt changes)
// ---------------------------------------------------------------------------

const _cacheByCollection = new Map();

async function loadCorpus(collection) {
  const rows = await prisma.chunk.findMany({
    where: { collectionId: collection.id },
    orderBy: [{ docId: 'asc' }, { chunkIndex: 'asc' }],
  });
  if (rows.length === 0) {
    const err = new Error('This collection has no indexed chunks — upload PDFs and run the pipeline first');
    err.status = 503;
    throw err;
  }

  const chunks = rows.map((row) => ({
    id:        row.chunkId,
    docId:     row.docId,
    filename:  row.filename,
    pages:     row.pages,
    prefixLen: row.prefixLen,
    heading:   row.heading,
    text:      row.text,
    embedding: row.embedding,
  }));

  return {
    dims: collection.embeddingsMeta?.dimensions,
    chunks,
    categories: collection.categories?.categories || [],   // absent → no keyword boost
    lexical: buildLexicalIndex(chunks),
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

async function getCorpus(collection) {
  const stamp = collection.corpusUpdatedAt?.getTime() ?? 0;
  const cached = _cacheByCollection.get(collection.id);
  if (cached?.stamp === stamp) return cached.data;
  const data = await loadCorpus(collection);
  _cacheByCollection.set(collection.id, { stamp, data });
  return data;
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
 * Rank one collection's chunks against a query.
 * @param {object}   collection     — Collection row (id, corpusUpdatedAt, categories, embeddingsMeta)
 * @param {number[]} queryEmbedding — L2-normalized, same model/dims as the corpus
 * @param {string}   queryText      — raw question, for keyword matching
 * @returns top-K [{docId, filename, heading, text, sim, boost, score}]
 */
export async function retrieve(collection, queryEmbedding, queryText, { topK = DEFAULT_TOP_K } = {}) {
  const corpus = await getCorpus(collection);
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
    chunkId:   chunk.id,
    docId:     chunk.docId,
    filename:  chunk.filename,
    heading:   chunk.heading,
    pages:     chunk.pages ?? null,
    text:      chunk.text,
    embedding: chunk.embedding,   // kept for citation grounding; chat.js strips it from sources
    sim:       Math.round(sim * 10000) / 10000,
    boost:     Math.round(boost * 10000) / 10000,
    lex:       Math.round(lex * 10000) / 10000,
    score:     Math.round(score * 10000) / 10000,
  }));
}

// ---------------------------------------------------------------------------
// Answer synthesis (Ollama, REASONING_MODEL)
// ---------------------------------------------------------------------------

/**
 * Answer the conversation using the retrieved chunks as grounding context.
 * @param {{role: string, content: string}[]} messages — chat history, last = question
 * @param {Awaited<ReturnType<typeof retrieve>>} chunks
 * @returns {Promise<{reply: string, model: string, quotesByChunk: string[][]}>}
 *          quotesByChunk[i] = verbatim quotes in the reply grounded by chunks[i]
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

  // The citation format the prompt demands is load-bearing: the frontend
  // turns [n] into a link into the PDF, so a model that writes [n1] or names
  // the file inline produces an answer with no working citations.
  const system = getPrompt('chat_system', { context });

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
  const { text: repairedReply, quotesByChunk } =
    await repairCitations(normalizeCitations(reply.trim(), chunks.length), chunks);
  return { reply: repairedReply, model, quotesByChunk };
}

// ---------------------------------------------------------------------------
// Citation repair
// ---------------------------------------------------------------------------

const CITE_MARKER   = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;
const QUOTED_SPAN   = /["“”']([^"“”']{25,})["“”']/g;
// words of literal overlap that pin a QUOTE to a chunk
const VERBATIM_RUN  = parseInt(process.env.RETRIEVER_VERBATIM_RUN || '8', 10);
// chars each side of a marker searched for its quote
const CLAIM_WINDOW  = parseInt(process.env.RETRIEVER_CLAIM_WINDOW || '400', 10);
// A paraphrase (no quote marks) is grounded when EITHER signal clears its bar:
//   - MIN_OVERLAP: fraction of the claim's content words present literally in a
//     chunk. Catches claims that reuse the source's vocabulary.
//   - GROUNDING_SIM: cosine between the claim and the chunk in embedding space.
//     Catches claims reworded in the model's own words, which share few literal
//     words but stay semantically close.
// Both, not just cosine: a short, generic true paraphrase can score LOWER on
// cosine than a plausible on-topic fabrication (measured: "…are enumerable"
// 0.45 vs a fabricated "…tabulated by Babbage on the difference engine" 0.48),
// but the true one reuses corpus words and clears MIN_OVERLAP. A claim is
// flagged [!] only when it fails both. GROUNDING_SIM is deliberately high so
// topical-but-unsupported claims still fall through.
const MIN_OVERLAP   = parseFloat(process.env.RETRIEVER_MIN_OVERLAP || '0.5');
const GROUNDING_SIM = parseFloat(process.env.RETRIEVER_GROUNDING_SIM || '0.6');

// Space-free comparison, same trick as the PDF viewer: immune to how each side
// hyphenates, spaces, or line-breaks the text.
const spaceless = (text) => tokenise(text || '').join('');

/**
 * Which retrieved chunk does this claim actually come from?
 * A literal run of VERBATIM_RUN words (or the whole claim, when shorter)
 * outranks any amount of loose word overlap; `verbatim` reports whether the
 * winning chunk contains such a run, since quotes are only trustable verbatim.
 */
function bestChunkFor(claim, chunkIndex) {
  const claimTokens = tokenise(claim);
  if (claimTokens.length < 4) return { chunkIdx: -1, score: 0, verbatim: false };
  const runLength = Math.min(VERBATIM_RUN, claimTokens.length);

  let best = { chunkIdx: -1, score: 0, verbatim: false };
  chunkIndex.forEach((indexed, chunkIdx) => {
    let verbatim = 0;
    for (let start = 0; start + runLength <= claimTokens.length; start++) {
      if (indexed.spaceless.includes(claimTokens.slice(start, start + runLength).join(''))) {
        verbatim = 1;
        break;
      }
    }
    const overlap = claimTokens.filter((token) => indexed.tokens.has(token)).length
      / claimTokens.length;
    const score = verbatim + overlap;
    if (score > best.score) best = { chunkIdx, score, verbatim: verbatim === 1 };
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

/**
 * The claim a marker refers to, when nothing nearby is quoted. Small models
 * routinely park the marker on its OWN line, after the sentence it supports:
 *
 *     - a computable number's decimal can be written down by a machine.
 *     [1]
 *
 * so we can't just split on \n and take the fragment touching the marker —
 * that fragment is the blank indentation, and the claim looks empty (which
 * flagged every such citation as unsupported). Instead: the claim is the last
 * NON-BLANK sentence before the marker; if it already ends a sentence (. ! ?)
 * that is the whole claim. Only a marker sitting mid-sentence glues on the
 * fragment that follows.
 */
function sentenceAround(text, markerStart, markerEnd) {
  const sentences = (segment) => segment
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  const before = sentences(text.slice(Math.max(0, markerStart - CLAIM_WINDOW), markerStart));
  const lastBefore = before[before.length - 1] || '';
  if (/[.!?]$/.test(lastBefore)) return lastBefore;

  // Glue on only what continues the SAME line. A marker at end-of-line (or on
  // its own line) belongs to the sentence before it — reaching into the next
  // line pulls an unrelated claim in and tanks the grounding of both.
  const restOfLine = text.slice(markerEnd, markerEnd + CLAIM_WINDOW).split('\n')[0];
  const after = sentences(restOfLine);
  return `${lastBefore} ${after[0] || ''}`.trim();
}

/**
 * Re-point every [n] at the chunk whose text the claim actually came from. The
 * model quotes one excerpt and cites another (and copies the source paper's own
 * [12]-style refs verbatim), so a number in the reply cannot be trusted as an
 * index into `chunks`.
 *
 * A marker no chunk supports is flagged rather than trusted or dropped: it keeps
 * the model's own number but as [n!] (frontend renders a clickable warning), so
 * the reader can still open what it points at while knowing we couldn't verify
 * it. A marker with no usable number becomes a bare [!].
 *
 * Returns { text, quotesByChunk } — quotesByChunk[i] lists the verbatim quotes
 * that grounded chunk i, so the PDF viewer can highlight the quoted lines
 * themselves instead of guessing from query similarity.
 *
 * `embedClaims(texts) → vectors` (default: the MiniLM backend embedder) supplies
 * the semantic side of paraphrase grounding. Injected so tests can stub it and
 * so a caller without a model degrades to lexical-only grounding.
 */
export const UNSUPPORTED_MARKER = '[!]';
export async function repairCitations(text, chunks, embedClaims = embedTexts) {
  const quotesByChunk = chunks.map(() => []);
  if (!chunks.length) return { text, quotesByChunk };
  const chunkIndex = chunks.map((chunk) => ({
    spaceless: spaceless(chunk.text),
    tokens: new Set(tokenise(chunk.text || '')),
    embedding: chunk.embedding || null,
  }));
  const spans = quotedSpans(text);

  // Resolve each marker to a quote (verbatim path) or a paraphrase claim.
  const markers = [...text.matchAll(CITE_MARKER)].map((marker) => {
    const markerStart = marker.index;
    const markerEnd = markerStart + marker[0].length;
    // Cite the NEAREST quote, not the best-scoring one in the neighbourhood —
    // otherwise a strong quote two paragraphs away steals its neighbour's marker.
    const quote = spans
      .map((span) => ({ span, gap: gapTo(span, markerStart, markerEnd) }))
      .filter((candidate) => candidate.gap <= CLAIM_WINDOW)
      .sort((a, b) => a.gap - b.gap)[0];
    const claim = quote ? null : sentenceAround(text, markerStart, markerEnd);
    // The model's own cited excerpt(s), kept so an ungrounded marker can still
    // link to what it points at instead of collapsing to a bare warning.
    const citedNumbers = (marker[0].match(/\d+/g) || [])
      .map(Number).filter((num) => num >= 1 && num <= chunks.length);
    return { markerStart, markerEnd, quote, claim, citedNumbers };
  });

  // Batch-embed the paraphrase claims once (strip stray [n] markers first so
  // they don't pollute the vector). Only touched when there are paraphrases,
  // so a quotes-only reply never loads the model.
  const claimTexts = [...new Set(markers
    .map((marker) => marker.claim)
    .filter((claim) => claim && claim.length >= 8)
    .map((claim) => claim.replace(CITE_MARKER, ' ').replace(/\s+/g, ' ').trim()))];
  const claimVectors = new Map();
  if (claimTexts.length) {
    try {
      const vectors = await embedClaims(claimTexts);
      claimTexts.forEach((claim, claimIdx) => claimVectors.set(claim, vectors[claimIdx]));
    } catch (err) {
      console.warn('[retriever] claim embedding failed — grounding lexically only:', err.message);
    }
  }

  // Best chunk for a paraphrase claim by cosine, and the similarity itself.
  const bestBySimilarity = (claim) => {
    const vector = claimVectors.get(claim.replace(CITE_MARKER, ' ').replace(/\s+/g, ' ').trim());
    if (!vector) return { chunkIdx: -1, sim: 0 };
    let best = { chunkIdx: -1, sim: 0 };
    chunkIndex.forEach((indexed, chunkIdx) => {
      if (!indexed.embedding) return;
      const sim = dot(vector, indexed.embedding);
      if (sim > best.sim) best = { chunkIdx, sim };
    });
    return best;
  };

  let repaired = '';
  let cursor = 0;
  for (const { markerStart, markerEnd, quote, claim, citedNumbers } of markers) {
    let target = -1;
    if (quote) {
      // A quote claims to be verbatim, so it must literally appear in a chunk —
      // verbatim run required. Word overlap and cosine are NOT enough: a quote
      // built from common words matches many chunks, and a reworded "quote" is
      // still a fabricated quote. If no chunk contains it, the model made it up.
      const best = bestChunkFor(quote.span.text, chunkIndex);
      if (best.verbatim) {
        target = best.chunkIdx;
        if (!quotesByChunk[target].includes(quote.span.text)) {
          quotesByChunk[target].push(quote.span.text);
        }
      }
    } else {
      // Paraphrase: grounded if it reuses the chunk's words (lexical overlap) OR
      // is semantically close to it (cosine). The model rewords the source, so
      // literal overlap alone flagged genuine paraphrases as fabricated.
      const lexical = bestChunkFor(claim, chunkIndex);
      if (lexical.score >= MIN_OVERLAP) {
        target = lexical.chunkIdx;
      } else {
        const semantic = bestBySimilarity(claim);
        if (semantic.sim >= GROUNDING_SIM) target = semantic.chunkIdx;
      }
    }

    repaired += text.slice(cursor, markerStart);
    // Grounded → the verified excerpt. Ungrounded → keep the model's own
    // citation but flag it ([n!]), so the reader can still open what it points
    // at while seeing it's unverified; only a marker with no usable number
    // drops to a bare [!].
    if (target >= 0) repaired += `[${target + 1}]`;
    else if (citedNumbers.length) repaired += `[${citedNumbers.join(', ')}!]`;
    else repaired += UNSUPPORTED_MARKER;
    cursor = markerEnd;
  }
  repaired += text.slice(cursor);

  // Tidy spacing the rewrite may have doubled up.
  return {
    text: repaired.replace(/ {2,}/g, ' ').replace(/ ([.,;:)])/g, '$1'),
    quotesByChunk,
  };
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
