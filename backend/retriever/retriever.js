/**
 * retriever.js — RAG retrieval + answer synthesis for /api/chat
 *
 * The query embedding arrives from the BROWSER (the frontend embeds the
 * user's question with the same MiniLM model the corpus was embedded with),
 * so retrieval here is pure math over embeddings.json — no model loads.
 *
 * Scoring: cosine similarity × category keyword boost. If a query token
 * matches a keyword of a category (categories.json), every chunk of every
 * doc in that category gets a 5% boost per matching keyword, multiplicative:
 * score = cosine × 1.05^matches.
 *
 * Answering: Ollama chat with REASONING_MODEL from .env; retrieved chunks
 * are injected as context in the system prompt.
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
const DEFAULT_TOP_K  = parseInt(process.env.RETRIEVER_TOP_K || '8', 10);
const OLLAMA_TIMEOUT = parseInt(process.env.OLLAMA_TIMEOUT_MS || '120000', 10);

// ---------------------------------------------------------------------------
// Corpus cache (reloaded when the underlying file timestamps change)
// ---------------------------------------------------------------------------

let _cache = null;

async function loadCorpus() {
  let store;
  try {
    store = JSON.parse(await fs.readFile(EMBEDDINGS_PATH, 'utf-8'));
  } catch {
    const err = new Error('embeddings.json not found — run the embed stage first');
    err.status = 503;
    throw err;
  }

  let categories = [];
  try {
    categories = JSON.parse(await fs.readFile(CATEGORIES_PATH, 'utf-8')).categories || [];
  } catch { /* no categories → no keyword boost */ }

  return { updated: store.metadata?.updated, dims: store.metadata?.dimensions, chunks: store.chunks, categories };
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
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Per-doc multiplicative boost from category keyword matches:
 * 1.05^(query tokens ∩ category keywords) for every doc in that category.
 */
function keywordBoosts(queryText, categories) {
  const queryTokens = new Set(tokenise(queryText || ''));
  const boost = new Map();
  for (const cat of categories) {
    const matches = (cat.keywords || []).filter((k) => queryTokens.has(k)).length;
    if (matches === 0) continue;
    const factor = KEYWORD_BOOST ** matches;
    for (const m of cat.members || []) boost.set(m.docId, factor);
  }
  return boost;
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

  const boosts = keywordBoosts(queryText, corpus.categories);
  const scored = corpus.chunks.map((c) => {
    const sim   = dot(queryEmbedding, c.embedding);
    const boost = boosts.get(c.docId) || 1;
    return { chunk: c, sim, boost, score: sim * boost };
  });
  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, topK).map(({ chunk, sim, boost, score }) => ({
    chunkId:  chunk.id,
    docId:    chunk.docId,
    filename: chunk.filename,
    heading:  chunk.heading,
    pages:    chunk.pages ?? null,
    text:     chunk.text,
    sim:      Math.round(sim * 10000) / 10000,
    boost:    Math.round(boost * 10000) / 10000,
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
    .map((c, i) => `[${i + 1}] ${c.filename}${c.heading ? ` — ${c.heading}` : ''}\n${c.text}`)
    .join('\n\n');

  const system =
    'You answer questions about a document corpus. Ground every claim in the ' +
    'context excerpts below and cite them by their [n] marker. If the context ' +
    'does not contain the answer, say so plainly.\n\nContext:\n' + context;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), OLLAMA_TIMEOUT);
  let resp;
  try {
    resp = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: ctrl.signal,
      body: JSON.stringify({
        model,
        stream: false,
        messages: [{ role: 'system', content: system }, ...messages],
        options: { temperature: 0.2 },
      }),
    });
  } catch (e) {
    const err = new Error(`Ollama unreachable at ${OLLAMA_URL} (${e.name === 'AbortError' ? 'timeout' : e.message})`);
    err.status = 502;
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) {
    const detail = await resp.text().catch(() => '');
    const err = new Error(`Ollama HTTP ${resp.status}${detail ? `: ${detail.slice(0, 200)}` : ''}`);
    err.status = 502;
    throw err;
  }

  const body = await resp.json();
  return { reply: (body.message?.content || '').trim(), model };
}
