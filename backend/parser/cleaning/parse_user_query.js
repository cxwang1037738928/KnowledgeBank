/**
 * parse_user_query.js — Stage 6: query classification
 *
 * Uses a lightweight local LLM (via Ollama) to route each user query to one
 * of two paths:
 *
 *   retrieval — the answer can be found by fetching relevant document chunks
 *               (standard RAG vector search)
 *   reason    — the answer requires multi-hop reasoning, cross-document
 *               synthesis, or structured knowledge (→ knowledge graph + Phi-4)
 *
 * The classifier is intentionally kept separate from the answer-generation
 * LLM so it stays fast and cheap — a 1-3B param model is sufficient.
 *
 * Config (env vars):
 *   OLLAMA_URL              default http://localhost:11434
 *   QUERY_CLASSIFIER_MODEL  default llama3.2:1b
 */

import 'dotenv/config';
import { getPrompt } from '../../prompts.js';

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const CLASSIFIER_MODEL = process.env.QUERY_CLASSIFIER_MODEL || 'llama3.2:1b';

// ---------------------------------------------------------------------------
// Core classifier
// ---------------------------------------------------------------------------

/**
 * Classifies a natural-language query as "retrieval" or "reason".
 *
 * @param {string} query
 * @returns {Promise<{ query: string, type: 'retrieval' | 'reason', confidence: number, model: string }>}
 */
export async function classifyQuery(query) {
  if (!query || typeof query !== 'string' || !query.trim()) {
    throw new Error('"query" must be a non-empty string');
  }

  const prompt = getPrompt('query_classifier', { query: query.trim() });

  const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model:  CLASSIFIER_MODEL,
      prompt,
      stream: false,
      options: { temperature: 0 }, // deterministic for classification
    }),
  });

  if (!resp.ok) {
    throw new Error(`Ollama returned HTTP ${resp.status}: ${await resp.text()}`);
  }

  const body = await resp.json();
  const raw  = (body.response ?? '').trim();

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Some models wrap JSON in markdown fences; strip them
    const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (match) {
      parsed = JSON.parse(match[1].trim());
    } else {
      // Best-effort fallback: look for the type keyword in the raw response
      const typeMatch = raw.match(/"?(retrieval|reason)"?/i);
      parsed = { type: typeMatch?.[1]?.toLowerCase() ?? 'retrieval', confidence: 0.5 };
    }
  }

  const type = parsed.type === 'reason' ? 'reason' : 'retrieval';
  const confidence = typeof parsed.confidence === 'number'
    ? Math.min(1, Math.max(0, parsed.confidence))
    : 0.5;

  return { query: query.trim(), type, confidence, model: CLASSIFIER_MODEL };
}

// ---------------------------------------------------------------------------
// CLI usage: node backend/parser/cleaning/parse_user_query.js "your question"
// ---------------------------------------------------------------------------

if (process.argv[1] && process.argv[1].endsWith('parse_user_query.js')) {
  const query = process.argv[2];
  if (!query) {
    console.error('Usage: node parse_user_query.js "<query>"');
    process.exit(1);
  }
  classifyQuery(query)
    .then((result) => console.log(JSON.stringify(result, null, 2)))
    .catch((err) => {
      console.error(err.message);
      process.exit(1);
    });
}
