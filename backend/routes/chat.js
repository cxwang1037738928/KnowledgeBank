/**
 * routes/chat.js — /api/chat
 *
 * RAG chat over the corpus. The frontend embeds the user's question in the
 * browser (same MiniLM model as the corpus, browser cache disabled) and sends
 * the vector along; retrieval + Ollama answering happen here (retriever/).
 *
 * POST /api/chat
 *   Request:  { messages: [{role:'user'|'assistant', content}], queryEmbedding: number[] }
 *   Response: { reply, model, sources: [{chunkId, docId, filename, heading, pages, sim, boost, score, quotes}] }
 *             sources[n-1] is what a [n] citation marker in reply refers to;
 *             quotes = verbatim spans in reply that this source grounds (the
 *             PDF viewer highlights exactly these when the citation is clicked)
 *   Errors:   400 bad payload · 502 Ollama failure · 503 missing corpus/model
 *
 * Every prompt → retrieved-context → response trio is appended to
 * chat_log.txt at the repo root.
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { retrieve, answer } from '../retriever/retriever.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
// Repo root, NOT backend/: the dev watcher watches backend/ (--watch-path),
// so a log written there would restart the server on every chat.
const CHAT_LOG_PATH = path.join(ROOT, 'chat_log.txt');

/**
 * Append one prompt → retrieved context → response trio to chat_log.txt.
 * Fire-and-forget: a logging failure must never fail the chat itself.
 */
function logChatTrio({ question, chunks, reply, model }) {
  const context = chunks
    .map((chunk, chunkIdx) =>
      `[${chunkIdx + 1}] ${chunk.filename}${chunk.heading ? ` — ${chunk.heading}` : ''}` +
      ` (score ${chunk.score})\n${chunk.text}`)
    .join('\n\n');
  const entry = [
    '='.repeat(78),
    `[${new Date().toISOString()}] model=${model}`,
    '',
    '--- prompt ---',
    question,
    '',
    '--- retrieved context ---',
    context || '(no chunks retrieved)',
    '',
    '--- response ---',
    reply,
    '', '',
  ].join('\n');
  return fs.appendFile(CHAT_LOG_PATH, entry, 'utf-8')
    .catch((err) => console.warn('[chat] could not write chat_log.txt:', err.message));
}

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

export const chatRouter = Router();

chatRouter.post('/', wrap(async (req, res) => {
  const { messages, queryEmbedding } = req.body ?? {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: '"messages" must be a non-empty array' });
  }
  const lastMessage = messages[messages.length - 1];
  if (lastMessage?.role !== 'user' || typeof lastMessage?.content !== 'string'
      || !lastMessage.content.trim()) {
    return res.status(400).json({ error: 'last message must be a user message with string "content"' });
  }
  if (!Array.isArray(queryEmbedding)
      || queryEmbedding.some((component) => typeof component !== 'number')) {
    return res.status(400).json({ error: '"queryEmbedding" must be a number array (computed in the browser)' });
  }

  const chunks = await retrieve(queryEmbedding, lastMessage.content);
  const { reply, model, quotesByChunk } = await answer(messages, chunks);

  logChatTrio({ question: lastMessage.content, chunks, reply, model });

  res.json({
    reply,
    model,
    sources: chunks.map(({ text, ...sourceMeta }, chunkIdx) =>   // full text stays server-side
      ({ ...sourceMeta, quotes: quotesByChunk[chunkIdx] })),
  });
}));
