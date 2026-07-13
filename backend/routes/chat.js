/**
 * routes/chat.js — /api/chat
 *
 * RAG chat over the corpus. The frontend embeds the user's question in the
 * browser (same MiniLM model as the corpus, browser cache disabled) and sends
 * the vector along; retrieval + Ollama answering happen here (retriever/).
 *
 * POST /api/chat
 *   Request:  { messages: [{role:'user'|'assistant', content}], queryEmbedding: number[] }
 *   Response: { reply, model, sources: [{chunkId, docId, filename, heading, pages, sim, boost, score}] }
 *             sources[n-1] is what a [n] citation marker in reply refers to
 *   Errors:   400 bad payload · 502 Ollama failure · 503 missing corpus/model
 */

import { Router } from 'express';
import { retrieve, answer } from '../retriever/retriever.js';

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
  const { reply, model } = await answer(messages, chunks);

  res.json({
    reply,
    model,
    sources: chunks.map(({ text, ...sourceMeta }) => sourceMeta),   // full text stays server-side
  });
}));
