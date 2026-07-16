/**
 * embedder.js — MiniLM sentence embeddings in the backend.
 *
 * Retrieval itself needs no model (it's math over the browser-supplied query
 * vector and embeddings.json). Citation grounding does: to tell a reworded
 * paraphrase from a fabrication we compare the claim to the cited chunk in
 * embedding space, and only the browser had a model until now. This loads the
 * SAME model + settings as embed.js (SAPPHIRE_EMBEDDING_MODEL, quantized, mean
 * pooling, L2-normalized) so claim vectors live in the corpus's vector space.
 *
 * Lazy singleton: the model is pulled in only on the first chat whose reply has
 * a paraphrase citation to ground, not at server start.
 */

import { pipeline } from '@xenova/transformers';

const MODEL = process.env.SAPPHIRE_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L12-v2';

let _extractor = null;

/** Embed texts → array of L2-normalized vectors (dot product = cosine). */
export async function embedTexts(texts) {
  if (!texts.length) return [];
  _extractor ??= await pipeline('feature-extraction', MODEL, { quantized: true });
  const output = await _extractor(texts, { pooling: 'mean', normalize: true });
  const dims = output.data.length / texts.length;
  return texts.map((_, textIdx) =>
    Array.from(output.data.slice(textIdx * dims, (textIdx + 1) * dims)));
}
