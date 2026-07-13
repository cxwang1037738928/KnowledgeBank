/**
 * embedder.js — shared in-browser MiniLM embedder (transformers.js).
 *
 * One lazily-created pipeline for the whole app: Chat embeds queries with it,
 * DocumentViewer embeds chunk sentences to pick which ones to highlight.
 * Must stay the same model the corpus was embedded with (backend/extraction/
 * embed.js).
 *
 * The browser cache is off by design, so the model is re-fetched every
 * session — from our own backend (/models, vendored by npm run fetch:model)
 * rather than huggingface.co, so chat doesn't depend on the public internet.
 * Falls back to the HF hub with a console warning if models/ is absent.
 */

export const EMBED_MODEL = 'Xenova/all-MiniLM-L12-v2';

let _embedder = null;

export async function getEmbedder(onStatus = () => {}) {
  if (!_embedder) {
    _embedder = (async () => {
      onStatus('loading embedding model…');
      const { pipeline, env } = await import('@xenova/transformers');

      env.useBrowserCache = false;
      env.allowLocalModels = true;
      env.localModelPath = '/models/';
      env.backends.onnx.wasm.wasmPaths = '/models/ort/';

      try {
        return await pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
      } catch (err) {
        console.warn('[embedder] local model load failed, falling back to huggingface.co:', err);
        onStatus('downloading embedding model from huggingface…');
        env.allowLocalModels = false;
        return pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
      }
    })().catch((err) => {
      _embedder = null;       // allow retry after a failed load
      throw err;
    });
  }
  return _embedder;
}

/** Embed texts → array of L2-normalized vectors (plain arrays). */
export async function embedTexts(texts, onStatus = () => {}) {
  const extractor = await getEmbedder(onStatus);
  const tensor = await extractor(texts, { pooling: 'mean', normalize: true });
  const dimensions = tensor.dims[tensor.dims.length - 1];
  return texts.map((_, textIdx) =>
    Array.from(tensor.data.slice(textIdx * dimensions, (textIdx + 1) * dimensions)));
}
