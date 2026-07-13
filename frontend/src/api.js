// Thin fetch wrappers over the Express /api/corpus routes.

async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

export const getEmbeddingMap = () => request('/api/corpus/embedding-map');
export const getGraph        = () => request('/api/corpus/graph');
export const getModels       = () => request('/api/corpus/models');
export const getDocuments    = () => request('/api/corpus/documents');
export const getChunk        = (chunkId) => request(`/api/corpus/chunks/${encodeURIComponent(chunkId)}`);
export const documentPdfUrl  = (docId) => `/api/corpus/documents/${encodeURIComponent(docId)}/pdf`;

export const saveSettings = (updates) =>
  request('/api/corpus/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });

export const postChat = (payload) =>
  request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
