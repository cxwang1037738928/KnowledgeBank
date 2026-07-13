// Thin fetch wrappers over the Express /api/corpus routes.

async function request(url, options) {
  const resp = await fetch(url, options);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

export const getEmbeddingMap = () => request('/api/corpus/embedding-map');
export const getGraph        = () => request('/api/corpus/graph');
export const getModels       = () => request('/api/corpus/models');

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
