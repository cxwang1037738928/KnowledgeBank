/**
 * test_api.js — auth + collections + chats + documents API against the real
 * Postgres (docker compose up -d postgres; same database as the app — the
 * test cleans up its own rows).
 *
 * Spawns the server on PORT=3998, then walks the API:
 *   register/login → create collection (orb color) → upload PDF → open chat
 *   on it → cross-user isolation → collection delete cascades everything.
 *
 * Run: node tests/test_api.js
 */

import 'dotenv/config';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { PrismaClient } from '@prisma/client';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 3998;
const BASE = `http://localhost:${PORT}`;
const TEST_EMAIL = 'test-api@opencrawl.local';
const TEST_EMAIL_2 = 'test-api-2@opencrawl.local';
const TEST_PDF = path.join(ROOT, 'tests', 'test-input',
  'Organophosphorus chemistry_ from model to application.pdf');

const prisma = new PrismaClient();
let failures = 0;

function check(name, condition, detail = '') {
  const mark = condition ? 'PASS' : 'FAIL';
  if (!condition) failures++;
  console.log(`[test_api] ${mark}  ${name}${detail ? ` — ${detail}` : ''}`);
}

async function api(token, method, route, body) {
  const isForm = body instanceof FormData;
  const response = await fetch(`${BASE}${route}`, {
    method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body && !isForm ? { 'Content-Type': 'application/json' } : {}),
    },
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });
  return { status: response.status, body: await response.json().catch(() => ({})) };
}

/** Remove any rows a previous (crashed) run left behind. */
const cleanup = () =>
  prisma.user.deleteMany({ where: { email: { in: [TEST_EMAIL, TEST_EMAIL_2] } } });

async function waitForServer() {
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const response = await fetch(`${BASE}/api/auth/me`);
      if (response.ok) return;
    } catch { /* not up yet */ }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error('server did not come up on port ' + PORT);
}

const server = spawn(process.execPath, ['backend/server.js'], {
  cwd: ROOT,
  env: { ...process.env, PORT: String(PORT) },
  stdio: ['ignore', 'ignore', 'inherit'],
});

try {
  await waitForServer();
  await cleanup();

  // ── Auth: everything but /api/auth requires a login ──
  const noAuth = await api(null, 'GET', '/api/collections');
  check('not logged in rejected (401)', noAuth.status === 401);

  const meAnonymous = await api(null, 'GET', '/api/auth/me');
  check('/me without token returns user: null', meAnonymous.body.user === null);

  const register = await api(null, 'POST', '/api/auth/register',
    { email: TEST_EMAIL, password: 'secret1' });
  check('register returns 201 + token', register.status === 201 && !!register.body.token);

  const badLogin = await api(null, 'POST', '/api/auth/login',
    { email: TEST_EMAIL, password: 'wrong-password' });
  check('wrong password rejected (401)', badLogin.status === 401);

  const login = await api(null, 'POST', '/api/auth/login',
    { email: TEST_EMAIL, password: 'secret1' });
  check('login returns token', login.status === 200 && !!login.body.token);
  const token = login.body.token;

  // ── Collections ──
  const noName = await api(token, 'POST', '/api/collections', {});
  check('collection without name rejected (400)', noName.status === 400);

  const created = await api(token, 'POST', '/api/collections', { name: 'papers', crawler: 'sapphire' });
  check('create collection with orb color',
    created.status === 201 && !!created.body.collection.color);
  const collectionId = created.body.collection.id;

  const second = await api(token, 'POST', '/api/collections', { name: 'misc' });
  check('second collection gets a different color',
    second.body.collection.color !== created.body.collection.color);

  // ── Documents ──
  const pdfBytes = await fs.readFile(TEST_PDF);
  const form = new FormData();
  form.append('files', new Blob([pdfBytes], { type: 'application/pdf' }), 'test.pdf');
  const upload = await api(token, 'POST', `/api/collections/${collectionId}/documents`, form);
  check('upload PDF', upload.status === 201 && upload.body.uploaded === 1,
    JSON.stringify(upload.body.results?.[0] ?? upload.body));

  const dupForm = new FormData();
  dupForm.append('files', new Blob([pdfBytes], { type: 'application/pdf' }), 'again.pdf');
  const dupUpload = await api(token, 'POST', `/api/collections/${collectionId}/documents`, dupForm);
  check('duplicate content rejected', dupUpload.body.uploaded === 0);

  // ── Chats (bound to a collection) ──
  const chatNoCollection = await api(token, 'POST', '/api/chats', {});
  check('chat without collectionId rejected (400)', chatNoCollection.status === 400);

  const chat = await api(token, 'POST', '/api/chats', { collectionId });
  check('chat opens on collection with its orb color',
    chat.status === 201 && chat.body.chat.collection?.id === collectionId
    && !!chat.body.chat.collection?.color);
  const chatId = chat.body.chat.id;

  // ── Cross-user isolation ──
  const other = await api(null, 'POST', '/api/auth/register',
    { email: TEST_EMAIL_2, password: 'secret2' });
  const otherToken = other.body.token;
  const foreignCollection = await api(otherToken, 'GET', `/api/collections/${collectionId}/documents`);
  check('another user cannot see the collection (404)', foreignCollection.status === 404);
  const foreignChat = await api(otherToken, 'GET', `/api/chats/${chatId}`);
  check('another user cannot see the chat (404)', foreignChat.status === 404);

  // ── Collection delete cascades everything under it ──
  const removed = await api(token, 'DELETE', `/api/collections/${collectionId}`);
  check('delete collection', removed.status === 200);
  const orphanDocs = await prisma.document.count({ where: { collectionId } });
  const orphanChats = await prisma.chat.count({ where: { collectionId } });
  check('documents + chats cascade-deleted', orphanDocs === 0 && orphanChats === 0);
} catch (err) {
  failures++;
  console.error('[test_api] ERROR:', err.message);
} finally {
  await cleanup().catch(() => {});
  await prisma.$disconnect();
  server.kill();
}

console.log(failures === 0 ? '[test_api] all checks passed' : `[test_api] ${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
