/**
 * routes/chats.js — /api/chats (login required; server.js applies
 * `requireAuth` first). A chat belongs to a collection; ownership is checked
 * through the collection's owner.
 *
 *   GET    /            — the owner's chats (newest activity first), each
 *                         with its collection's {id, name, color, crawler}
 *   POST   /            — create a chat {collectionId (required), title?}
 *   GET    /:chatId     — one chat incl. conversation
 *   PATCH  /:chatId     — {title?} rename · {conversation?} rewrite (the UI
 *                         uses the latter to persist deleted Q/A pairs)
 *   DELETE /:chatId     — delete the chat (its collection stays)
 *   POST   /:chatId/chat — RAG chat (chat.js)
 */

import 'dotenv/config';
import { Router } from 'express';
import { prisma } from '../db.js';
import { chatRouter } from './chat.js';

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

const chatSummary = (chat) => ({
  id:         chat.id,
  title:      chat.title,
  collection: chat.collection
    ? {
        id:      chat.collection.id,
        name:    chat.collection.name,
        color:   chat.collection.color,
        crawler: chat.collection.crawler,
      }
    : undefined,
  createdAt: chat.createdAt,
  updatedAt: chat.updatedAt,
});

export const chatsRouter = Router();

chatsRouter.get('/', wrap(async (req, res) => {
  const chats = await prisma.chat.findMany({
    where: { collection: { userId: req.user.id } },
    orderBy: { updatedAt: 'desc' },
    include: { collection: true },
  });
  res.json({ chats: chats.map(chatSummary) });
}));

chatsRouter.post('/', wrap(async (req, res) => {
  const { collectionId, title } = req.body ?? {};
  if (!Number.isInteger(collectionId)) throw httpError(400, '"collectionId" is required');
  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId: req.user.id },
  });
  if (!collection) throw httpError(404, `No collection ${collectionId}`);
  const chat = await prisma.chat.create({
    data: {
      title: typeof title === 'string' && title.trim() ? title.trim() : 'New chat',
      collectionId: collection.id,
    },
    include: { collection: true },
  });
  res.status(201).json({ chat: chatSummary(chat) });
}));

/** Everything below /:chatId is ownership-checked (via the collection) here;
 * req.chat carries the chat row with its collection included. */
const loadOwnedChat = wrap(async (req, res, next) => {
  const chatId = parseInt(req.params.chatId, 10);
  if (!Number.isInteger(chatId)) throw httpError(400, 'chatId must be an integer');
  const chat = await prisma.chat.findFirst({
    where: { id: chatId, collection: { userId: req.user.id } },
    include: { collection: true },
  });
  if (!chat) throw httpError(404, `No chat ${chatId}`);
  req.chat = chat;
  next();
});

chatsRouter.use('/:chatId', loadOwnedChat);

chatsRouter.get('/:chatId', (req, res) => {
  res.json({ chat: { ...chatSummary(req.chat), conversation: req.chat.conversation } });
});

chatsRouter.patch('/:chatId', wrap(async (req, res) => {
  const { title, conversation } = req.body ?? {};
  const data = {};
  if (title !== undefined) {
    if (typeof title !== 'string' || !title.trim()) throw httpError(400, '"title" must be a non-empty string');
    data.title = title.trim();
  }
  if (conversation !== undefined) {
    const isMessage = (message) =>
      message && typeof message.role === 'string' && typeof message.content === 'string';
    if (!Array.isArray(conversation) || !conversation.every(isMessage)) {
      throw httpError(400, '"conversation" must be an array of {role, content} messages');
    }
    data.conversation = conversation;
  }
  if (Object.keys(data).length === 0) throw httpError(400, 'Nothing to update');
  const chat = await prisma.chat.update({
    where: { id: req.chat.id },
    data,
    include: { collection: true },
  });
  res.json({ chat: chatSummary(chat) });
}));

chatsRouter.delete('/:chatId', wrap(async (req, res) => {
  await prisma.chat.delete({ where: { id: req.chat.id } });
  res.json({ ok: true, id: req.chat.id });
}));

chatsRouter.use('/:chatId/chat', chatRouter);
