/**
 * middleware/auth.js — JWT auth. Login is required to use the app.
 *
 * signToken(user)      — issues a token carrying {sub, email, isAdmin}.
 * userFromRequest(req) — the user in the Bearer token, or null.
 * requireAuth          — 401 unless a valid token is present; attaches
 *                        req.user = {id, email, isAdmin}.
 */

import 'dotenv/config';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'opencrawl-local-dev-secret';
const TOKEN_TTL  = process.env.JWT_TTL || '7d';

export function signToken(user) {
  return jwt.sign(
    { sub: user.id, email: user.email, isAdmin: user.isAdmin },
    JWT_SECRET,
    { expiresIn: TOKEN_TTL },
  );
}

/** The user carried by a Bearer token, or null (missing/expired/invalid). */
export function userFromRequest(req) {
  const header = req.headers.authorization || '';
  if (!header.startsWith('Bearer ')) return null;
  try {
    const payload = jwt.verify(header.slice(7), JWT_SECRET);
    return { id: payload.sub, email: payload.email, isAdmin: !!payload.isAdmin };
  } catch {
    return null;
  }
}

export function requireAuth(req, res, next) {
  req.user = userFromRequest(req);
  if (!req.user) return res.status(401).json({ error: 'Not logged in' });
  next();
}
