/**
 * DocumentViewer_utils.jsx — non-component helpers for DocumentViewer.jsx:
 * PDF text-layer matching (space-free normalization, page indexing, chunk
 * anchoring) and citation-focused sentence selection/scoring.
 */

import { embedTexts } from '../lib/embedder.js';

// ---------------------------------------------------------------------------
// Text matching
// ---------------------------------------------------------------------------

// Matching is done over SPACE-FREE normalized text: PDF text layers break
// words at line-end hyphens ("informa- tion") and tokenize differently from
// docling, so comparing with spaces removed sidesteps both.
export const normWords = (text) =>
  (text || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);

/** Space-free page text + char-range per text item, for offset→item mapping. */
export function indexPage(textContent) {
  const itemSpans = [];
  let joinedText = '';
  textContent.items.forEach((textItem, itemIdx) => {
    const normalized = normWords(textItem.str).join('');
    if (!normalized) return;
    itemSpans.push({ item: itemIdx, start: joinedText.length, end: joinedText.length + normalized.length });
    joinedText += normalized;
  });
  return { joined: joinedText, spans: itemSpans };
}

/** Item indices whose normalized range overlaps [rangeStart, rangeEnd). */
export const itemsInRange = (pageIndex, rangeStart, rangeEnd) =>
  pageIndex.spans
    .filter((span) => span.end > rangeStart && span.start < rangeEnd)
    .map((span) => span.item);

/**
 * Find the chunk body on one page. Tries the full body, then word-window
 * anchors at a few offsets (front matter and equations often diverge from
 * the text layer even when the rest of the chunk is present verbatim).
 * Returns the matched char range in index.joined, or null.
 */
export function matchOnPage(pageIndex, bodyWords) {
  if (!pageIndex.joined) return null;
  const bodyWordCount = bodyWords.length;
  const anchors = [[0, bodyWordCount]];
  for (const wordOffset of [0, 8, 20]) {
    for (const anchorLength of [30, 15, 8]) {
      if (wordOffset + anchorLength <= bodyWordCount) anchors.push([wordOffset, anchorLength]);
    }
  }

  for (const [wordOffset, anchorLength] of anchors) {
    const anchorText = bodyWords.slice(wordOffset, wordOffset + anchorLength).join('');
    if (anchorText.length < 16) continue;   // too short to trust (e.g. "By C. E. SHANNON.")
    const anchorAt = pageIndex.joined.indexOf(anchorText);
    if (anchorAt === -1) continue;

    // Extend the match to the chunk's tail if it appears later on the page.
    let start = anchorAt;
    let end = anchorAt + anchorText.length;
    if (wordOffset > 0 || anchorLength < bodyWordCount) {
      const tailText = bodyWords.slice(-10).join('');
      const tailAt = pageIndex.joined.indexOf(tailText, end);
      if (tailAt !== -1) end = tailAt + tailText.length;
    }
    // If the anchor skipped the head, pull the start back to it when nearby.
    if (wordOffset > 0) {
      const headText = bodyWords.slice(0, 6).join('');
      const headAt = pageIndex.joined.lastIndexOf(headText, anchorAt);
      if (headAt !== -1 && anchorAt - headAt < 600) start = headAt;
    }
    return { start, end };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Sentence selection — highlight only what answers the query
// ---------------------------------------------------------------------------

export const splitSentences = (text) =>
  text.split(/(?<=[.!?])\s+(?=[A-Z0-9("'[])/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 25);

export const dot = (vecA, vecB) => {
  let sum = 0;
  for (let dim = 0; dim < vecA.length; dim++) sum += vecA[dim] * vecB[dim];
  return sum;
};

// Plural-insensitive content tokens for the keyword bonus.
export const keywordTokens = (text) => new Set(
  (text || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, ' ')
    .split(' ')
    .filter((word) => word.length >= 3)
    .map((word) => (word.endsWith('s') ? word.slice(0, -1) : word)),
);

const KW_BONUS   = 0.1;    // per distinct focus token found in the sentence
const KEEP_RATIO = 0.6;    // loose focus (question): keep sentences ≥ 60% of the best
const TIE_MARGIN = 0.05;   // precise focus (citing sentence): the best + absolute near-ties

/**
 * Score each sentence of the chunk body against `focus` — in-browser cosine +
 * keyword bonus — and return the ones to highlight; the best sentence always
 * survives. `precise` = focus is the citation's own citing sentence: keep only
 * the best match and near-ties. Same-domain sentences all sit on a high cosine
 * floor, so the 60% ratio keeps half the chunk — acceptable when matching a
 * vague question (chip clicks), wrong when the claim names one exact sentence.
 * Empty on any failure, and the caller falls back to whole-chunk highlighting.
 */
export async function pickSentences(body, focus, onStatus, precise = false) {
  try {
    const sentences = splitSentences(body);
    if (sentences.length <= 1) return sentences;
    onStatus('scoring the cited passage…');
    const sentenceVectors = await embedTexts(sentences, onStatus);
    const focusTokens = keywordTokens(focus.text);
    const scores = sentences.map((sentence, sentenceIdx) => {
      const keywordHits = [...keywordTokens(sentence)].filter((token) => focusTokens.has(token)).length;
      return dot(sentenceVectors[sentenceIdx], focus.embedding) + KW_BONUS * keywordHits;
    });
    const bestScore = Math.max(...scores);
    const keeps = precise
      ? (score) => score >= bestScore - TIE_MARGIN
      : (score) => score >= bestScore * KEEP_RATIO;
    return sentences.filter(
      (_, sentenceIdx) => keeps(scores[sentenceIdx]) && scores[sentenceIdx] > 0);
  } catch (err) {
    console.warn('[viewer] sentence scoring failed, highlighting whole chunk:', err);
    return [];
  }
}

/**
 * Embed the citing sentence into the same {text, embedding} shape pickSentences
 * scores against — so a citation is highlighted by what ITS sentence says, not
 * the whole question. Null (→ caller falls back to the query) on empty or error.
 */
export async function embedFocus(text) {
  if (!text || text.trim().length < 8) return null;
  try {
    const [embedding] = await embedTexts([text]);
    return { text, embedding };
  } catch (err) {
    console.warn('[viewer] could not embed citing sentence:', err);
    return null;
  }
}
