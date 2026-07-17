/**
 * Chat.jsx — RAG chat over the corpus.
 *
 * The question is embedded HERE, in the browser, with the same MiniLM model
 * the corpus was embedded with (transformers.js; browser cache deliberately
 * off, so the model re-downloads each session). The vector goes to /api/chat,
 * which retrieves chunks (cosine × category keyword boost) and answers with
 * the Ollama reasoning model.
 */

import { useEffect, useRef, useState } from 'react';
import { postChat } from '../api.js';
import { embedTexts } from '../lib/embedder.js';

async function embedQuery(text, onStatus) {
  onStatus('embedding your question…');
  const [queryVector] = await embedTexts([text], onStatus);
  return queryVector;
}

// Verified citations. CITE_MARKER_RE matches a whole [n] marker (plus the
// [n1]/[#1]/[ref 1] variants small models drift into); CITE_NUMBERS_RE anchors
// it and captures the numbers.
const CITE_MARKER_RE = /\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*\d+(?:\s*,\s*\d+)*\s*\]/;
const CITE_NUMBERS_RE = /^\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*(\d+(?:\s*,\s*\d+)*)\s*\]$/;

// Ungrounded citations (retriever.js): [n!] still points at the model's cited
// excerpt but is flagged; a bare [!] is one with no usable excerpt number.
const UNSUPPORTED_MARKER = '[!]';
const FLAGGED_CITE_RE = /\[\s*\d+(?:\s*,\s*\d+)*\s*!\s*\]/;
const FLAGGED_NUMBERS_RE = /^\[\s*(\d+(?:\s*,\s*\d+)*)\s*!\s*\]$/;
const UNSUPPORTED_TIP =
  'This information is possibly hallucinated. No retrieved excerpt contains this exact sentence.';

// One capture group, so split() hands markers back interleaved with the prose.
const SEGMENT_RE = new RegExp(`(${CITE_MARKER_RE.source}|${FLAGGED_CITE_RE.source}|\\[!\\])`, 'g');

/**
 * The claim sentence each marker sits in — one entry per marker occurrence, in
 * document order, so clicking a specific marker can highlight what ITS sentence
 * refers to rather than every sentence citing the chunk. Works on the block's
 * markdown-stripped text with the markers themselves removed.
 */
function markerCitingSentences(blockText) {
  const plain = (blockText || '').replace(/\*\*|\*|`/g, '');
  const claims = [];
  for (const sentence of plain.split(/(?<=[.!?])\s+/)) {
    const markerCount = (sentence.match(SEGMENT_RE) || []).length;
    if (!markerCount) continue;
    const claim = sentence.replace(SEGMENT_RE, ' ').replace(/\s+/g, ' ').trim();
    for (let occurrence = 0; occurrence < markerCount; occurrence++) claims.push(claim);
  }
  return claims;
}

/**
 * [n] → a button opening the cited chunk; [n!] → the same but flagged unsupported
 * (red, with the disclaimer); [!] → a warning with no chunk to open.
 * `claimCursor` = { claims, next }: the block's citing sentences, consumed one
 * per marker in document order so each marker carries its own claim.
 */
function citeInline(text, sources, query, onCitation, keyBase, claimCursor) {
  return text.split(SEGMENT_RE).map((part, partIdx) => {
    if (part === UNSUPPORTED_MARKER) {
      if (claimCursor) claimCursor.next += 1;   // [!] is a marker occurrence too
      // Custom tooltip (not title=): the native one takes a second to appear
      // and is easy to miss; this needs to be readable on a quick hover.
      return (
        <span className="citation-unsupported" key={`${keyBase}u${partIdx}`}
              tabIndex={0} role="img" aria-label={UNSUPPORTED_TIP}>
          !
          <span className="unsupported-tip" role="tooltip" aria-hidden="true">
            {UNSUPPORTED_TIP}
          </span>
        </span>
      );
    }
    const flaggedMatch = part.match(FLAGGED_NUMBERS_RE);
    const citeMatch = flaggedMatch || part.match(CITE_NUMBERS_RE);
    if (!citeMatch) return part;                        // plain prose
    const citing = claimCursor ? claimCursor.claims[claimCursor.next++] : undefined;
    if (!sources?.length || !onCitation) return part;
    const sourceNumbers = citeMatch[1].split(',').map((number) => parseInt(number, 10));
    if (sourceNumbers.some((sourceNumber) => sourceNumber < 1 || sourceNumber > sources.length)) return part;
    return (
      <span className="citation-group" key={`${keyBase}c${partIdx}`}>
        {sourceNumbers.map((sourceNumber) => {
          const source = sources[sourceNumber - 1];
          const openChunk = () => onCitation(source, query, citing);
          // Flagged citation: still opens the model's cited chunk, but red and
          // carrying the disclaimer so the reader knows it's unverified.
          if (flaggedMatch) {
            return (
              <button
                key={sourceNumber}
                className="citation-unsupported citation-unsupported-link"
                onClick={openChunk}
                aria-label={`Excerpt ${sourceNumber}: ${UNSUPPORTED_TIP}`}
              >
                {sourceNumber}!
                <span className="unsupported-tip" role="tooltip" aria-hidden="true">
                  {UNSUPPORTED_TIP}
                </span>
              </button>
            );
          }
          return (
            <button
              key={sourceNumber}
              className="citation-link"
              title={`${source.filename}${source.heading ? ` — ${source.heading}` : ''}${source.pages ? ` · p.${source.pages[0]}` : ''}`}
              onClick={openChunk}
            >
              {sourceNumber}
            </button>
          );
        })}
      </span>
    );
  });
}

/** Inline markdown the models actually emit: **bold**, *italic*, `code`. */
function inlineMd(text, keyBase, cite) {
  const nodes = [];
  const emphasisPattern = /\*\*(.+?)\*\*|(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)|`([^`]+)`/g;
  let plainStart = 0;
  let emphasisMatch;
  while ((emphasisMatch = emphasisPattern.exec(text)) !== null) {
    if (emphasisMatch.index > plainStart) {
      nodes.push(cite(text.slice(plainStart, emphasisMatch.index), `${keyBase}t${plainStart}`));
    }
    const key = `${keyBase}m${emphasisMatch.index}`;
    const [whole, boldText, italicText, codeText] = emphasisMatch;
    if (boldText !== undefined)        nodes.push(<strong key={key}>{cite(boldText, key)}</strong>);
    else if (italicText !== undefined) nodes.push(<em key={key}>{cite(italicText, key)}</em>);
    else                               nodes.push(<code key={key}>{codeText}</code>);
    plainStart = emphasisMatch.index + whole.length;
  }
  if (plainStart < text.length) nodes.push(cite(text.slice(plainStart), `${keyBase}t${plainStart}`));
  return nodes;
}

/**
 * Minimal markdown renderer for assistant replies — headings, bullet and
 * numbered lists, blockquotes, paragraphs, plus inline emphasis. The models
 * emit markdown whether or not we ask, and raw ### / ** in the bubble reads
 * as garbage. Citation markers stay clickable inside all of it.
 */
function CitedText({ text, sources, query, onCitation }) {
  // Each block gets its own marker→claim cursor, so a [n] in one paragraph or
  // list item is matched to that block's citing sentence, not a neighbour's.
  const makeCite = (blockText) => {
    const claimCursor = { claims: markerCitingSentences(blockText), next: 0 };
    return (segment, key) => citeInline(segment, sources, query, onCitation, key, claimCursor);
  };
  const blocks = [];
  const lines = (text || '').split('\n');

  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    if (!line.trim()) continue;

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const [, , headingText] = headingMatch;
      blocks.push(
        <div className="md-h" key={lineIdx}>{inlineMd(headingText, `h${lineIdx}`, makeCite(headingText))}</div>
      );
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines = [];
      while (lineIdx < lines.length && /^\s*>\s?/.test(lines[lineIdx])) {
        quoteLines.push(lines[lineIdx].replace(/^\s*>\s?/, ''));
        lineIdx++;
      }
      lineIdx--;
      const quoteText = quoteLines.join(' ');
      blocks.push(
        <blockquote key={lineIdx}>{inlineMd(quoteText, `q${lineIdx}`, makeCite(quoteText))}</blockquote>
      );
      continue;
    }

    if (/^\s*(?:[-*+]|\d+[.)])\s+/.test(line)) {
      const listItems = [];
      const isOrdered = /^\s*\d+[.)]\s+/.test(line);
      while (lineIdx < lines.length && /^\s*(?:[-*+]|\d+[.)])\s+/.test(lines[lineIdx])) {
        listItems.push(lines[lineIdx].replace(/^\s*(?:[-*+]|\d+[.)])\s+/, ''));
        lineIdx++;
      }
      lineIdx--;
      const List = isOrdered ? 'ol' : 'ul';
      blocks.push(
        <List key={lineIdx}>
          {listItems.map((listItem, itemIdx) => (
            <li key={itemIdx}>{inlineMd(listItem, `l${lineIdx}_${itemIdx}`, makeCite(listItem))}</li>
          ))}
        </List>
      );
      continue;
    }

    // Paragraph: absorb following non-blank, non-structural lines.
    const paragraphLines = [line];
    while (lineIdx + 1 < lines.length && lines[lineIdx + 1].trim() &&
           !/^\s*(?:[-*+]|\d+[.)])\s+|^\s*>|^#{1,6}\s/.test(lines[lineIdx + 1])) {
      paragraphLines.push(lines[++lineIdx]);
    }
    const paragraphText = paragraphLines.join(' ');
    blocks.push(<p key={lineIdx}>{inlineMd(paragraphText, `p${lineIdx}`, makeCite(paragraphText))}</p>);
  }

  return blocks;
}

/** Excerpt numbers the reply cites with VERIFIED markers — flagged [n!] ones
 * are deliberately excluded, so an unverified claim never earns a source chip. */
function citedNumbers(text) {
  const numbers = new Set();
  for (const marker of (text || '').matchAll(new RegExp(CITE_MARKER_RE.source, 'g'))) {
    for (const number of marker[0].replace(/[^\d,]/g, '').split(',')) {
      if (number) numbers.add(parseInt(number, 10));
    }
  }
  return numbers;
}

function Sources({ sources, reply, query, onCitation }) {
  if (!sources?.length) return null;
  // Chips are the sources the answer cites — a retrieved but uncited chunk is
  // not a source, and linking it sent people to passages nothing referenced.
  const cited = citedNumbers(reply);
  const citedSources = sources.filter((_, sourceIdx) => cited.has(sourceIdx + 1));
  if (!citedSources.length) return null;

  // One chip per document, best-scoring chunk first.
  const bestSourcePerDoc = [];
  for (const source of citedSources) {
    if (!bestSourcePerDoc.some((kept) => kept.docId === source.docId)) bestSourcePerDoc.push(source);
  }
  return (
    <div className="msg-sources">
      {bestSourcePerDoc.map((source) => (
        <button
          className="source-chip"
          key={source.docId}
          title={`${source.heading || 'document'} · score ${source.score} — open in Documents`}
          onClick={() => onCitation?.(source, query)}
        >
          {source.filename.replace(/\.pdf$/i, '')}
        </button>
      ))}
    </div>
  );
}

/**
 * Delete control on a reply: a small × that widens on hover, then asks to
 * confirm before removing the question/answer pair.
 */
function DeletePair({ confirming, onAsk, onConfirm, onCancel }) {
  return (
    <div className="msg-actions">
      {confirming ? (
        <div className="delete-confirm" role="dialog" aria-label="Confirm delete">
          <span>Delete this response?</span>
          <button className="delete-yes" onClick={onConfirm}>Yes</button>
          <button className="delete-no" onClick={onCancel}>Cancel</button>
        </div>
      ) : (
        <button className="delete-pair" onClick={onAsk} aria-label="Delete this pair">
          <span className="delete-mark">×</span>
          <span className="delete-label">delete this pair?</span>
        </button>
      )}
    </div>
  );
}

export default function Chat({ active, onCitation }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState(null);   // busy string | null
  const [confirmingIdx, setConfirmingIdx] = useState(null);   // reply awaiting delete confirmation
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  // Drop a reply and the question above it. They leave the thread AND the
  // history posted with the next question, so the model stops seeing them.
  function deletePair(replyIdx) {
    setMessages((thread) => thread.filter((message, messageIdx) => {
      if (messageIdx === replyIdx) return false;
      return !(messageIdx === replyIdx - 1 && message.role === 'user');
    }));
    setConfirmingIdx(null);
  }

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, status]);

  useEffect(() => {
    if (active) inputRef.current?.focus();
  }, [active]);

  async function send() {
    const question = input.trim();
    if (!question || status) return;

    const history = [...messages, { role: 'user', content: question }];
    setMessages(history);
    setInput('');

    try {
      const queryEmbedding = await embedQuery(question, setStatus);
      setStatus('retrieving + reasoning…');
      const { reply, sources, model } = await postChat({
        messages: history.map(({ role, content }) => ({ role, content })),
        queryEmbedding,
      });
      // Keep the query with the reply: citation clicks pass it to the PDF
      // viewer, which scores the chunk's sentences against it to decide
      // what to highlight.
      const query = { text: question, embedding: queryEmbedding };
      setMessages((thread) => [...thread, { role: 'assistant', content: reply, sources, model, query }]);
    } catch (err) {
      setMessages((thread) => [...thread, { role: 'assistant', error: err.message }]);
    } finally {
      setStatus(null);
    }
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className="chat-wrap">
      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 && !status && (
          <div className="chat-hero">
            <h2>Ask your corpus</h2>
            <p>
              Answers are retrieved from your indexed documents and grounded with
              citations. The question is embedded locally in your browser.
            </p>
          </div>
        )}

        <div className="chat-thread">
          {messages.map((message, messageIdx) =>
            message.role === 'user' ? (
              <div className="msg user" key={messageIdx}>{message.content}</div>
            ) : message.error ? (
              <div className="msg assistant error" key={messageIdx}>
                {message.error}
                <DeletePair
                  confirming={confirmingIdx === messageIdx}
                  onAsk={() => setConfirmingIdx(messageIdx)}
                  onConfirm={() => deletePair(messageIdx)}
                  onCancel={() => setConfirmingIdx(null)}
                />
              </div>
            ) : (
              <div className="msg assistant" key={messageIdx}>
                <div className="msg-body">
                  <CitedText
                    text={message.content}
                    sources={message.sources}
                    query={message.query}
                    onCitation={onCitation}
                  />
                </div>
                <Sources
                  sources={message.sources}
                  reply={message.content}
                  query={message.query}
                  onCitation={onCitation}
                />
                <DeletePair
                  confirming={confirmingIdx === messageIdx}
                  onAsk={() => setConfirmingIdx(messageIdx)}
                  onConfirm={() => deletePair(messageIdx)}
                  onCancel={() => setConfirmingIdx(null)}
                />
              </div>
            )
          )}
          {status && <div className="chat-status">{status}</div>}
        </div>
      </div>

      <div className="composer">
        <textarea
          ref={inputRef}
          rows={1}
          placeholder="Ask about your documents…"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={!!status}
        />
        <button className="btn" onClick={send} disabled={!!status || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
