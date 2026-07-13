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

// Small models drift off the [n] format even when told not to; the backend
// normalizes what it can, and this accepts the leftovers ([n1], [#1], [ref 1]).
const CITE_RE = /(\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*\d+(?:\s*,\s*\d+)*\s*\])/g;

/** [n] markers → buttons that open the cited chunk in the Documents tab. */
function citeInline(text, sources, query, onCitation, keyBase) {
  if (!sources?.length || !onCitation) return text;
  return text.split(CITE_RE).map((part, partIdx) => {
    const citeMatch = part.match(/^\[\s*(?:n|N|#|source|excerpt|ref)?\s*[.:]?\s*(\d+(?:\s*,\s*\d+)*)\s*\]$/);
    if (!citeMatch) return part;
    const sourceNumbers = citeMatch[1].split(',').map((number) => parseInt(number, 10));
    if (sourceNumbers.some((sourceNumber) => sourceNumber < 1 || sourceNumber > sources.length)) return part;
    return (
      <span className="citation-group" key={`${keyBase}c${partIdx}`}>
        {sourceNumbers.map((sourceNumber) => {
          const source = sources[sourceNumber - 1];
          return (
            <button
              key={sourceNumber}
              className="citation-link"
              title={`${source.filename}${source.heading ? ` — ${source.heading}` : ''}${source.pages ? ` · p.${source.pages[0]}` : ''}`}
              onClick={() => onCitation(source, query)}
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
  const cite = (segment, key) => citeInline(segment, sources, query, onCitation, key);
  const blocks = [];
  const lines = (text || '').split('\n');

  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    if (!line.trim()) continue;

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const [, , headingText] = headingMatch;
      blocks.push(
        <div className="md-h" key={lineIdx}>{inlineMd(headingText, `h${lineIdx}`, cite)}</div>
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
      blocks.push(
        <blockquote key={lineIdx}>{inlineMd(quoteLines.join(' '), `q${lineIdx}`, cite)}</blockquote>
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
            <li key={itemIdx}>{inlineMd(listItem, `l${lineIdx}_${itemIdx}`, cite)}</li>
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
    blocks.push(<p key={lineIdx}>{inlineMd(paragraphLines.join(' '), `p${lineIdx}`, cite)}</p>);
  }

  return blocks;
}

/** Excerpt numbers the reply actually cites (post-repair, so they're verified). */
function citedNumbers(text) {
  const numbers = new Set();
  for (const marker of (text || '').matchAll(new RegExp(CITE_RE.source, 'g'))) {
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

export default function Chat({ active, onCitation }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState(null);   // busy string | null
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

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
              <div className="msg assistant error" key={messageIdx}>{message.error}</div>
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
