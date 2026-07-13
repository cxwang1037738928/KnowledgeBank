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

const EMBED_MODEL = 'Xenova/all-MiniLM-L12-v2';   // must match the corpus embedder

let _embedder = null;

async function getEmbedder(onStatus) {
  if (!_embedder) {
    _embedder = (async () => {
      onStatus('downloading embedding model…');
      const { pipeline, env } = await import('@xenova/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = false;
      return pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
    })().catch((e) => {
      _embedder = null;       // allow retry after a failed download
      throw e;
    });
  }
  return _embedder;
}

async function embedQuery(text, onStatus) {
  const extractor = await getEmbedder(onStatus);
  onStatus('embedding your question…');
  const out = await extractor([text], { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

function Sources({ sources }) {
  if (!sources?.length) return null;
  // One chip per document, best-scoring chunk first.
  const byDoc = [];
  for (const s of sources) {
    if (!byDoc.some((d) => d.docId === s.docId)) byDoc.push(s);
  }
  return (
    <div className="msg-sources">
      {byDoc.map((s) => (
        <span className="source-chip" key={s.docId} title={`${s.heading || 'document'} · score ${s.score}`}>
          {s.filename.replace(/\.pdf$/i, '')}
        </span>
      ))}
    </div>
  );
}

export default function Chat({ active }) {
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
      setMessages((m) => [...m, { role: 'assistant', content: reply, sources, model }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', error: e.message }]);
    } finally {
      setStatus(null);
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
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
          {messages.map((m, i) =>
            m.role === 'user' ? (
              <div className="msg user" key={i}>{m.content}</div>
            ) : m.error ? (
              <div className="msg assistant error" key={i}>{m.error}</div>
            ) : (
              <div className="msg assistant" key={i}>
                <div className="msg-body">{m.content}</div>
                <Sources sources={m.sources} />
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
          onChange={(e) => setInput(e.target.value)}
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
