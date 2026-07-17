/**
 * DocumentViewer.jsx — Documents tab: corpus PDF browser + citation targets.
 *
 * The sidebar (via portal) lists every indexed document; the main pane renders
 * the selected PDF with pdf.js, pages lazily rendered on scroll.
 *
 * Citation deep-links: App passes `target` = { docId, chunkId, quotes, citing,
 * query, nonce } when a [n] marker (or source chip) is clicked in Chat. The
 * viewer opens that document, locates the chunk's text in the PDF text layer
 * and lays a light yellow highlight over it, scrolled into view. Location
 * strategy: strip the chunk's embedded heading prefix (prefixLen), normalize
 * both sides, search the chunk's recorded page range first (±1), then the whole
 * document — chunks indexed before page provenance existed still resolve.
 *
 * Highlight priority, all scoped to the specific citation clicked (`citing` =
 * the sentence that [n] sat in, so a chunk cited by several sentences lights a
 * different span per citation): verbatim `quotes` contained in the citing
 * sentence → chunk sentences scoring above threshold (in-browser cosine +
 * keyword bonus) against the citing sentence → against `query` (chip clicks, or
 * a citation with no claim) → the whole chunk.
 */

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import * as pdfjsLib from 'pdfjs-dist';
import { getDocuments, getChunk, documentPdfUrl } from '../api.js';
import { embedTexts } from '../lib/embedder.js';

pdfjsLib.GlobalWorkerOptions.workerSrc =
  new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString();

const SCALE = 1.4;
const HIGHLIGHT_COLOR = 'rgba(255, 235, 59, 0.42)';   // light yellow

// ---------------------------------------------------------------------------
// Text matching
// ---------------------------------------------------------------------------

// Matching is done over SPACE-FREE normalized text: PDF text layers break
// words at line-end hyphens ("informa- tion") and tokenize differently from
// docling, so comparing with spaces removed sidesteps both.
const normWords = (text) =>
  (text || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);

/** Space-free page text + char-range per text item, for offset→item mapping. */
function indexPage(textContent) {
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
const itemsInRange = (pageIndex, rangeStart, rangeEnd) =>
  pageIndex.spans
    .filter((span) => span.end > rangeStart && span.start < rangeEnd)
    .map((span) => span.item);

/**
 * Find the chunk body on one page. Tries the full body, then word-window
 * anchors at a few offsets (front matter and equations often diverge from
 * the text layer even when the rest of the chunk is present verbatim).
 * Returns the matched char range in index.joined, or null.
 */
function matchOnPage(pageIndex, bodyWords) {
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

const splitSentences = (text) =>
  text.split(/(?<=[.!?])\s+(?=[A-Z0-9("'[])/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 25);

const dot = (vecA, vecB) => {
  let sum = 0;
  for (let dim = 0; dim < vecA.length; dim++) sum += vecA[dim] * vecB[dim];
  return sum;
};

// Plural-insensitive content tokens for the keyword bonus.
const keywordTokens = (text) => new Set(
  (text || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, ' ')
    .split(' ')
    .filter((word) => word.length >= 3)
    .map((word) => (word.endsWith('s') ? word.slice(0, -1) : word)),
);

const KW_BONUS   = 0.1;    // per distinct focus token found in the sentence
const KEEP_RATIO = 0.6;    // keep sentences scoring ≥ 60% of the best one

/**
 * Score each sentence of the chunk body against `focus` (the citing sentence,
 * or the query on chip clicks) — in-browser cosine + keyword bonus — and return
 * the ones above threshold; the best sentence always survives. Empty on any
 * failure, and the caller falls back to whole-chunk highlighting.
 */
async function pickSentences(body, focus, onStatus) {
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
    return sentences.filter(
      (_, sentenceIdx) => scores[sentenceIdx] >= bestScore * KEEP_RATIO && scores[sentenceIdx] > 0);
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
async function embedFocus(text) {
  if (!text || text.trim().length < 8) return null;
  try {
    const [embedding] = await embedTexts([text]);
    return { text, embedding };
  } catch (err) {
    console.warn('[viewer] could not embed citing sentence:', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Single page: lazy canvas render + highlight overlay
// ---------------------------------------------------------------------------

function PdfPage({ pdf, pageNum, size, highlightIds, onPageEl }) {
  const holderRef = useRef(null);
  const [rendered, setRendered] = useState(null);   // { canvasUrl?, rects }

  useEffect(() => {
    const holderEl = holderRef.current;
    if (!holderEl) return;
    let cancelled = false;

    const observer = new IntersectionObserver(async ([entry]) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();

      const page = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale: SCALE });
      const canvas = document.createElement('canvas');
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvas, canvasContext: canvas.getContext('2d'), viewport }).promise;
      if (cancelled) return;

      let highlightRects = [];
      if (highlightIds?.length) {
        const textContent = await page.getTextContent();
        highlightRects = highlightIds
          .map((itemIdx) => textContent.items[itemIdx])
          .filter(Boolean)
          .map((textItem) => {
            const deviceTransform = pdfjsLib.Util.transform(viewport.transform, textItem.transform);
            const glyphHeight = Math.hypot(deviceTransform[2], deviceTransform[3]);
            return {
              left: deviceTransform[4],
              top: deviceTransform[5] - glyphHeight,
              width: textItem.width * viewport.scale,
              height: glyphHeight * 1.18,
            };
          });
      }
      if (!cancelled) setRendered({ canvasUrl: canvas.toDataURL('image/png'), rects: highlightRects });
    }, { rootMargin: '600px' });

    observer.observe(holderEl);
    return () => { cancelled = true; observer.disconnect(); };
  }, [pdf, pageNum, highlightIds]);

  return (
    <div
      className="pdf-page"
      ref={(holderEl) => { holderRef.current = holderEl; if (onPageEl) onPageEl(pageNum, holderEl); }}
      style={{ width: size.width, height: size.height }}
    >
      {rendered ? (
        <>
          <img src={rendered.canvasUrl} alt={`Page ${pageNum}`} width={size.width} height={size.height} />
          {rendered.rects.map((highlightRect, rectIdx) => (
            <div
              key={rectIdx}
              className="pdf-highlight"
              style={{ ...highlightRect, background: HIGHLIGHT_COLOR }}
            />
          ))}
        </>
      ) : (
        <span className="pdf-page-num">{pageNum}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------

export default function DocumentViewer({ controlsEl, active, target }) {
  const [docs, setDocs] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [pdf, setPdf] = useState(null);
  const [pageSizes, setPageSizes] = useState([]);
  const [highlights, setHighlights] = useState({});     // pageNum -> itemIds
  const [status, setStatus] = useState(null);
  const [search, setSearch] = useState('');             // sidebar filter text
  const pageEls = useRef(new Map());
  const pendingScrollPage = useRef(null);               // pageNum to scroll to once its element mounts

  useEffect(() => {
    getDocuments()
      .then((response) => setDocs(response.documents))
      .catch((err) => setError(err.message));
  }, []);

  // Load the selected PDF and measure every page for stable lazy-scroll.
  useEffect(() => {
    if (!selectedDocId) return;
    let cancelled = false;
    setPdf(null);
    setPageSizes([]);
    (async () => {
      try {
        // pdf.js v6 only reads src.url — the positional-string form of
        // getDocument(url) from v4 silently resolves to "no url given".
        // wasmUrl: JBIG2/JPX scans decode in wasm; without it scanned pages
        // render blank white. Vendored by npm run fetch:model.
        const doc = await pdfjsLib.getDocument({
          url: documentPdfUrl(selectedDocId),
          wasmUrl: '/models/pdfjs/wasm/',
          standardFontDataUrl: '/models/pdfjs/standard_fonts/',
        }).promise;
        if (cancelled) return;
        const sizes = [];
        for (let pageNum = 1; pageNum <= doc.numPages; pageNum++) {
          const viewport = (await doc.getPage(pageNum)).getViewport({ scale: SCALE });
          sizes.push({ width: viewport.width, height: viewport.height });
        }
        if (cancelled) return;
        setPdf(doc);
        setPageSizes(sizes);
      } catch (err) {
        if (!cancelled) setStatus(`could not load PDF: ${err.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedDocId]);

  // Citation deep-link: open the doc, locate the chunk, highlight + scroll.
  useEffect(() => {
    if (!target) return;
    setHighlights({});
    setSelectedDocId(target.docId);
    setStatus('locating cited passage…');
  }, [target?.nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!target || !pdf || selectedDocId !== target.docId) return;
    let cancelled = false;
    (async () => {
      try {
        const chunk = await getChunk(target.chunkId);
        // Strip the embedded "title — heading\n" prefix; legacy chunks lack
        // prefixLen, but the prefix convention always ends at the first \n.
        const text = chunk.text || '';
        const body = chunk.prefixLen != null
          ? text.slice(chunk.prefixLen)
          : text.slice(text.indexOf('\n') + 1);
        const bodyWords = normWords(body);

        // Recorded page range (±1) first, then everything else.
        const candidatePages = [];
        if (chunk.pages) {
          for (let pageNum = Math.max(1, chunk.pages[0] - 1);
               pageNum <= Math.min(pdf.numPages, chunk.pages[1] + 1); pageNum++) {
            candidatePages.push(pageNum);
          }
        }
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          if (!candidatePages.includes(pageNum)) candidatePages.push(pageNum);
        }

        for (const pageNum of candidatePages) {
          const textContent = await (await pdf.getPage(pageNum)).getTextContent();
          if (cancelled) return;
          const pageIndex = indexPage(textContent);
          const bodyRange = matchOnPage(pageIndex, bodyWords);
          if (bodyRange) {
            // Highlight priority, all scoped to the SPECIFIC citation clicked:
            //   1. verbatim quotes from this citation's own sentence,
            //   2. chunk sentences scoring against the citing sentence,
            //   3. against the query (chip clicks, or a citation with no claim),
            //   4. the whole chunk.
            // A chunk cited by several sentences thus highlights a different
            // span per citation instead of the same one every time.
            let highlightItemIds = itemsInRange(pageIndex, bodyRange.start, bodyRange.end);

            // Only the quotes this citation's sentence actually contains — so
            // sentence 3's quote doesn't light up when you click sentence 1's [n].
            const citingJoined = target.citing ? normWords(target.citing).join('') : '';
            const relevantQuotes = (target.quotes || []).filter((quoteText) => {
              if (!citingJoined) return true;   // chip click: no sentence to scope by
              const quoteJoined = normWords(quoteText).join('');
              return quoteJoined.length >= 12 && citingJoined.includes(quoteJoined);
            });

            const quoteItemIds = [];
            for (const quoteText of relevantQuotes) {
              const quoteJoined = normWords(quoteText).join('');
              if (quoteJoined.length < 12) continue;
              const quoteAt = pageIndex.joined.indexOf(
                quoteJoined, Math.max(0, bodyRange.start - 300));
              if (quoteAt !== -1 && quoteAt < bodyRange.end + 300) {
                quoteItemIds.push(
                  ...itemsInRange(pageIndex, quoteAt, quoteAt + quoteJoined.length));
              }
            }

            if (quoteItemIds.length) {
              highlightItemIds = [...new Set(quoteItemIds)];
            } else {
              const focus = (await embedFocus(target.citing)) || target.query;
              if (cancelled) return;
              if (focus?.embedding) {
                const pickedSentences = await pickSentences(body, focus, setStatus);
                if (cancelled) return;
                const sentenceItemIds = [];
                for (const sentence of pickedSentences) {
                  const sentenceText = normWords(sentence).join('');
                  if (sentenceText.length < 12) continue;
                  const sentenceAt = pageIndex.joined.indexOf(
                    sentenceText, Math.max(0, bodyRange.start - 300));
                  if (sentenceAt !== -1 && sentenceAt < bodyRange.end + 300) {
                    sentenceItemIds.push(
                      ...itemsInRange(pageIndex, sentenceAt, sentenceAt + sentenceText.length));
                  }
                }
                if (sentenceItemIds.length) highlightItemIds = [...new Set(sentenceItemIds)];
              }
            }
            setHighlights({ [pageNum]: highlightItemIds });
            setStatus(null);
            pendingScrollPage.current = pageNum;
            const pageEl = pageEls.current.get(pageNum);
            if (pageEl) {
              pageEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
              pendingScrollPage.current = null;
            }
            return;
          }
        }
        // Text not locatable (scanned page, heavy equations): land on the page.
        const fallbackPage = chunk.pages?.[0] ?? 1;
        setStatus('passage could not be pinpointed — showing its page');
        pendingScrollPage.current = fallbackPage;
        pageEls.current.get(fallbackPage)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (err) {
        if (!cancelled) setStatus(`citation lookup failed: ${err.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [target?.nonce, pdf]);   // eslint-disable-line react-hooks/exhaustive-deps

  const registerPage = (pageNum, pageEl) => {
    if (pageEl) pageEls.current.set(pageNum, pageEl);
    else pageEls.current.delete(pageNum);
    if (pageEl && pendingScrollPage.current === pageNum) {
      pendingScrollPage.current = null;
      pageEl.scrollIntoView({ block: 'start' });
    }
  };

  // Plain lexical filter: the search text must appear literally in the title or
  // in one of the authors (case-insensitive). No stemming, no fuzziness.
  const needle = search.trim().toLowerCase();
  const shownDocs = (docs || []).filter((doc) => {
    if (!needle) return true;
    const titleAndAuthors = [doc.title || '', ...(doc.authors || [])].join(' ').toLowerCase();
    return titleAndAuthors.includes(needle);
  });

  const controls = (
    <div className="doc-list">
      <div className="control-label">Documents</div>
      {error && <div className="doc-list-error">{error}</div>}
      <input
        className="doc-search"
        type="search"
        value={search}
        placeholder="Search title or author…"
        aria-label="Search documents by title or author"
        onChange={(event) => setSearch(event.target.value)}
      />
      {shownDocs.map((doc) => (
        <button
          key={doc.docId}
          className={`doc-item ${selectedDocId === doc.docId ? 'active' : ''}`}
          onClick={() => { setHighlights({}); setStatus(null); setSelectedDocId(doc.docId); }}
          title={doc.filename}
        >
          <span className="doc-item-title">{doc.title}</span>
          {doc.authors?.length > 0 && (
            <span className="doc-item-authors">{doc.authors.slice(0, 3).join(', ')}</span>
          )}
        </button>
      ))}
      {docs?.length > 0 && shownDocs.length === 0 && (
        <div className="doc-list-error">No document matches “{search.trim()}”.</div>
      )}
      {docs && docs.length === 0 && <div className="doc-list-error">No documents indexed yet.</div>}
    </div>
  );

  return (
    <div className="pdf-wrap">
      {active && controlsEl && createPortal(controls, controlsEl)}
      {status && <div className="pdf-status">{status}</div>}
      {!selectedDocId ? (
        <div className="viz-empty">
          <h2>Documents</h2>
          <p>Pick a document from the sidebar, or click a citation in Chat to jump straight to the cited passage.</p>
        </div>
      ) : !pdf ? (
        <div className="viz-empty"><p>loading PDF…</p></div>
      ) : (
        <div className="pdf-scroll">
          {pageSizes.map((size, pageIdx) => (
            <PdfPage
              key={`${selectedDocId}_${pageIdx + 1}`}
              pdf={pdf}
              pageNum={pageIdx + 1}
              size={size}
              highlightIds={highlights[pageIdx + 1] || null}
              onPageEl={registerPage}
            />
          ))}
        </div>
      )}
    </div>
  );
}
