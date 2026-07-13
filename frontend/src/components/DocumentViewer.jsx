/**
 * DocumentViewer.jsx — Documents tab: corpus PDF browser + citation targets.
 *
 * The sidebar (via portal) lists every indexed document; the main pane renders
 * the selected PDF with pdf.js, pages lazily rendered on scroll.
 *
 * Citation deep-links: App passes `target` = { docId, chunkId, nonce } when a
 * [n] marker (or source chip) is clicked in Chat. The viewer opens that
 * document, locates the chunk's text in the PDF text layer and lays a light
 * yellow highlight over it, scrolled into view. Location strategy: strip the
 * chunk's embedded heading prefix (prefixLen), normalize both sides, search
 * the chunk's recorded page range first (±1), then the whole document —
 * chunks indexed before page provenance existed still resolve.
 */

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import * as pdfjsLib from 'pdfjs-dist';
import { getDocuments, getChunk, documentPdfUrl } from '../api.js';

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
const normWords = (s) =>
  (s || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);

/** Space-free page text + char-range per text item, for offset→item mapping. */
function indexPage(textContent) {
  const spans = [];
  let joined = '';
  textContent.items.forEach((item, i) => {
    const n = normWords(item.str).join('');
    if (!n) return;
    spans.push({ item: i, start: joined.length, end: joined.length + n.length });
    joined += n;
  });
  return { joined, spans };
}

/** Item indices whose normalized range overlaps [from, to). */
const itemsInRange = (index, from, to) =>
  index.spans.filter((s) => s.end > from && s.start < to).map((s) => s.item);

/**
 * Find the chunk body on one page. Tries the full body, then word-window
 * anchors at a few offsets (front matter and equations often diverge from
 * the text layer even when the rest of the chunk is present verbatim).
 * Returns { itemIds } or null.
 */
function matchOnPage(index, bodyWords) {
  if (!index.joined) return null;
  const W = bodyWords.length;
  const tries = [[0, W]];
  for (const off of [0, 8, 20]) {
    for (const take of [30, 15, 8]) {
      if (off + take <= W) tries.push([off, take]);
    }
  }

  for (const [off, take] of tries) {
    const needle = bodyWords.slice(off, off + take).join('');
    if (needle.length < 16) continue;   // too short to trust (e.g. "By C. E. SHANNON.")
    const at = index.joined.indexOf(needle);
    if (at === -1) continue;

    // Extend the match to the chunk's tail if it appears later on the page.
    let start = at;
    let end = at + needle.length;
    if (off > 0 || take < W) {
      const tail = bodyWords.slice(-10).join('');
      const tailAt = index.joined.indexOf(tail, end);
      if (tailAt !== -1) end = tailAt + tail.length;
    }
    // If the anchor skipped the head, pull the start back to it when nearby.
    if (off > 0) {
      const head = bodyWords.slice(0, 6).join('');
      const headAt = index.joined.lastIndexOf(head, at);
      if (headAt !== -1 && at - headAt < 600) start = headAt;
    }
    return { itemIds: itemsInRange(index, start, end) };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Single page: lazy canvas render + highlight overlay
// ---------------------------------------------------------------------------

function PdfPage({ pdf, pageNum, size, highlightIds, pageRef }) {
  const holderRef = useRef(null);
  const [rendered, setRendered] = useState(null);   // { canvasUrl?, rects }

  useEffect(() => {
    const el = holderRef.current;
    if (!el) return;
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

      let rects = [];
      if (highlightIds?.length) {
        const tc = await page.getTextContent();
        rects = highlightIds
          .map((i) => tc.items[i])
          .filter(Boolean)
          .map((item) => {
            const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
            const h = Math.hypot(tx[2], tx[3]);
            return {
              left: tx[4],
              top: tx[5] - h,
              width: item.width * viewport.scale,
              height: h * 1.18,
            };
          });
      }
      if (!cancelled) setRendered({ canvasUrl: canvas.toDataURL('image/png'), rects });
    }, { rootMargin: '600px' });

    observer.observe(el);
    return () => { cancelled = true; observer.disconnect(); };
  }, [pdf, pageNum, highlightIds]);

  return (
    <div
      className="pdf-page"
      ref={(el) => { holderRef.current = el; if (pageRef) pageRef(pageNum, el); }}
      style={{ width: size.width, height: size.height }}
    >
      {rendered ? (
        <>
          <img src={rendered.canvasUrl} alt={`Page ${pageNum}`} width={size.width} height={size.height} />
          {rendered.rects.map((r, i) => (
            <div
              key={i}
              className="pdf-highlight"
              style={{ ...r, background: HIGHLIGHT_COLOR }}
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
  const [selected, setSelected] = useState(null);       // docId
  const [pdf, setPdf] = useState(null);
  const [pageSizes, setPageSizes] = useState([]);
  const [highlights, setHighlights] = useState({});     // pageNum -> itemIds
  const [status, setStatus] = useState(null);
  const pageEls = useRef(new Map());
  const scrollTo = useRef(null);                        // pageNum pending scroll

  useEffect(() => {
    getDocuments().then((r) => setDocs(r.documents)).catch((e) => setError(e.message));
  }, []);

  // Load the selected PDF and measure every page for stable lazy-scroll.
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setPdf(null);
    setPageSizes([]);
    (async () => {
      try {
        // pdf.js v6 only reads src.url — the positional-string form of
        // getDocument(url) from v4 silently resolves to "no url given".
        const doc = await pdfjsLib.getDocument({ url: documentPdfUrl(selected) }).promise;
        if (cancelled) return;
        const sizes = [];
        for (let n = 1; n <= doc.numPages; n++) {
          const vp = (await doc.getPage(n)).getViewport({ scale: SCALE });
          sizes.push({ width: vp.width, height: vp.height });
        }
        if (cancelled) return;
        setPdf(doc);
        setPageSizes(sizes);
      } catch (e) {
        if (!cancelled) setStatus(`could not load PDF: ${e.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [selected]);

  // Citation deep-link: open the doc, locate the chunk, highlight + scroll.
  useEffect(() => {
    if (!target) return;
    setHighlights({});
    setSelected(target.docId);
    setStatus('locating cited passage…');
  }, [target?.nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!target || !pdf || selected !== target.docId) return;
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
        const candidates = [];
        if (chunk.pages) {
          for (let p = Math.max(1, chunk.pages[0] - 1);
               p <= Math.min(pdf.numPages, chunk.pages[1] + 1); p++) candidates.push(p);
        }
        for (let p = 1; p <= pdf.numPages; p++) {
          if (!candidates.includes(p)) candidates.push(p);
        }

        for (const pageNum of candidates) {
          const tc = await (await pdf.getPage(pageNum)).getTextContent();
          if (cancelled) return;
          const found = matchOnPage(indexPage(tc), bodyWords);
          if (found) {
            setHighlights({ [pageNum]: found.itemIds });
            setStatus(null);
            scrollTo.current = pageNum;
            const el = pageEls.current.get(pageNum);
            if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); scrollTo.current = null; }
            return;
          }
        }
        // Text not locatable (scanned page, heavy equations): land on the page.
        const fallback = chunk.pages?.[0] ?? 1;
        setStatus('passage could not be pinpointed — showing its page');
        scrollTo.current = fallback;
        pageEls.current.get(fallback)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (e) {
        if (!cancelled) setStatus(`citation lookup failed: ${e.message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [target?.nonce, pdf]);   // eslint-disable-line react-hooks/exhaustive-deps

  const registerPage = (pageNum, el) => {
    if (el) pageEls.current.set(pageNum, el);
    else pageEls.current.delete(pageNum);
    if (el && scrollTo.current === pageNum) {
      scrollTo.current = null;
      el.scrollIntoView({ block: 'start' });
    }
  };

  const controls = (
    <div className="doc-list">
      <div className="control-label">Documents</div>
      {error && <div className="doc-list-error">{error}</div>}
      {docs?.map((d) => (
        <button
          key={d.docId}
          className={`doc-item ${selected === d.docId ? 'active' : ''}`}
          onClick={() => { setHighlights({}); setStatus(null); setSelected(d.docId); }}
          title={d.filename}
        >
          <span className="doc-item-title">{d.title}</span>
          {d.authors?.length > 0 && (
            <span className="doc-item-authors">{d.authors.slice(0, 3).join(', ')}</span>
          )}
        </button>
      ))}
      {docs && docs.length === 0 && <div className="doc-list-error">No documents indexed yet.</div>}
    </div>
  );

  return (
    <div className="pdf-wrap">
      {active && controlsEl && createPortal(controls, controlsEl)}
      {status && <div className="pdf-status">{status}</div>}
      {!selected ? (
        <div className="viz-empty">
          <h2>Documents</h2>
          <p>Pick a document from the sidebar, or click a citation in Chat to jump straight to the cited passage.</p>
        </div>
      ) : !pdf ? (
        <div className="viz-empty"><p>loading PDF…</p></div>
      ) : (
        <div className="pdf-scroll">
          {pageSizes.map((size, i) => (
            <PdfPage
              key={`${selected}_${i + 1}`}
              pdf={pdf}
              pageNum={i + 1}
              size={size}
              highlightIds={highlights[i + 1] || null}
              pageRef={registerPage}
            />
          ))}
        </div>
      )}
    </div>
  );
}
