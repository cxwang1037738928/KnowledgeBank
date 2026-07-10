"""
extract.py

Converts every PDF tracked in data/documents.json into a structured
DoclingDocument.  Digital PDFs go through docling's standard pipeline;
scanned/mixed PDFs use docling's OCR pipeline backed by Tesseract.

GROBID (CRF models) runs alongside docling on the same PDF: docling owns
text/markdown/sections/tables, GROBID owns metadata + references. The old
docling/LLM heuristics remain as fallbacks when the GROBID server is down.

Output: data/doclings.json  — dict keyed by docId, each entry holds:
  text             : full extracted plain text (docling)
  markdown         : structured markdown (docling)
  sections         : [{heading, text}] extracted from the document hierarchy
  tables           : [str] table text, one entry per table
  references       : [str] raw bibliographic reference strings (GROBID)
  parsedReferences : [{title, authors, raw}] structured refs (GROBID CRF) —
                     lets heuristic.py skip its LLM reference-parsing pass
  metadata         : {title, authors, abstract} (GROBID header model;
                     docling/LLM fallback), plus doi added later by
                     doi_regex.js
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root — two levels up from this file)
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / os.environ.get("DATA_DIR", "data")
DOCUMENTS_META = DATA_DIR / "documents.json"
DOCLINGS_OUT   = DATA_DIR / "doclings.json"
ENHANCED_DIR   = Path(os.environ.get("ENHANCED_DIR", str(DATA_DIR / "enhanced")))

# OCR_PATH: directory containing the Tesseract binary (e.g. C:\Program Files\Tesseract-OCR\).
# When unset, assumes "tesseract" is already on the system PATH (Linux/Docker).
_ocr_dir = os.environ.get("OCR_PATH", "").strip()
if _ocr_dir:
    _exe = "tesseract.exe" if os.name == "nt" else "tesseract"
    TESSERACT_CMD = str(Path(_ocr_dir) / _exe)
else:
    TESSERACT_CMD = "tesseract"

OLLAMA_URL       = os.environ.get("OLLAMA_URL",       "http://localhost:11434")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "ministral:3b")
METADATA_MODEL   = os.environ.get("METADATA_MODEL",   "ministral:3b")
METADATA_WORDS   = int(os.environ.get("METADATA_WORDS", "800"))

# GROBID server (CRF models — run the lightweight image, e.g.
#   docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0
# The full grobid/grobid image swaps in deep-learning models; we
# deliberately use the CRF-only image for speed and low memory).
GROBID_URL     = os.environ.get("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT = int(os.environ.get("GROBID_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# docling imports
# ---------------------------------------------------------------------------

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling_core.types.doc.labels import DocItemLabel

# ---------------------------------------------------------------------------
# Converters (built once, reused across documents)
# ---------------------------------------------------------------------------

def _make_converter(ocr: bool) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = ocr
    if ocr:
        opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=False, tesseract_cmd=TESSERACT_CMD)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

_converter_digital = _make_converter(ocr=False)
_converter_ocr     = _make_converter(ocr=True)


def _choose_converter(doc_meta: dict) -> DocumentConverter:
    """
    Check the enhance_pdf report (data/enhanced/<docId>.json) to see if
    the majority of pages are scanned/mixed.  Fall back to OCR converter
    if the report doesn't exist yet.
    """
    doc_id = doc_meta.get("docId", "")
    report_path = ENHANCED_DIR / f"{doc_id}.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            pages = report.get("pages", [])
            scanned = sum(1 for p in pages if p.get("pageType") != "digital")
            if pages and scanned / len(pages) < 0.3:
                return _converter_digital
        except Exception:
            pass
    return _converter_ocr


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_sections(doc) -> list[dict]:
    """Walk the document body and collect (heading, body-text) pairs."""
    sections = []
    current_heading = None
    current_text_parts = []

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = getattr(item, "text", "").strip()
        if not text:
            continue

        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            if current_heading is not None or current_text_parts:
                sections.append({
                    "heading": current_heading or "",
                    "text": " ".join(current_text_parts),
                })
            current_heading = text
            current_text_parts = []
        else:
            current_text_parts.append(text)

    if current_heading is not None or current_text_parts:
        sections.append({
            "heading": current_heading or "",
            "text": " ".join(current_text_parts),
        })
    return sections


def _extract_tables(doc) -> list[str]:
    tables = []
    for tbl in doc.tables:
        try:
            tables.append(tbl.export_to_markdown())
        except Exception:
            tables.append(str(tbl))
    return tables


# ---------------------------------------------------------------------------
# References — shared constants
# ---------------------------------------------------------------------------
# Single canonical definition (lowercase, for membership checks).  NOTE: the
# old file defined this twice — once as a lowercase frozenset and once as a
# capitalized list.  The list silently overwrote the frozenset, which broke
# every `text.lower() in _REF_SECTION_HEADINGS` membership check.
_REF_SECTION_HEADINGS = frozenset({
    "references", "bibliography", "works cited", "literature cited", "citations",
})

_CODE_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)

# Caption lines that leak into the refs section from PDF text extraction.
_REF_CAPTION_RE = re.compile(r'(?im)^(?:Figure|Table|Scheme|Chart|Fig\.?|Eq\.?)\s*\d+\s*[:.]')
# Spurious glued double-numbering left after marker stripping, e.g. "18.Sierra".
_REF_LEADING_NUM_RE = re.compile(r'^\d{1,3}\.(?=\S)')
# Leading numbered/bracketed marker on an individual entry, e.g. "[3] ", "(3) ", "3. ".
_REF_MARKER_PREFIX_RE = re.compile(r'^(?:[-\u2013\u2014\u2022*]\s+)?(?:\[\d+\]|\(\d+\)|\d+[a-zA-Z]?\.)\s+')

# Docling labels that are never bibliography content — dropped during the
# section-walk fallback (running headers/footers, captions, footnotes, ...).
_REF_DROP_LABELS = frozenset({
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.TABLE,
})


def _norm_heading(text: str) -> str:
    """Normalize a heading for refs-section matching: lowercase, strip
    leading numbering ('7. References', 'VII. References') and trailing
    punctuation."""
    t = text.lower().strip()
    t = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', t)   # '7. ' / 'vii. ' / '7) '
    return t.rstrip(' .:')


def _is_ref_heading(text: str) -> bool:
    return _norm_heading(text) in _REF_SECTION_HEADINGS


def _finalize_entries(entries: list[str]) -> list[str]:
    """Post-process split entries: strip spurious glued leading numbers and
    drop caption-only / empty entries. Applied to every extraction tier."""
    out = []
    for e in entries:
        e = _REF_LEADING_NUM_RE.sub('', e.strip()).strip()
        if e and not _REF_CAPTION_RE.match(e):
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# LLM extraction via Ollama
# ---------------------------------------------------------------------------


def _parse_json(raw: str):
    """Strip markdown fences and parse JSON; return None on failure."""
    try:
        return json.loads(_CODE_FENCE.sub("", raw).strip())
    except Exception:
        return None


_METADATA_SKIP = frozenset({
    "abstract", "introduction", "background", "methods", "results",
    "discussion", "conclusion", "conclusions", "references", "bibliography",
    "acknowledgements", "acknowledgments", "appendix", "supplementary",
    "related work", "future work", "limitations", "keywords", "overview",
    "summary", "notation", "funding", "license", "orcid",
})

# Non-title heading patterns: copyright lines, publisher banners, etc.
_NOT_TITLE_RE = re.compile(
    r'(?i)^('
    r'provided proper attribution|permission to reproduce|all rights reserved|'
    r'published by|journal of|proceedings of|transactions on|'
    r'vol\.|volume\s+\d|issue\s+\d|doi:|arxiv:'
    r')',
)


def _llm_extract_metadata(sections: list[dict]) -> dict | None:
    """Extract title, authors, and abstract from the first few sections.

    Title and abstract are extracted structurally (deterministic).  Small LLMs
    are unreliable at answering "is this first heading the title?", so we don't
    ask them — we just take the first non-skip, non-publisher heading.  Authors
    are harder to separate from affiliations, so the LLM handles those alone.
    """
    if not sections:
        return None

    # Title: first non-empty heading that is not a known section keyword and
    # does not look like a copyright/publisher notice.
    title: str | None = None
    for s in sections[:7]:
        heading = (s.get("heading") or "").strip()
        if (heading
                and heading.lower().strip() not in _METADATA_SKIP
                and not _NOT_TITLE_RE.match(heading)):
            title = heading
            break

    # Abstract: first section whose heading starts with "abstract".
    abstract: str | None = None
    for s in sections:
        if (s.get("heading") or "").lower().strip().startswith("abstract"):
            abstract = (s.get("text") or "").strip() or None
            break

    # Authors: LLM on the text of the first 1-2 non-empty sections.  Author
    # names typically appear just below the title heading.  Focused prompt so
    # the model isn't distracted by also finding title / abstract.
    authors: list[str] = []
    candidate_parts: list[str] = []
    for s in sections[:5]:
        text = (s.get("text") or "").strip()
        if text:
            candidate_parts.append(" ".join(text.split()[:300]))
            if len(candidate_parts) >= 2:
                break

    if candidate_parts:
        combined = "\n".join(candidate_parts)
        prompt = (
            "Extract the author names from this academic paper header text.\n"
            'Return ONLY a JSON array of strings, each a full author name (e.g. ["John Smith", "Jane Doe"]).\n'
            "Ignore affiliations, email addresses, degree titles (PhD, Member IEEE, etc.), and institution names.\n"
            "Return [] if no clear author names are present. No explanation, no extra text.\n\n"
            f"Text:\n{combined}"
        )
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": METADATA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0}},
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            parsed = _parse_json(raw)
            if isinstance(parsed, list):
                authors = [a for a in parsed if isinstance(a, str) and a.strip()]
        except Exception:
            pass

    if title is None and not authors and abstract is None:
        return None
    return {"title": title, "authors": authors, "abstract": abstract}


# ---------------------------------------------------------------------------
# References — regex fallback (Tier 3, text-dump splitting)
# ---------------------------------------------------------------------------

def _ref_section_from_text(full_text: str) -> str | None:
    """Extract the references section from the full document text, preserving
    per-reference line breaks.
    _extract_sections joins items with ' ' (spaces), destroying the line breaks
    between individual references.  This function works directly on
    export_to_text() output so each numbered entry stays on its own line.
    Returns everything after the last known refs heading to end-of-document.
    """
    heading_pat = '|'.join(
        re.escape(h) for h in sorted(_REF_SECTION_HEADINGS, key=len, reverse=True)
    )
    matches = list(re.finditer(rf'(?im)^(?:{heading_pat})\s*\n', full_text))
    if not matches:
        return None
    return full_text[matches[-1].end():].strip()


def _split_references_regex(text: str) -> list[str]:
    """Split a references-section blob into individual entries.
    Preferred input is text from _ref_section_from_text() which preserves
    newlines; _extract_sections() space-joins items so step 2 exists as a
    fallback for that case.
    Tries in order:
    1. Numbered/bracketed markers at LINE START — well-formatted plain text
       ([1] IEEE/ACS, (1) ACS-alt, 1. APA/Vancouver/CSE, 1A. AIP,
        optionally preceded by a bullet/dash e.g. "- [1]")
    2. Inline N. markers — handles space-joined text where line breaks are lost
       (lookbehind prevents matching abbreviations like J., No., and URLs doi:10.)
    3. Inline bracket markers [N] — IEEE style after reflow
    4. Blank-line paragraph separation — APA/MLA/Chicago author-date
    5. Author-date line starts (Lastname, F.)
    6. One reference per physical line

    All strategies run through _finalize_entries(), which drops figure/table
    captions and strips spurious glued leading numbers (e.g. "18.Sierra").
    """
    text = text.strip()
    if not text:
        return []

    def _spans_to_entries(spans):
        entries = []
        for i, (_, me) in enumerate(spans):
            end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
            body = text[me:end].strip()
            if not body:
                continue
            # split off embedded caption lines so they don't pollute a reference
            cur = []
            for ln in body.split('\n'):
                s = ln.strip()
                if _REF_CAPTION_RE.match(s):      # caption -> flush current, drop caption
                    if cur:
                        entries.append(' '.join(cur).strip())
                        cur = []
                    continue
                cur.append(s)
            if cur:
                entries.append(' '.join(cur).strip())
        return entries

    # 1. Line-start markers: [1], (1), 1., 1A. — optional leading bullet/dash "- [1]"
    spans = [(m.start(), m.end()) for m in
             re.finditer(r'(?m)^[ \t]*(?:[-\u2013\u2014\u2022*]\s+)?'
                         r'(?:\[\d+\]|\(\d+\)|\d+[a-zA-Z]?\.)\s+', text)]
    if len(spans) >= 2:
        entries = _spans_to_entries(spans)
        if entries:
            return _finalize_entries(entries)

    # 2. Inline N. — space-joined paragraph (e.g. from _extract_sections)
    #    (?<![a-zA-Z:]) avoids J., No., Fig., doi:10.
    #    (?=[A-Z\("[]) avoids matching volume/page numbers before lowercase
    spans = [(m.start(), m.end()) for m in
             re.finditer(r'(?<![a-zA-Z:])\d{1,3}\.\s+(?=[A-Z\("[])', text)]
    if len(spans) >= 2:
        entries = _spans_to_entries(spans)
        if entries:
            return _finalize_entries(entries)

    # 3. Inline bracket markers [N]
    spans = [(m.start(), m.end()) for m in re.finditer(r'\[\d+\]\s+', text)]
    if len(spans) >= 2:
        entries = _spans_to_entries(spans)
        if entries:
            return _finalize_entries(entries)

    # 4. Blank-line paragraph separation
    paras = [p.strip() for p in re.split(r'\n[ \t]*\n', text) if len(p.strip()) > 15]
    if len(paras) >= 2:
        return _finalize_entries(paras)

    # 5. Author-date: each entry starts on a new line with Lastname, First
    starts = [m.start() for m in re.finditer(r'(?m)^[A-Z][a-z\-]{1,25},\s+[A-Z]', text)]
    if len(starts) >= 2:
        entries = []
        for i, pos in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            body = text[pos:end].strip()
            if len(body) > 15:
                entries.append(body)
        if entries:
            return _finalize_entries(entries)

    # 6. One reference per line
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    return _finalize_entries(lines) if len(lines) >= 2 else ([text] if text else [])


# ---------------------------------------------------------------------------
# References — tiered extraction (structure first, regex last)
# ---------------------------------------------------------------------------

def _clean_structured_entry(text: str) -> str:
    """Clean a single structure-derived entry: collapse internal line breaks
    (wrapped lines within one item) and strip a leading '[3] ' / '3. ' marker."""
    t = ' '.join(ln.strip() for ln in text.splitlines() if ln.strip())
    return _REF_MARKER_PREFIX_RE.sub('', t).strip()


def _refs_from_labels(doc) -> list[str]:
    """Tier 1: docling's layout model labeled individual bibliography entries
    as DocItemLabel.REFERENCE.  Cleanest path — no splitting heuristics."""
    refs = []
    for t in getattr(doc, "texts", []):
        if getattr(t, "label", None) == DocItemLabel.REFERENCE:
            txt = (getattr(t, "text", "") or "").strip()
            if txt:
                refs.append(_clean_structured_entry(txt))
    return _finalize_entries(refs)


def _refs_from_section_walk(doc) -> list[str]:
    """Tier 2: walk body reading order; collect body-layer items between the
    References heading and the next section heading, dropping items whose
    labels mark them as non-bibliographic (captions, headers, footers, ...)."""
    collected = []
    in_refs = False
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            if in_refs:
                break                       # next section started — refs done
            in_refs = _is_ref_heading(text)
            continue
        if not in_refs or not text or label in _REF_DROP_LABELS:
            continue
        collected.append(text)

    if not collected:
        return []
    if len(collected) == 1:
        # Single blob — docling merged the entries; delegate to the splitter.
        return _split_references_regex(collected[0])
    # If each collected item still bundles several entries (multi-line),
    # split those; otherwise treat one item = one reference.
    entries = []
    for c in collected:
        lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
        if len(lines) > 1 and sum(bool(_REF_MARKER_PREFIX_RE.match(ln)) for ln in lines) >= 2:
            entries.extend(_split_references_regex(c))
        else:
            entries.append(_clean_structured_entry(c))
    return _finalize_entries(entries)


def _refs_from_text_dump(doc, sections: list[dict] | None = None) -> list[str]:
    """Tier 3: regex over the flat text export (last resort)."""
    full_text = doc.export_to_text()
    ref_section_text = _ref_section_from_text(full_text)
    if not ref_section_text and sections:
        ref_section_text = next(
            (s["text"] for s in reversed(sections)
             if _is_ref_heading(s.get("heading", ""))),
            None,
        )
    return _split_references_regex(ref_section_text) if ref_section_text else []


def _extract_references(doc, sections: list[dict] | None = None) -> list[str]:
    """Extract bibliography entries, preferring docling's structure over
    text-dump regex:
      Tier 1 — REFERENCE-labelled items (layout model segmentation)
      Tier 2 — reading-order walk of the References section, label-filtered
      Tier 3 — regex splitting of the flat text export
    A tier's result is accepted only if it yields >= 2 entries, since a
    single 'entry' usually means the tier saw one merged blob."""
    refs = _refs_from_labels(doc)
    if len(refs) >= 2:
        return refs
    refs = _refs_from_section_walk(doc)
    if len(refs) >= 2:
        return refs
    return _refs_from_text_dump(doc, sections)


# ---------------------------------------------------------------------------
# Metadata fallback from docling labels (used when Ollama is unavailable)
# ---------------------------------------------------------------------------

def _extract_metadata(doc) -> dict:
    """Extract title/authors/abstract from docling item labels with header fallbacks."""
    meta = {"title": None, "authors": [], "abstract": None}
    section_count = 0
    _SKIP = frozenset({
        "abstract", "introduction", "background", "methods", "results",
        "discussion", "conclusion", "conclusions", "references", "bibliography",
        "acknowledgements", "acknowledgments", "appendix", "supplementary",
        "related work", "future work", "limitations", "orcid", "license",
        "terms", "funding", "keywords", "overview", "notation", "summary",
    })
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label == DocItemLabel.TITLE and meta["title"] is None:
            meta["title"] = text
        if label == DocItemLabel.SECTION_HEADER:
            section_count += 1
            lower = text.lower().strip()
            if meta["title"] is None and lower not in _SKIP and len(text.split()) <= 15:
                meta["title"] = text
            elif (section_count <= 6 and not meta["authors"]
                  and 1 <= len(text.split()) <= 5
                  and lower not in _SKIP
                  and all(re.match(r'^[A-Z][a-zA-Z\-\.]*$', w) for w in text.split())):
                meta["authors"].append(text)
            if lower.startswith("abstract"):
                meta["abstract"] = ""
        elif meta["abstract"] == "" and label not in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            meta["abstract"] = text
    return meta


# ---------------------------------------------------------------------------
# GROBID — title / authors / abstract / references (CRF models)
# ---------------------------------------------------------------------------
# GROBID consumes the raw PDF over HTTP and returns TEI XML. It replaces the
# regex + LLM metadata/reference parsing as the PRIMARY path; the docling/LLM
# heuristics below survive only as fallbacks for when the server is down.

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _tei_text(el) -> str:
    """Flattened text content of a TEI element (None-safe)."""
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _tei_persname(pers) -> str:
    """'First Middle Last' from a TEI <persName> element."""
    forenames = [_tei_text(f) for f in pers.findall("tei:forename", _TEI_NS)]
    surname   = _tei_text(pers.find("tei:surname", _TEI_NS))
    parts = [p for p in forenames + [surname] if p]
    return " ".join(parts)


def _grobid_alive() -> bool:
    try:
        return requests.get(f"{GROBID_URL}/api/isalive", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def _grobid_header(pdf_path: str) -> dict | None:
    """Title/authors/abstract from GROBID's processHeaderDocument (CRF header
    model). Returns None on any transport or parse failure."""
    try:
        with open(pdf_path, "rb") as fh:
            resp = requests.post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={"input": (os.path.basename(pdf_path), fh, "application/pdf")},
                data={"consolidateHeader": "0"},
                # 0.8.x defaults this endpoint to BibTeX — demand TEI XML
                headers={"Accept": "application/xml"},
                timeout=GROBID_TIMEOUT,
            )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None

    title = _tei_text(root.find(".//tei:titleStmt/tei:title", _TEI_NS)) or None

    authors = []
    for author in root.findall(
            ".//tei:sourceDesc//tei:biblStruct//tei:author/tei:persName", _TEI_NS):
        name = _tei_persname(author)
        if name and name not in authors:
            authors.append(name)

    abstract = _tei_text(root.find(".//tei:profileDesc/tei:abstract", _TEI_NS)) or None

    if title is None and not authors and abstract is None:
        return None
    return {"title": title, "authors": authors, "abstract": abstract}


def _grobid_references(pdf_path: str) -> tuple[list[str], list[dict]] | None:
    """Bibliography via GROBID's processReferences (CRF citation model).
    Returns (raw_strings, parsed) where parsed is [{title, authors, raw}] —
    already structured, so heuristic.py needs no LLM pass. None on failure."""
    try:
        with open(pdf_path, "rb") as fh:
            resp = requests.post(
                f"{GROBID_URL}/api/processReferences",
                files={"input": (os.path.basename(pdf_path), fh, "application/pdf")},
                data={"consolidateCitations": "0", "includeRawCitations": "1"},
                headers={"Accept": "application/xml"},
                timeout=GROBID_TIMEOUT,
            )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None

    raw_refs: list[str] = []
    parsed:   list[dict] = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", _TEI_NS):
        # Article title lives in <analytic>; for books/theses only <monogr>
        # exists, so fall back to it.
        title_el = (bibl.find("tei:analytic/tei:title", _TEI_NS)
                    if bibl.find("tei:analytic", _TEI_NS) is not None else None)
        if title_el is None or not _tei_text(title_el):
            title_el = bibl.find("tei:monogr/tei:title", _TEI_NS)
        title = _tei_text(title_el)

        authors = []
        for pers in bibl.findall(".//tei:author/tei:persName", _TEI_NS):
            name = _tei_persname(pers)
            if name and name not in authors:
                authors.append(name)

        raw = _tei_text(bibl.find("tei:note[@type='raw_reference']", _TEI_NS))
        if not raw:
            raw = " ".join(p for p in [", ".join(authors), title] if p)

        if raw:
            raw_refs.append(raw)
        if title or authors:
            parsed.append({"title": title, "authors": authors, "raw": raw})

    return raw_refs, parsed


def _grobid_extract(pdf_path: str) -> dict | None:
    """Header + references in one call. None when the server is unreachable,
    so convert_document can fall back to the docling/LLM path wholesale."""
    if not _grobid_alive():
        return None
    header = _grobid_header(pdf_path)
    refs   = _grobid_references(pdf_path)
    if header is None and refs is None:
        return None
    raw_refs, parsed = refs if refs is not None else ([], [])
    return {
        "metadata":         header,
        "references":       raw_refs,
        "parsedReferences": parsed,
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_document(doc_meta: dict) -> dict:
    file_path = doc_meta.get("filePath", "")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # GROBID and docling consume the same PDF in parallel: GROBID runs on its
    # server over HTTP while docling churns locally in this process.
    converter = _choose_converter(doc_meta)
    with ThreadPoolExecutor(max_workers=1) as pool:
        grobid_future = pool.submit(_grobid_extract, file_path)
        result = converter.convert(file_path)
        grobid = grobid_future.result()

    doc = result.document

    full_text = doc.export_to_text()
    sections  = _extract_sections(doc)

    # Metadata + references: GROBID (CRF) is the primary parser; the
    # docling/LLM heuristics only fill in when the server is down or a
    # field/section comes back empty.
    if grobid is None:
        print("[extract]   GROBID unreachable — falling back to docling/LLM parsing",
              file=sys.stderr)

    metadata = (grobid or {}).get("metadata")
    if metadata is None:
        metadata = _llm_extract_metadata(sections) or _extract_metadata(doc)

    refs        = (grobid or {}).get("references") or []
    parsed_refs = (grobid or {}).get("parsedReferences") or []
    if not refs:
        refs = _extract_references(doc, sections)

    return {
        "docId":            doc_meta["docId"],
        "filename":         doc_meta["filename"],
        "filePath":         file_path,
        "extractedAt":      datetime.now(timezone.utc).isoformat(),
        "text":             full_text,
        "markdown":         doc.export_to_markdown(),
        "sections":         sections,
        "tables":           _extract_tables(doc),
        "references":       refs,
        "parsedReferences": parsed_refs,
        "metadata":         metadata,
    }


def run(force: bool = False) -> None:
    if not DOCUMENTS_META.exists():
        print(f"[extract] {DOCUMENTS_META} not found — run the ingest API first.", file=sys.stderr)
        sys.exit(1)

    docs_meta = json.loads(DOCUMENTS_META.read_text()).get("documents", {})
    if not docs_meta:
        print("[extract] No documents in documents.json. Nothing to do.")
        return

    existing: dict = {}
    if DOCLINGS_OUT.exists():
        try:
            existing = json.loads(DOCLINGS_OUT.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, ValueError):
            existing = {}

    results = dict(existing)
    errors  = []

    for doc_id, meta in docs_meta.items():
        if not force and doc_id in existing:
            print(f"[extract] Skipping {meta['filename']} (already extracted)")
            continue

        if meta.get("status") not in ("completed", "pending", None):
            print(f"[extract] Skipping {meta['filename']} (status={meta.get('status')})")
            continue

        print(f"[extract] Processing {meta['filename']} ...")
        try:
            results[doc_id] = convert_document(meta)
            print(f"[extract]   → {len(results[doc_id]['sections'])} sections, "
                  f"{len(results[doc_id]['references'])} references")
        except Exception as exc:
            print(f"[extract]   ERROR: {exc}", file=sys.stderr)
            errors.append({"docId": doc_id, "filename": meta.get("filename"), "error": str(exc)})

    DOCLINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCLINGS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[extract] Wrote {len(results)} documents to {DOCLINGS_OUT}")

    if errors:
        print(f"[extract] {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e['filename']}: {e['error']}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from PDFs via docling")
    parser.add_argument("--force", action="store_true", help="Re-extract already-processed documents")
    args = parser.parse_args()
    run(force=args.force)