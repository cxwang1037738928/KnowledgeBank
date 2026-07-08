"""
extract.py

Converts every PDF tracked in data/documents.json into a structured
DoclingDocument.  Digital PDFs go through docling's standard pipeline;
scanned/mixed PDFs use docling's OCR pipeline backed by Tesseract.

Output: data/doclings.json  — dict keyed by docId, each entry holds:
  text       : full extracted plain text
  markdown   : structured markdown (headings, tables, lists preserved)
  sections   : [{heading, text}] extracted from the document hierarchy
  tables     : [str] table text, one entry per table
  references : [str] bibliographic reference strings (for connectivity graph)
  metadata   : {title, authors, abstract, ...} from docling's document model
"""

import json
import os
import re
import sys
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


_REF_SECTION_HEADINGS = frozenset({"references", "bibliography", "works cited", "literature cited", "citations"})
_CODE_FENCE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)

# ---------------------------------------------------------------------------
# LLM extraction via Ollama
# ---------------------------------------------------------------------------

def _llm_call(prompt: str) -> str | None:
    """POST a prompt to Ollama; return the response text or None on any error."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": EXTRACTION_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None


def _parse_json(raw: str):
    """Strip markdown fences and parse JSON; return None on failure."""
    try:
        return json.loads(_CODE_FENCE.sub("", raw).strip())
    except Exception:
        return None


def _llm_extract_metadata(full_text: str) -> dict | None:
    """Ask the LLM to extract title, authors, abstract from the paper's opening text."""
    snippet = " ".join(full_text.split()[:800])
    prompt = (
        "Extract the title, author names, and abstract from this academic paper excerpt.\n"
        'Return ONLY a JSON object with keys "title" (string or null), '
        '"authors" (array of name strings), "abstract" (string or null).\n'
        "No explanation, no extra text.\n\n"
        f"Paper excerpt:\n{snippet}"
    )
    raw = _llm_call(prompt)
    if not raw:
        return None
    result = _parse_json(raw)
    if not isinstance(result, dict):
        return None
    return {
        "title":    result.get("title") or None,
        "authors":  result.get("authors") if isinstance(result.get("authors"), list) else [],
        "abstract": result.get("abstract") or None,
    }


def _llm_extract_references(ref_section_text: str) -> list[str] | None:
    """Ask the LLM to split a references section blob into individual entries."""
    snippet = " ".join(ref_section_text.split()[:2000])
    prompt = (
        "Split the following references section into individual reference entries.\n"
        "Return ONLY a JSON array of strings, one string per reference.\n"
        "No explanation, no extra text.\n\n"
        f"References section:\n{snippet}"
    )
    raw = _llm_call(prompt)
    if not raw:
        return None
    result = _parse_json(raw)
    return result if isinstance(result, list) else None


# ---------------------------------------------------------------------------
# Docling-label fallbacks (used when Ollama is unavailable)
# ---------------------------------------------------------------------------

def _extract_references(doc) -> list[str]:
    """Collect ref strings from REFERENCE-labelled items; if none, collect text
    items inside the References/Bibliography section."""
    raw = []
    in_ref_section = False
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label == DocItemLabel.REFERENCE:
            raw.append(text)
            continue
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            in_ref_section = text.lower().strip() in _REF_SECTION_HEADINGS
            continue
        if in_ref_section:
            raw.append(text)
    # Split any blob entries by numbered markers or newlines
    result = []
    for entry in raw:
        lines = [ln.strip() for ln in entry.splitlines() if ln.strip()]
        result.extend(lines if len(lines) > 1 else [entry])
    return result


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
# Main conversion
# ---------------------------------------------------------------------------

def convert_document(doc_meta: dict) -> dict:
    file_path = doc_meta.get("filePath", "")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    converter = _choose_converter(doc_meta)
    result = converter.convert(file_path)
    doc = result.document

    full_text = doc.export_to_text()
    sections  = _extract_sections(doc)

    # Find the references section text to pass to the LLM
    ref_section_text = next(
        (s["text"] for s in reversed(sections)
         if s.get("heading", "").lower().strip() in _REF_SECTION_HEADINGS),
        None,
    )

    # LLM extraction — falls back to docling-label heuristics if Ollama is down
    metadata = _llm_extract_metadata(full_text) or _extract_metadata(doc)
    refs = (
        (_llm_extract_references(ref_section_text) if ref_section_text else None)
        or _extract_references(doc)
    )

    return {
        "docId":       doc_meta["docId"],
        "filename":    doc_meta["filename"],
        "filePath":    file_path,
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "text":        full_text,
        "markdown":    doc.export_to_markdown(),
        "sections":    sections,
        "tables":      _extract_tables(doc),
        "references":  refs,
        "metadata":    metadata,
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
