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
import sys
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root — two levels up from this file)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_META = ROOT / "data" / "documents.json"
DOCLINGS_OUT   = ROOT / "data" / "doclings.json"
ENHANCED_DIR   = ROOT / "data" / "enhanced"

# ---------------------------------------------------------------------------
# docling imports
# ---------------------------------------------------------------------------

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling_core.types.doc import DocItemLabel

# ---------------------------------------------------------------------------
# Converters (built once, reused across documents)
# ---------------------------------------------------------------------------

def _make_converter(ocr: bool) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = ocr
    if ocr:
        opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=False)
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


def _extract_references(doc) -> list[str]:
    refs = []
    for item, _ in doc.iterate_items():
        if getattr(item, "label", None) == DocItemLabel.REFERENCE:
            text = getattr(item, "text", "").strip()
            if text:
                refs.append(text)
    return refs


def _extract_metadata(doc) -> dict:
    meta = {"title": None, "authors": [], "abstract": None, "keywords": []}
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if meta["title"] is None and label == DocItemLabel.TITLE:
            meta["title"] = text
        elif label == DocItemLabel.SECTION_HEADER and text.lower().startswith("abstract"):
            meta["abstract"] = ""  # next text items belong to the abstract
        elif meta["abstract"] == "":
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

    return {
        "docId":       doc_meta["docId"],
        "filename":    doc_meta["filename"],
        "filePath":    file_path,
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "text":        doc.export_to_text(),
        "markdown":    doc.export_to_markdown(),
        "sections":    _extract_sections(doc),
        "tables":      _extract_tables(doc),
        "references":  _extract_references(doc),
        "metadata":    _extract_metadata(doc),
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
        existing = json.loads(DOCLINGS_OUT.read_text())

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
    DOCLINGS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
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
