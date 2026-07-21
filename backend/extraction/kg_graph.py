"""
kg_graph.py — knowledge graph over a collection's documents, via kg-gen.

Replaces the old citation/section graph (build_graph.js). kg-gen prompts an
LLM for (subject, predicate, object) triples, so the graph now holds concepts
and their relations instead of document structure — nothing upstream is
needed beyond the extracted text.

Reads:
  <DATA_DIR>/doclings.json          — extracted document content (extract.py)
  <DATA_DIR>/heuristic_output.json  — top-k doc ranking (heuristic.py); when
                     present, the graph is built over ONLY those docs. The
                     stage is LLM-bound, so restricting it to the k documents
                     the heuristic deemed important is the main speed lever
                     (e.g. 13 docs → 2). Absent → every document (fallback).

Writes:
  <DATA_DIR>/graph.json      — {createdAt, model, entities, edges, relations,
                                entityClusters, edgeClusters, sourceDocIds}
  <DATA_DIR>/kg_view.html    — standalone interactive visualization (kg-gen)

Env:
  KG_MODEL           Ollama model tag; a bare tag is prefixed with
                     'ollama_chat/' for LiteLLM, which kg-gen routes through.
  OLLAMA_URL         default http://localhost:11434
  KG_CHARS_PER_DOC   per-document text budget (default 3000). Generation is
                     LLM-bound and slow, so the corpus is capped rather than
                     sent whole.
  KG_CHUNK_SIZE      characters per kg-gen chunk (default 2000)

Generation is retried over a small temperature ladder: a local model can emit
a malformed triple that kg-gen rejects, and re-sampling escapes it.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from kg_gen import KGGen
from kg_gen.utils.visualize_kg import visualize

# Force UTF-8 console streams, as extract.py does: the pipeline pipes stdout,
# which on Windows defaults to cp1252, and a non-Latin-1 char in a print here
# would fail the stage after the graph was already built.
for _console_stream in (sys.stdout, sys.stderr):
    _reconfigure_stream = getattr(_console_stream, "reconfigure", None)
    if callable(_reconfigure_stream):
        try:
            _reconfigure_stream(encoding="utf-8")
        except Exception:
            pass

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / os.environ.get("DATA_DIR", "data")
DOCLINGS_PATH = DATA_DIR / "doclings.json"
GRAPH_PATH    = DATA_DIR / "graph.json"
VIEW_PATH     = DATA_DIR / "kg_view.html"

KG_MODEL   = os.environ.get("KG_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHARS_PER_DOC = int(os.environ.get("KG_CHARS_PER_DOC", "3000"))
CHUNK_SIZE    = int(os.environ.get("KG_CHUNK_SIZE", "2000"))
# Temperatures tried in order until kg-gen returns a valid graph (see build_kg).
_RETRY_TEMPERATURES = (0.0, 0.4, 0.7)

_REF_HEADINGS = frozenset({
    "references", "bibliography", "works cited", "literature cited", "citations",
})


def _norm_heading(heading: str) -> str:
    """Lowercase, drop leading numbering ('7. References'), strip trailing punctuation."""
    normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', heading.lower().strip())
    return normalized.rstrip(' .:')


def _document_text(docling_entry: dict) -> str:
    """Body text of one document, bibliographies excluded — reference lists are
    citation strings and would fill the graph with author/title noise."""
    body_sections = [
        section.get("text", "")
        for section in docling_entry.get("sections", [])
        if _norm_heading(section.get("heading", "")) not in _REF_HEADINGS
        and (section.get("text") or "").strip()
    ]
    text = " ".join(body_sections) if body_sections else docling_entry.get("text", "")
    return " ".join(text.split())[:CHARS_PER_DOC]


def _litellm_model(model: str) -> str:
    """kg-gen calls LiteLLM, which needs a provider prefix; bare tags are Ollama."""
    return model if "/" in model else f"ollama_chat/{model}"


def _top_k_doc_ids(data_dir: Path) -> list[str] | None:
    """docIds of the heuristic's top-k, in rank order; None when the ranking
    isn't available (no heuristic run yet, or an unreadable/empty file) so the
    caller falls back to every document."""
    heuristic_path = data_dir / "heuristic_output.json"
    if not heuristic_path.exists():
        return None
    try:
        top_k = json.loads(heuristic_path.read_text(encoding="utf-8")).get("topK", [])
    except (json.JSONDecodeError, ValueError):
        return None
    doc_ids = [entry["docId"] for entry in top_k if entry.get("docId")]
    return doc_ids or None


def build_kg(data_dir: Path = DATA_DIR) -> dict:
    doclings_path = data_dir / "doclings.json"
    if not doclings_path.exists():
        raise FileNotFoundError(f"{doclings_path} not found — run extract.py first")

    doclings: dict = json.loads(doclings_path.read_text(encoding="utf-8"))
    if not doclings:
        raise ValueError("doclings.json is empty — nothing to build a graph from")

    # Restrict to the heuristic's top-k when available (the speed lever); else
    # every document present. Unknown top-k ids (not in doclings) are skipped.
    top_k_ids = _top_k_doc_ids(data_dir)
    if top_k_ids:
        selected_ids = [doc_id for doc_id in top_k_ids if doc_id in doclings]
        print(f"[kg_graph] top-k ranking found — graphing {len(selected_ids)} "
              f"of {len(doclings)} document(s)")
    else:
        selected_ids = list(doclings.keys())
        print(f"[kg_graph] no top-k ranking — graphing all {len(selected_ids)} document(s)")

    doc_ids, texts = [], []
    for doc_id in selected_ids:
        text = _document_text(doclings[doc_id])
        if text:
            doc_ids.append(doc_id)
            texts.append(text)
    if not texts:
        raise ValueError("no document text available — nothing to build a graph from")

    model = _litellm_model(KG_MODEL)
    print(f"[kg_graph] {len(texts)} document(s), model={model}")

    corpus_text = "\n\n".join(texts)
    kg = KGGen(model=model, api_base=OLLAMA_URL, api_key="ollama", temperature=0.0)
    # Only chunk when there's more than one chunk's worth; kg-gen sends short
    # input in a single call.
    chunk_size = CHUNK_SIZE if len(corpus_text) > CHUNK_SIZE else None

    # Retry ladder over temperature. A small local model sometimes emits a
    # relation with a null subject/object; kg-gen validates triples as strict
    # (subject, predicate, object) strings and raises, failing the whole stage.
    # At temp 0 that completion is deterministic (and dspy-cached), so retrying
    # at a higher temperature re-samples AND dodges the cache to escape it.
    graph = None
    last_error = None
    for temperature in _RETRY_TEMPERATURES:
        try:
            graph = kg.generate(input_data=corpus_text, chunk_size=chunk_size,
                                temperature=temperature)
            break
        except Exception as exc:  # dspy/pydantic ValidationError on malformed triples
            last_error = exc
            print(f"[kg_graph] generate failed at temperature={temperature} "
                  f"({type(exc).__name__}); retrying", file=sys.stderr)
    if graph is None:
        raise RuntimeError(
            f"kg-gen could not produce a valid graph after "
            f"{len(_RETRY_TEMPERATURES)} attempt(s): {last_error}")

    # Drop junk the small model leaks past kg-gen's validation: empty/blank
    # entities and triples with any empty element (they'd render as blank nodes).
    # Reassign onto the graph so visualize() below sees the cleaned version too.
    ok = lambda value: isinstance(value, str) and value.strip()
    graph.entities  = {entity for entity in graph.entities if ok(entity)}
    graph.relations = {relation for relation in graph.relations
                       if len(relation) == 3 and all(ok(part) for part in relation)}

    # kg-gen returns sets/tuples; sort into lists so the JSON is deterministic.
    payload = {
        "createdAt":      datetime.now(timezone.utc).isoformat(),
        "model":          model,
        "sourceDocIds":   doc_ids,
        "entities":       sorted(graph.entities),
        "edges":          sorted(graph.edges),
        "relations":      sorted([list(relation) for relation in graph.relations]),
        "entityClusters": {key: sorted(values)
                           for key, values in (graph.entity_clusters or {}).items()},
        "edgeClusters":   {key: sorted(values)
                           for key, values in (graph.edge_clusters or {}).items()},
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "graph.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    visualize(graph, str(data_dir / "kg_view.html"), open_in_browser=False)

    print(f"[kg_graph] {len(payload['entities'])} entities, "
          f"{len(payload['relations'])} relations → {data_dir / 'graph.json'}")
    return payload


if __name__ == "__main__":
    try:
        build_kg()
    except Exception as exc:
        print(f"[kg_graph] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
