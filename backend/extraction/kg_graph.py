"""
kg_graph.py — knowledge graph over a collection's documents, via kg-gen.

Replaces the old citation/section graph (build_graph.js). kg-gen prompts an
LLM for (subject, predicate, object) triples, so the graph holds concepts and
their relations instead of document structure.

The unit of extraction is the embed stage's docling chunk: chunks are
structure-aware (never cross a section — or document — boundary) and carry a
"title — heading" prefix that grounds entity names, so they beat re-splitting
raw text at arbitrary character offsets. One kg-gen call per chunk, never
merging chunks together; a chunk longer than the context-window guard is
split and sent as separate calls instead. Per-chunk graphs are unioned with
kg-gen's aggregate(). This makes the graph stage depend on embed:
extract → embed → heuristic → graph.

Reads:
  <DATA_DIR>/embeddings.json        — chunk store (embed.js / chunker.js)
  <DATA_DIR>/heuristic_output.json  — top-k doc ranking (heuristic.py); when
                     present, only chunks of the TOP_DOCUMENTS highest-ranked
                     docs are graphed. Absent → every document (fallback).

Writes:
  <DATA_DIR>/graph.json      — {createdAt, model, entities, edges, relations,
                                sourceDocIds, chunksProcessed, chunksFailed}
  <DATA_DIR>/kg_view.html    — standalone interactive visualization (kg-gen)

Env:
  KG_MODEL           Ollama model tag; a bare tag is prefixed with
                     'ollama_chat/' for LiteLLM, which kg-gen routes through.
  OLLAMA_URL         default http://localhost:11434
  TOP_DOCUMENTS      how many ranked docs to graph (default 2)
  KG_MAX_CHUNK_CHARS context-window guard per call (default 8000)
  KG_RETRY_TEMPERATURES  retry ladder for malformed-triple failures

Reference/bibliography chunks are excluded — citation strings would flood the
graph with author/title noise. Each chunk piece is retried over the
temperature ladder; a piece that fails every temperature is skipped (and
counted) rather than failing the multi-hour stage.
"""

import json
import math
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

KG_MODEL   = os.environ.get("KG_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# How many of the heuristic's ranked documents the graph is built from. Capped
# by however many heuristic.py actually emitted (HEURISTIC_K).
TOP_DOCUMENTS = int(os.environ.get("TOP_DOCUMENTS", "2"))
# Context-window guard: a docling chunk is one call, but never a call larger
# than this — an oversized chunk is split into separate calls (never merged
# with its neighbour). ~4 chars/token; Ollama's default num_ctx is 4096
# tokens, and the window must also hold instructions + entity list + output.
MAX_CHUNK_CHARS = int(os.environ.get("KG_MAX_CHUNK_CHARS", "8000"))
# Temperatures tried in order until kg-gen returns a valid graph for a chunk.
_RETRY_TEMPERATURES = tuple(
    float(value) for value in
    os.environ.get("KG_RETRY_TEMPERATURES", "0.0,0.4,0.7").split(",") if value.strip()
)

# Bibliography headings, shared with extract.py / heuristic.py / regex_utils.js.
_REF_HEADINGS = frozenset(
    heading.strip().lower()
    for heading in os.environ.get(
        "PIPELINE_REF_HEADINGS",
        "references,bibliography,works cited,literature cited,citations").split(",")
    if heading.strip()
)


def _norm_heading(heading: str) -> str:
    """Lowercase, drop leading numbering ('7. References'), strip trailing punctuation."""
    normalized = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', heading.lower().strip())
    return normalized.rstrip(' .:')


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


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split one chunk into near-equal pieces under max_chars, cutting on word
    boundaries. Called only for chunks larger than the context guard."""
    piece_count = math.ceil(len(text) / max_chars)
    words = text.split(" ")
    words_per_piece = math.ceil(len(words) / piece_count)
    return [
        " ".join(words[start:start + words_per_piece])
        for start in range(0, len(words), words_per_piece)
    ]


def _chunk_pieces(chunks: list[dict], selected_doc_ids: list[str]) -> tuple[list[str], int]:
    """Per-call texts for the selected docs: one docling chunk per call,
    bibliography chunks dropped, oversized chunks split (never merged).
    Returns (pieces, chunks_used)."""
    selected = set(selected_doc_ids)
    pieces: list[str] = []
    chunks_used = 0
    for chunk in chunks:
        if chunk.get("docId") not in selected:
            continue
        if _norm_heading(chunk.get("heading") or "") in _REF_HEADINGS:
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        chunks_used += 1
        if len(text) > MAX_CHUNK_CHARS:
            pieces.extend(_split_oversized(text, MAX_CHUNK_CHARS))
        else:
            pieces.append(text)
    return pieces, chunks_used


def _generate_with_retry(kg: KGGen, text: str):
    """One chunk piece → Graph, retrying up the temperature ladder. A small
    local model sometimes emits a triple with a null subject/object; kg-gen
    validates triples strictly and raises. Temp 0 is deterministic (and
    dspy-cached), so a retry must raise the temperature to re-sample. Returns
    None when every temperature fails."""
    for temperature in _RETRY_TEMPERATURES:
        try:
            return kg.generate(input_data=text, temperature=temperature)
        except Exception as exc:  # dspy/pydantic ValidationError on malformed triples
            print(f"[kg_graph]   generate failed at temperature={temperature} "
                  f"({type(exc).__name__})", file=sys.stderr)
    return None


def build_kg(data_dir: Path = DATA_DIR) -> dict:
    embeddings_path = data_dir / "embeddings.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} not found — run the embed stage first")
    chunks = json.loads(embeddings_path.read_text(encoding="utf-8")).get("chunks", [])
    if not chunks:
        raise ValueError("embeddings.json has no chunks — run the embed stage first")

    # Doc order: heuristic rank when available, else chunk-store order.
    chunk_doc_ids = list(dict.fromkeys(chunk.get("docId") for chunk in chunks))
    top_k_ids = _top_k_doc_ids(data_dir)
    if top_k_ids:
        selected_ids = [doc_id for doc_id in top_k_ids
                        if doc_id in set(chunk_doc_ids)][:TOP_DOCUMENTS]
        print(f"[kg_graph] top-k ranking found — graphing {len(selected_ids)} "
              f"of {len(chunk_doc_ids)} document(s) (TOP_DOCUMENTS={TOP_DOCUMENTS})")
    else:
        selected_ids = chunk_doc_ids
        print(f"[kg_graph] no top-k ranking — graphing all {len(selected_ids)} document(s)")

    pieces, chunks_used = _chunk_pieces(chunks, selected_ids)
    if not pieces:
        raise ValueError("no chunk text available — nothing to build a graph from")

    model = _litellm_model(KG_MODEL)
    print(f"[kg_graph] {chunks_used} chunk(s) → {len(pieces)} call(s), model={model}")

    kg = KGGen(model=model, api_base=OLLAMA_URL, api_key="ollama", temperature=0.0)
    piece_graphs = []
    failed_pieces = 0
    for piece_idx, piece in enumerate(pieces):
        print(f"[kg_graph]   chunk {piece_idx + 1}/{len(pieces)} ({len(piece)} chars)")
        piece_graph = _generate_with_retry(kg, piece)
        if piece_graph is None:
            failed_pieces += 1
            print(f"[kg_graph]   chunk {piece_idx + 1} failed every temperature — skipped",
                  file=sys.stderr)
        else:
            piece_graphs.append(piece_graph)
    if not piece_graphs:
        raise RuntimeError(f"kg-gen produced no valid graph for any of {len(pieces)} chunk(s)")

    graph = piece_graphs[0] if len(piece_graphs) == 1 else kg.aggregate(piece_graphs)

    # Drop junk the small model leaks past kg-gen's validation: empty/blank
    # entities and triples with any empty element (they'd render as blank nodes).
    # Reassign onto the graph so visualize() below sees the cleaned version too.
    ok = lambda value: isinstance(value, str) and value.strip()
    graph.entities  = {entity for entity in graph.entities if ok(entity)}
    graph.relations = {relation for relation in graph.relations
                       if len(relation) == 3 and all(ok(part) for part in relation)}

    # kg-gen returns sets/tuples; sort into lists so the JSON is deterministic.
    payload = {
        "createdAt":       datetime.now(timezone.utc).isoformat(),
        "model":           model,
        "sourceDocIds":    selected_ids,
        "chunksProcessed": chunks_used,
        "chunksFailed":    failed_pieces,
        "entities":        sorted(graph.entities),
        "edges":           sorted(graph.edges),
        "relations":       sorted([list(relation) for relation in graph.relations]),
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
