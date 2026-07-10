"""
heuristic.py — select the k most important documents

Orchestration only: every scoring/graph primitive lives in
heuristic_utils.py; this file owns the PARAMETERS and the pipeline glue
(read doclings/categories → score → rank → write heuristic_output.json).

Scoring model:
  final_score = ALPHA * bm25_component + (1 - ALPHA) * pagerank_score

  bm25_component — blend of two percentile-normalized sub-signals:
    - representativeness [CHANGED, issue 6]: now the mean of the TOP_M_CHUNKS
      best chunk-level BM25 scores against the cluster keywords, instead of
      one whole-document score. Whole-doc scoring rewarded keyword COVERAGE,
      which grows with document length — a 100-page doc beat a focused
      10-pager on volume alone. Top-m chunk scoring ranks documents on the
      density of their best material: long docs get no credit for filler,
      and (unlike a plain per-chunk average) broad documents aren't punished
      for off-topic appendices. See topm_chunk_representativeness().
      Cluster keywords are also capped per member (PER_DOC_KEYWORD_CAP) so
      the longest member's vocabulary no longer writes the keyword set the
      whole cluster is graded against.
    - novelty: average IDF of a document's unique vocabulary vs. the full
      corpus — counterweight to representativeness, which rewards typicality.

  pagerank_score — PageRank over the citation graph built by LLM-parsing
    each document's reference strings and matching title+author against the
    corpus. Rank flows through incoming edges only.

Writes data/heuristic_output.json:
  topK  : [{docId, filename, finalScore, bm25Score, bm25Representativeness,
            bm25Novelty, pagerankScore}]
  edges : [{source, target}]  citation edges across the full corpus

Dependencies: networkx, requests (pip install networkx requests).
"""

import json
import os
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from heuristic_utils import (
    BM25,
    build_connectivity,
    cluster_keywords,
    compute_pagerank,
    novelty_score,
    parse_references_ollama,
    percentile_normalize,
    tokenise,
    top_terms,
    topm_chunk_representativeness,
)

# ---------------------------------------------------------------------------
# Parameters — the single place tunables live
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT / os.environ.get("DATA_DIR", "data")
DOCLINGS_PATH   = DATA_DIR / "doclings.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"
OUTPUT_PATH     = DATA_DIR / "heuristic_output.json"

OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CITATION_MODEL = os.environ.get("CITATION_MODEL", "phi4")

K              = 2      # top-k documents to select
ALPHA          = 0.25   # weight of the BM25 component vs. PageRank
NOVELTY_WEIGHT = 0.2    # how much a document's rare vocabulary boosts its final ranking score

BM25_K1        = 1.5    # term-frequency saturation
BM25_B         = 0.75   # length normalization strength

CHUNK_WORDS         = 180   # window size (tokens) for chunk-level scoring —
                            # matches the embedding pipeline's CHUNK_SIZE
TOP_M_CHUNKS        = 5     # how many best chunks define representativeness
PER_DOC_KEYWORD_CAP = 50    # max terms each cluster member contributes to the
                            # keyword pool (None = uncapped, old behavior)
KEYWORDS_N          = int(os.environ.get("KEYWORDS_N", "20"))

_REF_HEADINGS = frozenset({"references", "bibliography", "works cited", "literature cited", "citations"})

MIN_KEY_LENGTH   = 4    # skip title/author match keys shorter than this, short author last names causes over matching
REF_BATCH_SIZE   = 10   # references per LLM parsing call — small on purpose: a
                        # 3b model given 50 refs at once overflows both the
                        # context window and the output budget, truncating the
                        # JSON and silently zeroing the batch
PAGERANK_DAMPING = 0.85


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(k: int = K) -> None:
    if not DOCLINGS_PATH.exists():
        print(f"[heuristic] {DOCLINGS_PATH} not found — run extract.py first.", file=sys.stderr)
        sys.exit(1)

    doclings: dict = json.loads(DOCLINGS_PATH.read_text(encoding='utf-8'))
    if not doclings:
        print("[heuristic] doclings.json is empty — nothing to score.")
        return

    doc_ids   = list(doclings.keys())
    filenames = {d: doclings[d]["filename"] for d in doc_ids}

    # Tokenize body text only — exclude reference sections so author names and
    # cited-paper titles don't inflate keyword scores.
    def _body_text(entry: dict) -> str:
        sections = entry.get("sections", [])
        body = [s["text"] for s in sections
                if s.get("heading", "").lower().strip() not in _REF_HEADINGS
                and (s.get("text") or "").strip()]
        return " ".join(body) if body else entry.get("text", "")

    tokenised = {d: tokenise(_body_text(doclings[d])) for d in doc_ids}

    bm25 = BM25([tokenised[d] for d in doc_ids], k1=BM25_K1, b=BM25_B)

    # --- Cluster keywords (per-member contribution capped) -------------------
    fallback_kws = top_terms(bm25, n=KEYWORDS_N)
    doc_keywords: dict[str, list[str]] = {}
    if CATEGORIES_PATH.exists():
        categories = json.loads(CATEGORIES_PATH.read_text(encoding='utf-8'))
        for cat in categories.get("categories", []):
            member_ids = [m["docId"] for m in cat.get("members", []) if m["docId"] in tokenised]
            if not member_ids:
                continue
            kws = cluster_keywords(
                [tokenised[d] for d in member_ids], bm25,
                n=KEYWORDS_N, per_doc_cap=PER_DOC_KEYWORD_CAP,
            )
            for d in member_ids:
                doc_keywords[d] = kws
    for d in doc_ids:
        doc_keywords.setdefault(d, fallback_kws)

    # --- BM25 component: top-m chunk representativeness + novelty ------------
    raw_repr = {
        d: topm_chunk_representativeness(
            bm25, tokenised[d], doc_keywords[d],
            chunk_words=CHUNK_WORDS, top_m=TOP_M_CHUNKS,
        )
        for d in doc_ids
    }
    raw_novelty = {d: novelty_score(bm25, tokenised[d]) for d in doc_ids}

    norm_repr    = percentile_normalize(raw_repr)
    norm_novelty = percentile_normalize(raw_novelty)
    bm25_component = {
        d: (1 - NOVELTY_WEIGHT) * norm_repr[d] + NOVELTY_WEIGHT * norm_novelty[d]
        for d in doc_ids
    }

    # --- Citation graph + PageRank -------------------------------------------
    print("[heuristic] Building citation connectivity graph ...")
    parse_fn = partial(
        parse_references_ollama,
        ollama_url=OLLAMA_URL, model=CITATION_MODEL, batch_size=REF_BATCH_SIZE,
    )
    adjacency = build_connectivity(doclings, parse_fn, min_key_length=MIN_KEY_LENGTH)

    edges = [{"source": src, "target": tgt}
             for src, targets in adjacency.items() for tgt in targets]

    norm_pagerank = percentile_normalize(
        compute_pagerank(adjacency, doc_ids, damping=PAGERANK_DAMPING)
    )

    # --- Combine, rank, write -------------------------------------------------
    scored = [
        {
            "docId":                  d,
            "filename":               filenames[d],
            "finalScore":             round(ALPHA * bm25_component[d] + (1 - ALPHA) * norm_pagerank[d], 4),
            "bm25Score":              round(bm25_component[d], 4),
            "bm25Representativeness": round(norm_repr[d], 4),
            "bm25Novelty":            round(norm_novelty[d], 4),
            "pagerankScore":          round(norm_pagerank[d], 4),
        }
        for d in doc_ids
    ]
    scored.sort(key=lambda x: -x["finalScore"])
    top_k = scored[:k]

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "k":           k,
        "topK":        top_k,
        "edges":       edges,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[heuristic] Top-{k} documents:")
    for entry in top_k:
        print(f"  {entry['filename']}  (score={entry['finalScore']})")
    print(f"[heuristic] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Rank documents by BM-25 (top-m chunk representativeness + novelty) + citation PageRank"
    )
    parser.add_argument("--k", type=int, default=K, help=f"Number of top documents to select (default {K})")
    args = parser.parse_args()
    run(k=args.k)