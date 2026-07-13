"""
heuristic.py — select the k most important documents

Orchestration only: every scoring/graph primitive lives in
heuristic_utils.py; this file owns the PARAMETERS and the pipeline glue
(read doclings/categories → score → rank → write heuristic_output.json).

Scoring model:
  final_score = ALPHA * bm25_component + (1 - ALPHA) * pagerank_score

  bm25_component — blend of two sub-signals:
    - representativeness: mean of the TOP_M_CHUNKS best chunk-level BM25
      scores against the document's cluster keywords (read straight from
      categories.json — generate_categories.js is the single source of
      keyword truth). Top-m chunk scoring ranks documents on the density of
      their best material: long docs get no credit for filler, and (unlike
      a plain per-chunk average) broad documents aren't punished for
      off-topic appendices. See topm_chunk_representativeness().
      Normalized WITHIN each cluster (percentile among cluster members),
      not globally: every doc is graded against its own cluster's keywords,
      so cross-cluster raw scores were never comparable — a singleton
      trivially aces the exam its own vocabulary wrote.
    - novelty: average IDF of a document's unique non-hapax vocabulary vs.
      the full corpus — counterweight to representativeness, which rewards
      typicality. Percentile-normalized globally.

  pagerank_score — PageRank over the citation graph built from GROBID's
    structured parsedReferences (DOI exact match + indexed fuzzy title
    match with author + created-year sanity checks; no LLM). Rank flows
    through incoming edges only. Percentile-normalized globally.

Top-k selection uses per-cluster quotas, proportional to cluster size
(largest-remainder apportionment): a cluster holding 40% of the corpus gets
~40% of the slots, each filled by its highest-scoring members. Singletons
only compete for remainder slots, which neutralizes their within-cluster
percentile of 1.0.

Writes data/heuristic_output.json:
  topK  : [{docId, filename, cluster, finalScore, bm25Score,
            bm25Representativeness, bm25Novelty, pagerankScore}]
  edges : [{source, target}]  citation edges across the full corpus

Dependencies: networkx (pip install networkx).
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from heuristic_utils import (
    BM25,
    build_connectivity,
    compute_pagerank,
    novelty_score,
    percentile_normalize,
    tokenise,
    top_terms,
    topm_chunk_representativeness,
)

# ---------------------------------------------------------------------------
# Parameters — the single place tunables live
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).resolve().parents[3]
DATA_DIR        = ROOT / os.environ.get("DATA_DIR", "data")
DOCLINGS_PATH   = DATA_DIR / "doclings.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"
OUTPUT_PATH     = DATA_DIR / "heuristic_output.json"

K              = 2      # top-k documents to select
ALPHA          = 0.25   # weight of the BM25 component vs. PageRank
NOVELTY_WEIGHT = 0.2    # how much a document's rare vocabulary boosts its final ranking score

BM25_K1        = 1.5    # term-frequency saturation
BM25_B         = 0.75   # length normalization strength

CHUNK_WORDS         = 180   # window size (tokens) for chunk-level scoring —
                            # matches the embedding pipeline's CHUNK_SIZE
TOP_M_CHUNKS        = 5     # how many best chunks define representativeness
KEYWORDS_N          = int(os.environ.get("KEYWORDS_N", "20"))

_REF_HEADINGS = frozenset({"references", "bibliography", "works cited", "literature cited", "citations"})

MIN_KEY_LENGTH      = 4   # skip title/author match keys shorter than this, short author last names causes over matching
MIN_CONTAINED_TITLE = 15  # proper title containment (one inside the other) only
                          # counts when the contained title has at least this many
                          # chars — blocks short generic titles ('networks') from
                          # matching every longer title that mentions the word
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
    # cited-paper titles don't inflate keyword scores. Headings are normalized
    # first ('7. References', 'REFERENCES.' → 'references') — an exact-string
    # check silently let numbered bibliography headings through, leaking
    # rare-author-name tokens into BM25 novelty and cluster keywords.
    def _norm_heading(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r'^[\divxlc]+[\.\)]?\s+', '', t)   # '7. ' / 'vii. ' / '7) '
        return t.rstrip(' .:')

    def _body_text(entry: dict) -> str:
        sections = entry.get("sections", [])
        body = [s["text"] for s in sections
                if _norm_heading(s.get("heading", "")) not in _REF_HEADINGS
                and (s.get("text") or "").strip()]
        return " ".join(body) if body else entry.get("text", "")

    tokenised = {d: tokenise(_body_text(doclings[d])) for d in doc_ids}

    bm25 = BM25([tokenised[d] for d in doc_ids], k1=BM25_K1, b=BM25_B)

    # --- Clusters + keywords from categories.json ----------------------------
    # generate_categories.js is the single source of keyword truth (TF-IDF over
    # cluster body text) — recomputing them here with a different formula made
    # the two stages disagree about what each cluster is about. Docs absent
    # from categories.json (stale file, or no file at all) form one pseudo-
    # cluster graded against corpus-wide fallback keywords.
    fallback_kws = top_terms(bm25, n=KEYWORDS_N)
    clusters: list[list[str]] = []      # member doc_ids per cluster
    cluster_kws: list[list[str]] = []   # keywords per cluster (parallel)
    if CATEGORIES_PATH.exists():
        categories = json.loads(CATEGORIES_PATH.read_text(encoding='utf-8'))
        for cat in categories.get("categories", []):
            member_ids = [m["docId"] for m in cat.get("members", []) if m["docId"] in tokenised]
            if not member_ids:
                continue
            clusters.append(member_ids)
            cluster_kws.append(cat.get("keywords") or fallback_kws)
    categorized = {d for members in clusters for d in members}
    leftover = [d for d in doc_ids if d not in categorized]
    if leftover:
        clusters.append(leftover)
        cluster_kws.append(fallback_kws)

    # --- BM25 component: top-m chunk representativeness + novelty ------------
    # Representativeness is percentile-normalized WITHIN each cluster: every
    # doc is scored against its own cluster's keywords, so raw scores are not
    # comparable across clusters (a singleton is graded on an exam its own
    # vocabulary wrote and trivially scores highest — its within-cluster
    # percentile is 1.0 by definition, which the per-cluster quotas below
    # neutralize). Novelty is a corpus-level signal, normalized globally.
    norm_repr: dict[str, float] = {}
    cluster_of: dict[str, int] = {}
    for ci, members in enumerate(clusters):
        raw = {
            d: topm_chunk_representativeness(
                bm25, tokenised[d], cluster_kws[ci],
                chunk_words=CHUNK_WORDS, top_m=TOP_M_CHUNKS,
            )
            for d in members
        }
        norm_repr.update(percentile_normalize(raw))
        for d in members:
            cluster_of[d] = ci

    raw_novelty  = {d: novelty_score(bm25, tokenised[d]) for d in doc_ids}
    norm_novelty = percentile_normalize(raw_novelty)
    bm25_component = {
        d: (1 - NOVELTY_WEIGHT) * norm_repr[d] + NOVELTY_WEIGHT * norm_novelty[d]
        for d in doc_ids
    }

    # --- Citation graph + PageRank -------------------------------------------
    print("[heuristic] Building citation connectivity graph ...")
    adjacency = build_connectivity(doclings, min_key_length=MIN_KEY_LENGTH,
                                   min_contained_length=MIN_CONTAINED_TITLE)

    edges = [{"source": src, "target": tgt}
             for src, targets in adjacency.items() for tgt in targets]

    norm_pagerank = percentile_normalize(
        compute_pagerank(adjacency, doc_ids, damping=PAGERANK_DAMPING)
    )

    # --- Combine, apportion, rank, write --------------------------------------
    scored = [
        {
            "docId":                  d,
            "filename":               filenames[d],
            "cluster":                cluster_of[d],
            "finalScore":             round(ALPHA * bm25_component[d] + (1 - ALPHA) * norm_pagerank[d], 4),
            "bm25Score":              round(bm25_component[d], 4),
            "bm25Representativeness": round(norm_repr[d], 4),
            "bm25Novelty":            round(norm_novelty[d], 4),
            "pagerankScore":          round(norm_pagerank[d], 4),
        }
        for d in doc_ids
    ]

    # Per-cluster quotas, proportional to cluster size (largest-remainder
    # apportionment, capacity-capped): a cluster with 40% of the docs gets
    # ~40% of the k slots, each filled by its highest-scoring members. This
    # is what actually neutralizes the singleton bias — a singleton's
    # within-cluster repr is 1.0 by construction, but it only competes for
    # remainder slots.
    k = min(k, len(doc_ids))
    sizes = [len(members) for members in clusters]
    exact = [k * s / len(doc_ids) for s in sizes]
    quota = [int(e) for e in exact]
    remaining = k - sum(quota)
    # Largest fractional remainder first; ties broken by larger cluster,
    # then cluster index, so apportionment is deterministic run-to-run.
    order = sorted(range(len(clusters)),
                   key=lambda ci: (-(exact[ci] - quota[ci]), -sizes[ci], ci))
    while remaining > 0:
        for ci in order:
            if remaining == 0:
                break
            if quota[ci] < sizes[ci]:
                quota[ci] += 1
                remaining -= 1

    by_cluster: dict[int, list[dict]] = {}
    for entry in scored:
        by_cluster.setdefault(entry["cluster"], []).append(entry)
    top_k: list[dict] = []
    for ci, members in by_cluster.items():
        members.sort(key=lambda x: -x["finalScore"])
        top_k.extend(members[:quota[ci]])
    top_k.sort(key=lambda x: -x["finalScore"])

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