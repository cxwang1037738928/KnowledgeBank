"""
heuristic.py — Stage 2: select the k most important documents

Scoring model:
  final_score = ALPHA * bm25_score + (1 - ALPHA) * connectivity_score

  bm25_score      — how much a document contains its category's top keywords;
                    falls back to corpus-wide discriminative terms when
                    data/categories.json hasn't been generated yet.
  connectivity_score — normalised in-degree + out-degree in the citation
                    graph built by matching Grobid-extracted references against
                    titles/authors in data/doclings.json.

Writes data/heuristic_output.json with:
  top_k       : [{docId, filename, finalScore, bm25Score, connectivityScore}]
  edges       : [{source, target}]  citation edges across the full corpus
"""

import json
import re
import sys
import math
from pathlib import Path
from collections import defaultdict

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).resolve().parents[2]
DOCLINGS_PATH = ROOT / "data" / "doclings.json"
CATEGORIES_PATH = ROOT / "data" / "categories.json"
OUTPUT_PATH   = ROOT / "data" / "heuristic_output.json"

GROBID_URL    = "http://localhost:8070/api/processReferences"
K             = 5
ALPHA         = 0.25   # weight between BM-25 and connectivity
BM25_K1       = 1.5
BM25_B        = 0.75

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","this","that","these","those","it","its","i","we","you","he",
    "she","they","their","our","us","not","no","so","if","than","then",
}

def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


# ---------------------------------------------------------------------------
# BM-25 helpers
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.k1   = k1
        self.b    = b
        self.N    = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.df: dict[str, int] = defaultdict(int)
        self.tf_per_doc: list[dict[str, int]] = []

        for doc in corpus:
            tf: dict[str, int] = defaultdict(int)
            seen: set[str] = set()
            for term in doc:
                tf[term] += 1
                if term not in seen:
                    self.df[term] += 1
                    seen.add(term)
            self.tf_per_doc.append(tf)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score_doc(self, doc_tokens: list[str], query: list[str]) -> float:
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            tf_map[t] += 1
        dl = len(doc_tokens)
        score = 0.0
        for term in set(query):
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf(term)
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += idf * tf_norm
        return score


def top_terms(bm25: BM25, n: int = 20) -> list[str]:
    """Return the n terms with the highest summed BM25 score across all docs."""
    term_scores: dict[str, float] = defaultdict(float)
    for doc_tokens in [list(tf.keys()) for tf in bm25.tf_per_doc]:
        for term in set(doc_tokens):
            term_scores[term] += bm25.idf(term)
    return [t for t, _ in sorted(term_scores.items(), key=lambda x: -x[1])[:n]]


# ---------------------------------------------------------------------------
# Grobid citation extraction
# ---------------------------------------------------------------------------

def extract_references_grobid(text: str) -> list[str]:
    """
    POST the document text to Grobid's processReferences endpoint.
    Returns a list of extracted reference strings.
    Falls back to [] on any error (caller handles missing Grobid gracefully).
    """
    try:
        resp = requests.post(
            GROBID_URL,
            data={"citations": text, "consolidateCitations": "0"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        # Grobid returns TEI XML; extract <title> and <author> text as a quick
        # bag-of-words match target — full TEI parsing would need lxml.
        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.text)
        refs = []
        for bibl in root.iter("{http://www.tei-c.org/ns/1.0}biblStruct"):
            title_el = bibl.find(".//{http://www.tei-c.org/ns/1.0}title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            authors = [
                " ".join(filter(None, [
                    (n.findtext("{http://www.tei-c.org/ns/1.0}forename") or "").strip(),
                    (n.findtext("{http://www.tei-c.org/ns/1.0}surname")  or "").strip(),
                ]))
                for n in bibl.findall(".//{http://www.tei-c.org/ns/1.0}persName")
            ]
            combined = " ".join(filter(None, [title] + authors))
            if combined:
                refs.append(combined)
        return refs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Citation → connectivity graph
# ---------------------------------------------------------------------------

def build_connectivity(doclings: dict) -> dict[str, set[str]]:
    """
    For each document, extract citations (via Grobid) and match them against
    the title/authors of every other document using exact substring matching.
    Returns adjacency dict: source_docId → set of target_docIds it cites.
    """
    # Pre-build a lookup: lowercased title/author strings → docId
    lookup: dict[str, str] = {}
    for doc_id, entry in doclings.items():
        meta = entry.get("metadata", {})
        title = (meta.get("title") or "").strip().lower()
        if title:
            lookup[title] = doc_id
        for author in (meta.get("authors") or []):
            a = author.strip().lower()
            if a:
                lookup[a] = doc_id

    adjacency: dict[str, set[str]] = {doc_id: set() for doc_id in doclings}

    for doc_id, entry in doclings.items():
        # Use raw references from docling first; enrich with Grobid if available
        raw_refs = entry.get("references", [])
        grobid_refs = extract_references_grobid(entry.get("text", "")[:50_000])
        all_refs = raw_refs + grobid_refs

        for ref in all_refs:
            ref_lower = ref.lower()
            for key, target_id in lookup.items():
                if target_id != doc_id and key and key in ref_lower:
                    adjacency[doc_id].add(target_id)

    return adjacency


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def run(k: int = K) -> None:
    if not DOCLINGS_PATH.exists():
        print(f"[heuristic] {DOCLINGS_PATH} not found — run extract.py first.", file=sys.stderr)
        sys.exit(1)

    doclings: dict = json.loads(DOCLINGS_PATH.read_text())
    if not doclings:
        print("[heuristic] doclings.json is empty — nothing to score.")
        return

    doc_ids   = list(doclings.keys())
    filenames = {d: doclings[d]["filename"] for d in doc_ids}
    texts     = {d: doclings[d].get("text", "") for d in doc_ids}
    tokenised = {d: tokenise(texts[d]) for d in doc_ids}
    corpus    = [tokenised[d] for d in doc_ids]

    bm25 = BM25(corpus)

    # Determine query keywords: prefer per-category from categories.json,
    # fall back to corpus-wide discriminative terms.
    if CATEGORIES_PATH.exists():
        categories = json.loads(CATEGORIES_PATH.read_text())
        # Map docId → its category's top keywords
        doc_keywords: dict[str, list[str]] = {}
        for cat in categories.get("categories", []):
            kws = [kw["term"] for kw in cat.get("topKeywords", [])]
            for member in cat.get("members", []):
                doc_keywords[member["docId"]] = kws
        fallback_kws = top_terms(bm25, n=20)
        for d in doc_ids:
            if d not in doc_keywords:
                doc_keywords[d] = fallback_kws
    else:
        fallback_kws = top_terms(bm25, n=20)
        doc_keywords = {d: fallback_kws for d in doc_ids}

    # BM-25 scores
    raw_bm25 = {d: bm25.score_doc(tokenised[d], doc_keywords[d]) for d in doc_ids}
    max_bm25 = max(raw_bm25.values(), default=1) or 1
    norm_bm25 = {d: raw_bm25[d] / max_bm25 for d in doc_ids}

    # Connectivity scores
    print("[heuristic] Building citation connectivity graph ...")
    adjacency = build_connectivity(doclings)
    in_degree  = defaultdict(int)
    out_degree = defaultdict(int)
    edges = []
    for src, targets in adjacency.items():
        for tgt in targets:
            out_degree[src] += 1
            in_degree[tgt]  += 1
            edges.append({"source": src, "target": tgt})

    max_deg = max((in_degree[d] + out_degree[d] for d in doc_ids), default=1) or 1
    norm_conn = {d: (in_degree[d] + out_degree[d]) / max_deg for d in doc_ids}

    # Final scores
    scored = [
        {
            "docId":             d,
            "filename":          filenames[d],
            "finalScore":        round(ALPHA * norm_bm25[d] + (1 - ALPHA) * norm_conn[d], 4),
            "bm25Score":         round(norm_bm25[d], 4),
            "connectivityScore": round(norm_conn[d], 4),
        }
        for d in doc_ids
    ]
    scored.sort(key=lambda x: -x["finalScore"])
    top_k = scored[:k]

    output = {
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "k":           k,
        "topK":        top_k,
        "edges":       edges,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[heuristic] Top-{k} documents:")
    for entry in top_k:
        print(f"  {entry['filename']}  (score={entry['finalScore']})")
    print(f"[heuristic] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score and rank documents by BM-25 + citation connectivity")
    parser.add_argument("--k", type=int, default=K, help=f"Number of top documents to select (default {K})")
    args = parser.parse_args()
    run(k=args.k)
