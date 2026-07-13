"""
heuristic_utils.py — reusable scoring / graph primitives for heuristic.py

Every function here is pure with respect to configuration: all tunables
arrive as arguments (no module-level constants), so heuristic.py owns the
parameters and this module owns the mechanics. Nothing here reads or
writes files.

Contents:
  Tokenisation      tokenise
  BM25              BM25, top_terms
  Doc scoring       topm_chunk_representativeness, novelty_score
  Normalization     percentile_normalize
  Citation graph    build_connectivity, compute_pagerank
"""

import math
import re
import sys
from collections import Counter, defaultdict

import networkx as nx

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
# BM-25
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
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

    def score_tokens(self, doc_tokens: list[str], query: list[str],
                     avgdl: float | None = None) -> float:
        """
        BM25 score of a token sequence against a query term set.
        `avgdl` overrides the corpus average document length — pass the
        expected CHUNK length when scoring chunk-sized windows, so length
        normalization is calibrated to the unit actually being scored
        rather than to full documents.
        """
        avgdl = avgdl if avgdl is not None else self.avgdl
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
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
            score += idf * tf_norm
        return score


def top_terms(bm25: BM25, n: int) -> list[str]:
    """The n terms with the highest summed IDF across all docs (corpus-wide fallback keywords)."""
    term_scores: dict[str, float] = defaultdict(float)
    for tf in bm25.tf_per_doc:
        for term in tf.keys():
            term_scores[term] += bm25.idf(term)
    # Secondary alphabetical key so ties at the top-n cutoff resolve the same
    # way every run (dict/set iteration order is hash-seed dependent).
    return [t for t, _ in sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))[:n]]


# ---------------------------------------------------------------------------
# Document scoring
# ---------------------------------------------------------------------------

def topm_chunk_representativeness(bm25: BM25, doc_tokens: list[str],
                                  query: list[str], chunk_words: int,
                                  top_m: int) -> float:
    """
    Mean of the top-m fixed-window BM25 scores (avgdl pinned to the window
    size). Ranks a document on the density of its best material: whole-doc
    scoring rewards keyword coverage, which grows with length.

    The divisor is ALWAYS top_m (missing windows count as zero) so every
    document is graded on the same scale — dividing by the actual window
    count would let a 150-word fragment with one dense window outrank a
    paper with m strong sections.
    """
    if not doc_tokens:
        return 0.0
    windows = [doc_tokens[i:i + chunk_words]
               for i in range(0, len(doc_tokens), chunk_words)]
    # Drop a trailing fragment window when the doc has full windows to
    # spare — a 30-token tail scores erratically under chunk-calibrated
    # normalization.
    if len(windows) > 1 and len(windows[-1]) < chunk_words // 4:
        windows.pop()
    scores = sorted(
        (bm25.score_tokens(w, query, avgdl=float(chunk_words)) for w in windows),
        reverse=True,
    )[:top_m]
    return sum(scores) / top_m


def novelty_score(bm25: BM25, doc_tokens: list[str]) -> float:
    """
    Average IDF of a document's unique vocabulary — counterweight to
    representativeness, which rewards typicality. Hapax terms (df == 1) are
    excluded: on scanned corpora they're OCR artifacts, not vocabulary.
    """
    scoreable = [t for t in set(doc_tokens) if bm25.df.get(t, 0) >= 2]
    if not scoreable:
        return 0.0
    return sum(bm25.idf(t) for t in scoreable) / len(scoreable)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def percentile_normalize(scores: dict[str, float]) -> dict[str, float]:
    """
    Percentile-rank normalization to [0, 1] (max-normalization compresses
    heavy-tailed signals like PageRank toward zero). Ties get the average of
    the ranks they span, compared at 12 decimals so solver float noise
    doesn't split a genuine tie.
    """
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda x: x[1])
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 1.0}

    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and round(ordered[j + 1][1], 12) == round(ordered[i][1], 12):
            j += 1
        avg_rank = (i + j) / 2
        for k in range(i, j + 1):
            out[ordered[k][0]] = avg_rank / (n - 1)
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# Citation connectivity + PageRank
# ---------------------------------------------------------------------------

def _surname(name: str) -> str:
    """Normalized surname: 'Gomez, Aidan N.' and 'Aidan N. Gomez' → 'gomez'."""
    name = name.strip().lower()
    if "," in name:
        name = name.split(",", 1)[0]
    tokens = re.findall(r"[a-zà-öø-ÿ'\-]{2,}", name)
    return tokens[-1] if tokens else ""


def _norm_title(s: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _norm_doi(doi: str) -> str:
    """Lowercase; strip URL/'doi:' prefix and trailing punctuation."""
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d.strip().rstrip(".")


def _titles_match(key: str, ref_title: str, min_contained: int) -> bool:
    """Bidirectional containment; the contained side must be at least
    `min_contained` chars so short generic titles ('networks') don't hit
    every longer title containing the word."""
    if key == ref_title:
        return True
    if key in ref_title and len(key) >= min_contained:
        return True
    if ref_title in key and len(ref_title) >= min_contained:
        return True
    return False


def _index_tokens(norm_title: str) -> set[str]:
    """Informative tokens (no stopwords / 1-2 char tokens) for the inverted
    index; all-stopword titles fall back to their full token set."""
    tokens = {t for t in norm_title.split()
              if len(t) > 2 and t not in _STOPWORDS}
    return tokens or set(norm_title.split())


def build_connectivity(doclings: dict, min_key_length: int,
                       min_contained_length: int = 15) -> dict[str, set[str]]:
    """
    Directed citation adjacency: source_docId -> target_docIds it cites.
    Two phases, unioned:

      Phase 1 — exact DOI match on crossrefReferences (certain edges).
      Phase 2 — fuzzy title match on GROBID parsedReferences: bidirectional
        containment (_titles_match) + shared author surname when both sides
        have authors + created-year sanity check (a paper can't cite a
        target created more than a year after it).

    Phase-2 candidates come from an exact-title hash join plus an inverted
    token index (>= 2 shared informative tokens, or 1 for single-token
    titles) — never a full O(refs x titles) scan. Containment implies the
    contained side's tokens all appear in the containing side, so the
    token filter can't drop a true match.
    """
    title_lookup: dict[str, str] = {}          # normalized title -> doc_id
    doi_lookup: dict[str, str] = {}
    surnames_of: dict[str, set[str]] = {}      # doc_id -> author surnames
    year_of: dict[str, int] = {}               # doc_id -> created year

    for doc_id, entry in doclings.items():
        meta = entry.get("metadata", {})
        title = _norm_title(meta.get("title") or "")
        if title and len(title) >= min_key_length:
            if title in title_lookup:
                # e.g. arXiv + published version — keep the first, deterministically
                print(f"[heuristic] WARNING: duplicate corpus title {title!r} — "
                      f"keeping {title_lookup[title]}, ignoring {doc_id} as a citation target",
                      file=sys.stderr)
            else:
                title_lookup[title] = doc_id
        surnames_of[doc_id] = {
            s for s in (_surname(a) for a in (meta.get("authors") or [])
                        if len(a.strip()) >= min_key_length)
            if s
        }
        doi = _norm_doi(meta.get("doi") or "")
        if doi:
            doi_lookup[doi] = doc_id
        created = meta.get("created") or {}
        if isinstance(created, dict) and created.get("year"):
            year_of[doc_id] = created["year"]

    # Inverted token index over corpus titles (phase-2 candidate generation).
    key_of: dict[str, str] = {tid: key for key, tid in title_lookup.items()}
    token_index: dict[str, set[str]] = defaultdict(set)
    single_token_targets: dict[str, set[str]] = defaultdict(set)
    for key, target_id in title_lookup.items():
        tokens = _index_tokens(key)
        for t in tokens:
            token_index[t].add(target_id)
        if len(tokens) == 1:
            # can never reach the 2-shared-token bar — tracked separately
            single_token_targets[next(iter(tokens))].add(target_id)

    adjacency: dict[str, set[str]] = {doc_id: set() for doc_id in doclings}

    # Phase 1: exact DOI matching
    for doc_id, entry in doclings.items():
        for ref in (entry.get("crossrefReferences") or []):
            ref_doi = _norm_doi(ref.get("doi") or "")
            if not ref_doi:
                continue
            target_id = doi_lookup.get(ref_doi)
            if target_id and target_id != doc_id:
                adjacency[doc_id].add(target_id)

    # Phase 2: fuzzy title matching
    for doc_id, entry in doclings.items():
        src_year = year_of.get(doc_id)
        for ref in (entry.get("parsedReferences") or []):
            ref_title = _norm_title(ref.get("title") or "")
            if not ref_title or len(ref_title) < min_key_length:
                continue

            ref_tokens = _index_tokens(ref_title)
            need = 1 if len(ref_tokens) == 1 else 2
            counts = Counter()
            for t in ref_tokens:
                for target_id in token_index.get(t, ()):
                    counts[target_id] += 1
            candidates = {tid for tid, c in counts.items() if c >= need}
            for t in ref_tokens:
                candidates |= single_token_targets.get(t, set())
            exact = title_lookup.get(ref_title)
            if exact:
                candidates.add(exact)

            ref_surnames = {
                s for s in (_surname(a) for a in ref.get("authors", []) if a.strip())
                if s
            }

            for target_id in candidates:
                if target_id == doc_id:
                    continue
                if not _titles_match(key_of[target_id], ref_title, min_contained_length):
                    continue
                # +1 year of slack absorbs preprint-vs-published skew
                tgt_year = year_of.get(target_id)
                if src_year and tgt_year and tgt_year > src_year + 1:
                    continue
                # No extracted authors on the target → accept the title match
                # alone rather than silently dropping the edge.
                target_surnames = surnames_of.get(target_id, set())
                if target_surnames and ref_surnames:
                    if not (ref_surnames & target_surnames):
                        continue
                adjacency[doc_id].add(target_id)

    return adjacency


def compute_pagerank(adjacency: dict[str, set[str]], doc_ids: list[str],
                     damping: float = 0.85) -> dict[str, float]:
    """
    PageRank over the citation graph. Edge src -> tgt means "src cites
    tgt": rank flows through INCOMING edges, so being cited drives score
    and long reference lists don't. Uniform scores on an edgeless graph.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(doc_ids)
    for src, targets in adjacency.items():
        for tgt in targets:
            if tgt in graph:
                graph.add_edge(src, tgt)

    if graph.number_of_edges() == 0:
        return {d: 1.0 / len(doc_ids) for d in doc_ids}

    return nx.pagerank(graph, alpha=damping)