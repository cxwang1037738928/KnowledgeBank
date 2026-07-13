"""
heuristic_utils.py — reusable scoring / graph primitives for heuristic.py

Every function here is pure with respect to configuration: all tunables
arrive as arguments (no module-level constants), so heuristic.py owns the
parameters and this module owns the mechanics. Nothing here reads or
writes files.

Contents:
  Tokenisation      tokenise
  BM25              BM25, top_terms
  Doc scoring       topm_chunk_representativeness, whole_doc_representativeness,
                    novelty_score
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

def whole_doc_representativeness(bm25: BM25, doc_tokens: list[str],
                                 query: list[str]) -> float:
    """Original behavior: one BM25 score over the entire document. Kept for
    comparison runs; biased toward long documents via keyword coverage."""
    return bm25.score_tokens(doc_tokens, query)


def topm_chunk_representativeness(bm25: BM25, doc_tokens: list[str],
                                  query: list[str], chunk_words: int,
                                  top_m: int) -> float:
    """
    Length-robust representativeness: mean of the top-m chunk scores.

    The document's token stream is split into fixed windows of
    `chunk_words` tokens; each window is BM25-scored against the cluster
    keywords with avgdl pinned to the window size (so length
    normalization is calibrated to chunks, not full documents); the final
    score is the mean of the `top_m` best windows.

    Why this beats whole-doc scoring: a whole-doc score rewards keyword
    COVERAGE, which grows with length — a 100-page document contains more
    of any keyword set than a 10-pager, even with perfect per-term length
    normalization. Scoring fixed-size windows and keeping only the best m
    means a document ranks on the density of its BEST material: long docs
    get no credit for sheer volume, and (unlike a plain mean over all
    chunks) genuinely broad documents are not punished for having
    off-topic appendices diluting their average.

    The divisor is ALWAYS top_m (missing windows count as zero), so every
    document is graded on the same scale: "how much on-topic material do
    you have, up to m windows' worth." Dividing by the actual window
    count instead would hand short documents an artificial boost — a
    150-word note with one dense window would be scored on that window
    alone, while a paper whose equally-dense section sits among m windows
    gets averaged down. Fixed-divisor semantics give bounded breadth
    credit: material beyond the best m windows earns nothing, and a tiny
    fragment cannot outrank a paper with m strong sections.
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
    Corpus-wide novelty: average IDF of a document's unique vocabulary,
    against the full corpus. Counterweight to representativeness, which
    on its own rewards the most TYPICAL member of a cluster.

    Hapax terms (df == 1) are excluded: they carry maximum IDF but are
    dominated by OCR artifacts, typos, and ligature damage rather than
    genuine vocabulary — on a scanned corpus, unfiltered novelty is
    effectively an OCR-noise detector. A term must appear in at least two
    documents to count as novel vocabulary rather than noise.
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
    Percentile-rank normalization to [0, 1]. Used instead of
    max-normalization because heavy-tailed signals (PageRank especially)
    compress everyone but the top outlier toward zero under score/max,
    erasing the blend weights for the bulk of the corpus.

    Tied raw scores receive the AVERAGE of the ranks they span (fractional
    ranking). Without this, docs tied at PageRank's teleport floor were
    spread across 0..k/(n-1) by dict insertion order — an arbitrary and
    ranking-distorting tiebreak at 75% blend weight. Values are compared
    after rounding to 12 decimals so float noise from iterative solvers
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
    """Normalized surname for author matching. Handles both citation-style
    'Last, First' (surname before the comma) and corpus-style 'First Last'
    (surname is the final token). Initials and punctuation are stripped so
    'Gomez, Aidan N.' and 'Aidan N. Gomez' both yield 'gomez'."""
    name = name.strip().lower()
    if "," in name:
        name = name.split(",", 1)[0]
    tokens = re.findall(r"[a-zà-öø-ÿ'\-]{2,}", name)
    return tokens[-1] if tokens else ""


def _norm_title(s: str) -> str:
    """Normalize a title for containment matching: lowercase, fold all
    punctuation/hyphens to single spaces, collapse whitespace. So
    'BERT: Pre-training…' and 'BERT Pre training …' compare equal, and
    double-spaced OCR titles ('ON  COMPUTABLE  NUMBERS') normalize cleanly."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _norm_doi(doi: str) -> str:
    """Normalize a DOI for exact comparison: lowercase (DOIs are
    case-insensitive), strip a URL / 'doi:' prefix and trailing punctuation.
    So 'https://doi.org/10.3762/BJOC.19.8' and '10.3762/bjoc.19.8' compare
    equal."""
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d.strip().rstrip(".")


def _titles_match(key: str, ref_title: str, min_contained: int) -> bool:
    """Bidirectional title containment with a length guard on the CONTAINED
    side. Exact equality always matches. Proper containment only matches when
    the contained string is at least `min_contained` chars — otherwise a short
    generic title ('networks') spuriously hits every longer title containing
    that word. The guard is an absolute length, not a percentage: the reverse
    direction exists to let a clean short reference title match a corpus title
    polluted with prefixed boilerplate, where the real title may be a small
    fraction of the stored string."""
    if key == ref_title:
        return True
    if key in ref_title and len(key) >= min_contained:
        return True
    if ref_title in key and len(ref_title) >= min_contained:
        return True
    return False


def _index_tokens(norm_title: str) -> set[str]:
    """Informative tokens of a normalized title, for the inverted index.
    Stopwords and 1-2 char tokens are dropped so posting lists stay short;
    titles made entirely of such tokens fall back to all their tokens (the
    exact-title hash join covers them regardless)."""
    tokens = {t for t in norm_title.split()
              if len(t) > 2 and t not in _STOPWORDS}
    return tokens or set(norm_title.split())


def build_connectivity(doclings: dict, min_key_length: int,
                       min_contained_length: int = 15) -> dict[str, set[str]]:
    """
    Directed citation adjacency: source_docId -> set of target_docIds it
    cites. Edges are found in two phases, unioned together:

      Phase 1 — exact DOI match (highest confidence). Crossref reference
        lists (search_doi.js -> crossrefReferences) carry each cited work's
        DOI; when it equals another corpus document's DOI the edge is
        certain, needing no title/author agreement.
      Phase 2 — fuzzy title match (fallback) over GROBID's structured
        parsedReferences (extract.py). The reference's title must match a
        target by bidirectional containment (via _titles_match: exact
        equality, or proper containment where the contained side is at
        least `min_contained_length` chars) AND — when both sides have
        extracted authors — share at least one author surname (via
        _surname, which bridges 'Last, First' vs 'First Last'). Keys
        shorter than `min_key_length` are skipped as too ambiguous.
        When both sides carry a created year (metadata.created), edges
        pointing forward in time are rejected: a paper cannot cite a
        target created more than a year after it (the +1 absorbs
        preprint-vs-published skew).

    Docs without parsedReferences contribute no phase-2 edges — the old
    Ollama fallback for raw reference strings is gone (at 10k docs it meant
    tens of thousands of LLM calls; GROBID is the only supported parser).

    Phase-2 candidates come from indexes rather than scanning every corpus
    title per reference (O(refs x titles) dies at 10k docs): an exact hash
    join on the normalized title, plus an inverted token index — only
    corpus titles sharing >= 2 informative tokens with the reference title
    (>= 1 when either side has a single informative token) are verified
    with _titles_match. Containment implies the contained side's tokens
    all appear in the containing side, so the token filter never drops a
    true containment match with >= 2 informative tokens.
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
                # Two corpus docs normalize to the same title (e.g. arXiv +
                # published version). Keep the first deterministically; edges
                # will only ever point at that one.
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
            # A 1-token corpus title contained in a longer reference title
            # shares only that token — it can never reach the 2-token bar.
            single_token_targets[next(iter(tokens))].add(target_id)

    adjacency: dict[str, set[str]] = {doc_id: set() for doc_id in doclings}

    # --- Phase 1: exact DOI matching -----------------------------------------
    # Crossref reference DOIs that resolve to a corpus document are certain
    # citations. References without a resolvable DOI fall through to the fuzzy
    # title phase below; the two edge sets are unioned in `adjacency`.
    for doc_id, entry in doclings.items():
        for ref in (entry.get("crossrefReferences") or []):
            ref_doi = _norm_doi(ref.get("doi") or "")
            if not ref_doi:
                continue
            target_id = doi_lookup.get(ref_doi)
            if target_id and target_id != doc_id:
                adjacency[doc_id].add(target_id)

    # --- Phase 2: fuzzy title matching (fallback) ----------------------------
    for doc_id, entry in doclings.items():
        src_year = year_of.get(doc_id)
        for ref in (entry.get("parsedReferences") or []):
            ref_title = _norm_title(ref.get("title") or "")
            if not ref_title or len(ref_title) < min_key_length:
                continue

            # Candidate targets from the indexes: exact hash join, plus corpus
            # titles sharing enough informative tokens for containment to be
            # possible. Verified below with _titles_match, so candidate
            # generation only needs recall.
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
                # Bidirectional containment: the corpus title inside the
                # reference title OR vice-versa (see _titles_match). The
                # reverse direction rescues targets whose stored title carries
                # extra text (e.g. a copyright banner GROBID glued onto the
                # real title); the contained-side length guard stops short
                # generic titles ('networks') from hitting every longer title.
                if not _titles_match(key_of[target_id], ref_title, min_contained_length):
                    continue
                # A source can't cite a target created after it. Only enforced
                # when both years are known; +1 year of slack absorbs
                # preprint-vs-published and month-unknown skew.
                tgt_year = year_of.get(target_id)
                if src_year and tgt_year and tgt_year > src_year + 1:
                    continue
                # If the corpus doc has no extracted authors we can't verify
                # via author matching — accept the title match alone rather
                # than silently dropping the edge. If both sides have authors,
                # require at least one surname in common.
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