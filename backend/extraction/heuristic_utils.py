"""
heuristic_utils.py — reusable scoring / graph primitives for heuristic.py

Every function here is pure with respect to configuration: all tunables
arrive as arguments (no module-level constants), so heuristic.py owns the
parameters and this module owns the mechanics. Nothing here reads or
writes files.

Contents:
  Tokenisation      tokenise
  BM25              BM25, top_terms, cluster_keywords
  Doc scoring       topm_chunk_representativeness, whole_doc_representativeness,
                    novelty_score
  Normalization     percentile_normalize
  Citation graph    parse_references_ollama, build_connectivity, compute_pagerank
"""

import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
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


def cluster_keywords(member_token_lists: list[list[str]], bm25: BM25,
                     n: int, per_doc_cap: int | None = None) -> list[str]:
    """
    Per-cluster keywords: summed IDF over members' unique terms.

    `per_doc_cap` limits how many terms each member may contribute (its
    top-cap terms by IDF). Without a cap, a 100-page member with a 5-10x
    larger vocabulary than its 10-page siblings dominates the keyword
    pool — and representativeness then grades every member against a
    keyword set the longest member effectively wrote. Pass None to
    restore the old uncapped behavior.
    """
    scores: dict[str, float] = defaultdict(float)
    for tokens in member_token_lists:
        unique = set(tokens)
        if per_doc_cap is not None and len(unique) > per_doc_cap:
            # (-idf, term): the term tiebreak makes the per-doc cap pick the
            # same terms every run when several share an IDF at the cutoff.
            unique = set(sorted(unique, key=lambda t: (-bm25.idf(t), t))[:per_doc_cap])
        for term in unique:
            scores[term] += bm25.idf(term)
    # Alphabetical tiebreak on the final cutoff too — without it the keyword
    # set (and thus representativeness) varies run-to-run on IDF ties.
    return [t for t, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:n]]


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
    """
    unique_terms = set(doc_tokens)
    if not unique_terms:
        return 0.0
    return sum(bm25.idf(t) for t in unique_terms) / len(unique_terms)


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
# Citation parsing (Ollama) + connectivity + PageRank
# ---------------------------------------------------------------------------

_PARSE_PROMPT = """You are a citation parser. Given the list of bibliographic reference strings below, extract each one's title and author names.
Return ONLY a JSON array with no explanation or markdown fences:
[{{"title": "...", "authors": ["Last, First", ...]}}, ...]

If a field cannot be determined, use an empty string or empty list.

References:
{refs}"""


def _salvage_ref_objects(raw: str) -> list[dict]:
    """Recover complete {title, authors} objects from a truncated or
    malformed JSON array. Entry objects contain no nested braces, so each
    non-greedy {...} block is parseable on its own; a cut-off final entry
    is simply skipped instead of poisoning the whole batch."""
    objects = []
    for m in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


def parse_references_ollama(raw_refs: list[str], ollama_url: str, model: str,
                            batch_size: int = 10, timeout: int = 120) -> list[dict]:
    """
    Parses reference strings into structured {title, authors} dicts via an
    Ollama-hosted model. References are independent of one another, so
    long lists are split into batches of `batch_size` and the parsed
    arrays concatenated. Batches must stay small: a 3b model given 50
    refs at once blows past both the context window (input truncation)
    and the output token budget (done_reason=length, JSON cut mid-entry),
    which silently zeroed every batch. num_ctx/num_predict are raised
    explicitly so Ollama's defaults (4096 ctx, capped output) don't
    truncate; a batch whose JSON still arrives damaged is salvaged
    object-by-object rather than dropped whole.
    """
    parsed: list[dict] = []
    for i in range(0, len(raw_refs), batch_size):
        batch = raw_refs[i:i + batch_size]
        try:
            prompt = _PARSE_PROMPT.format(refs="\n".join(f"- {r}" for r in batch))
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0,
                                  "num_ctx": 8192,
                                  "num_predict": -1}},
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            raw = resp.json().get("response", "").strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            try:
                batch_parsed = json.loads(match.group() if match else raw)
            except json.JSONDecodeError:
                batch_parsed = _salvage_ref_objects(raw)
            if isinstance(batch_parsed, list):
                parsed.extend(p for p in batch_parsed if isinstance(p, dict))
        except Exception:
            continue
    return parsed




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


def build_connectivity(doclings: dict, parse_fn, min_key_length: int,
                       min_contained_length: int = 15) -> dict[str, set[str]]:
    """
    Directed citation adjacency: source_docId -> set of target_docIds it
    cites. Edges are found in two phases, unioned together:

      Phase 1 — exact DOI match (highest confidence). Crossref reference
        lists (search_doi.js -> crossrefReferences) carry each cited work's
        DOI; when it equals another corpus document's DOI the edge is
        certain, needing no title/author agreement.
      Phase 2 — fuzzy title match (fallback). For references without a
        resolvable DOI, the parsed reference's title must match a target by
        bidirectional containment (via _titles_match: exact equality, or
        proper containment where the contained side is at least
        `min_contained_length` chars) AND — when both sides have extracted
        authors — share at least one author surname (via _surname, which
        bridges 'Last, First' vs 'First Last'). Keys shorter than
        `min_key_length` are skipped as too ambiguous.

    Both phases degrade gracefully: a doc with no crossrefReferences simply
    contributes nothing in phase 1 and relies on phase 2; a doc with no DOI
    can still be a phase-2 target. `parse_fn` is injected (raw_refs ->
    [{title, authors}]) so the LLM transport is swappable and the matching
    logic is testable without a model.
    """
    title_lookup: dict[str, str] = {}
    author_lookup: dict[str, set[str]] = defaultdict(set)
    doi_lookup: dict[str, str] = {}

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
        for author in (meta.get("authors") or []):
            a = author.strip().lower()
            if a and len(a) >= min_key_length:
                author_lookup[a].add(doc_id)
        doi = _norm_doi(meta.get("doi") or "")
        if doi:
            doi_lookup[doi] = doc_id

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

    # GROBID (extract.py) already ships structured {title, authors} refs in
    # parsedReferences — use them directly. Only docs missing them (extracted
    # before the GROBID integration, or while the server was down) go through
    # the LLM parse, fired in parallel since each call is independent I/O.
    parsed_map: dict[str, list[dict]] = {}
    needs_llm: dict[str, list[str]] = {}
    for doc_id, entry in doclings.items():
        pre_parsed = entry.get("parsedReferences") or []
        raw_refs   = entry.get("references") or []
        if pre_parsed:
            parsed_map[doc_id] = pre_parsed
        elif raw_refs:
            needs_llm[doc_id] = raw_refs
        else:
            parsed_map[doc_id] = []

    if needs_llm:
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(parse_fn, refs): doc_id
                for doc_id, refs in needs_llm.items()
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    parsed_map[doc_id] = future.result()
                except Exception:
                    parsed_map[doc_id] = []

    # --- Phase 2: fuzzy title matching (fallback) ----------------------------
    for doc_id, parsed in parsed_map.items():
        for ref in parsed:
            ref_title = _norm_title(ref.get("title") or "")
            ref_authors = [a.strip().lower() for a in ref.get("authors", []) if a.strip()]

            if not ref_title or len(ref_title) < min_key_length:
                continue

            # Bidirectional containment: match when the corpus title is inside
            # the reference title OR vice-versa (see _titles_match). The
            # reverse direction rescues targets whose stored title carries
            # extra text (e.g. a copyright banner GROBID glued onto the real
            # title); the contained-side length guard stops short generic
            # titles ('networks') from hitting every longer title.
            title_candidates = {
                target_id for key, target_id in title_lookup.items()
                if target_id != doc_id
                and _titles_match(key, ref_title, min_contained_length)
            }
            if not title_candidates:
                continue

            ref_surnames = {s for s in (_surname(a) for a in ref_authors) if s}

            for target_id in title_candidates:
                target_authors = {a for a, ids in author_lookup.items() if target_id in ids}
                target_surnames = {s for s in (_surname(a) for a in target_authors) if s}
                # If the corpus doc has no extracted authors we can't verify via
                # author matching — accept the title match alone rather than
                # silently dropping the edge. If both sides have authors,
                # require at least one surname in common.
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