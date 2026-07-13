# Sapphire — academic paper crawler

The pipeline for academic PDFs. Owns every stage that depends on
academic-only structure (DOIs, GROBID TEI headers, reference lists,
citation graphs):

| File                 | Stage |
|----------------------|-------|
| `extract.py`         | docling + GROBID extraction → doclings.json |
| `doi_regex.js`       | stamp each paper's own DOI (regex over the doc head) |
| `search_doi.js`      | Crossref enrichment (title/authors/abstract/refs/citedBy) |
| `heuristic.py`       | BM25 + PageRank-over-citations top-k ranking |
| `heuristic_utils.py` | scoring / reference-matching primitives |
| `build_graph.js`     | document/section/citation knowledge graph |

Crawler-agnostic stages (chunking, embedding, clustering, shared regexes)
live one level up in `extraction/` and are reused by ruby and topaz.
