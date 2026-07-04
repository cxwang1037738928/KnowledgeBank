/**
 * bootstrap_queries.js — synthesize plausible user queries per category
 *
 * For every cluster in data/categories.json, assembles a context pack and
 * prompts Phi-4 (via Ollama) to generate the kinds of questions a user
 * might actually ask against that slice of the corpus. Output is used to
 * pre-warm/evaluate retrieval and to seed query-parsing tests.
 *
 * Context pack per category (all computed locally, no LLM needed):
 *   - top BM25-IDF keywords   (same summed-IDF-over-members logic as heuristic.py)
 *   - member titles           (metadata.title, falling back to filename)
 *   - abstracts               (metadata.abstract, falling back to the first
 *                              non-trivial section — usually the abstract in
 *                              academic papers)
 *   - distinctive headings    (section headings that recur across members or
 *                              are unusually specific — skips boilerplate like
 *                              "Introduction"/"References")
 *   - medoid excerpt          (the chunk whose embedding is closest to the
 *                              category centroid — the single most
 *                              "representative" passage of the cluster)
 *
 * Reads:  data/categories.json, data/doclings.json, data/embeddings.json (optional)
 * Writes: data/bootstrap_queries.json
 *
 * Run: node backend/extraction/bootstrap_queries.js --per-category 8 --model phi4
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT            = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const CATEGORIES_PATH = path.join(ROOT, 'data', 'categories.json');
const DOCLINGS_PATH   = path.join(ROOT, 'data', 'doclings.json');
const EMBEDDINGS_PATH = path.join(ROOT, 'data', 'embeddings.json');
const OUTPUT_PATH     = path.join(ROOT, 'data', 'bootstrap_queries.json');

const OLLAMA_URL   = process.env.OLLAMA_URL || 'http://localhost:11434';
const MODEL        = process.env.BOOTSTRAP_MODEL || 'phi4';
const PER_CATEGORY = 8;
const KEYWORDS_N   = 15;

// Query types we ask Phi-4 to cover. Mirrors how real users interrogate a
// document corpus and gives the retrieval evaluator labeled difficulty tiers:
// factual/definition hit single chunks, comparison/synthesis require
// multi-doc retrieval, methodology exercises section-level structure.
const QUERY_TYPES = ['factual', 'definition', 'comparison', 'synthesis', 'methodology'];

const BOILERPLATE_HEADINGS = new Set([
  'abstract', 'introduction', 'conclusion', 'conclusions', 'references',
  'acknowledgments', 'acknowledgements', 'appendix', 'related work', 'discussion',
]);

// ---------------------------------------------------------------------------
// Tokenisation + IDF (mirrors heuristic.py so keywords agree across languages)
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  'the','a','an','and','or','but','in','on','at','to','for','of','with',
  'by','from','as','is','was','are','were','be','been','being','have',
  'has','had','do','does','did','will','would','could','should','may',
  'might','this','that','these','those','it','its','i','we','you','he',
  'she','they','their','our','us','not','no','so','if','than','then',
]);

function tokenise(text) {
  const tokens = (text.toLowerCase().match(/[a-z]+/g) || []);
  return tokens.filter((t) => !STOPWORDS.has(t) && t.length > 2);
}

/** Corpus-wide document frequency + IDF, identical formula to heuristic.py. */
function buildIdf(tokenisedDocs) {
  const N = tokenisedDocs.length;
  const df = new Map();
  for (const tokens of tokenisedDocs) {
    for (const t of new Set(tokens)) df.set(t, (df.get(t) || 0) + 1);
  }
  return (term) => {
    const d = df.get(term) || 0;
    return Math.log((N - d + 0.5) / (d + 0.5) + 1);
  };
}

/** Per-cluster keywords: summed IDF over members' unique terms (heuristic.py port). */
function clusterKeywords(memberTokens, idf, n = KEYWORDS_N) {
  const scores = new Map();
  for (const tokens of memberTokens) {
    for (const t of new Set(tokens)) scores.set(t, (scores.get(t) || 0) + idf(t));
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([t]) => t);
}

// ---------------------------------------------------------------------------
// Context assembly helpers
// ---------------------------------------------------------------------------

function docTitle(entry) {
  return entry?.metadata?.title?.trim() || entry?.filename || 'untitled';
}

function docAbstract(entry, maxWords = 120) {
  let abs = entry?.metadata?.abstract?.trim();
  if (!abs) {
    // Fall back: first section with a real body — in papers that's the abstract
    const sec = (entry?.sections || []).find((s) => (s.text || '').trim().length > 200);
    abs = sec?.text?.trim() || '';
  }
  const words = abs.split(/\s+/);
  return words.length > maxWords ? words.slice(0, maxWords).join(' ') + ' …' : abs;
}

function distinctiveHeadings(entries, maxHeadings = 8) {
  const counts = new Map();
  for (const entry of entries) {
    for (const sec of entry?.sections || []) {
      const h = (sec.heading || '').trim();
      if (!h) continue;
      const key = h.toLowerCase().replace(/^[0-9.\s]+/, ''); // strip "3.1 " numbering
      if (!key || BOILERPLATE_HEADINGS.has(key) || key.length < 4) continue;
      counts.set(h, (counts.get(h) || 0) + 1);
    }
  }
  // Prefer headings shared by multiple members, then longer/more specific ones
  return [...counts.entries()]
    .sort((a, b) => (b[1] - a[1]) || (b[0].length - a[0].length))
    .slice(0, maxHeadings)
    .map(([h]) => h);
}

/**
 * Medoid excerpt: among all chunks belonging to the category's members,
 * find the one closest to the centroid of member doc vectors. Embeddings
 * are L2-normalised, so dot product = cosine similarity.
 */
function medoidExcerpt(memberIds, embedStore, maxWords = 150) {
  if (!embedStore) return null;
  const memberSet = new Set(memberIds);
  const chunks = (embedStore.chunks || []).filter((c) => memberSet.has(c.docId) && c.embedding);
  if (chunks.length === 0) return null;

  const dim = chunks[0].embedding.length;
  const centroid = new Float64Array(dim);
  for (const c of chunks) for (let i = 0; i < dim; i++) centroid[i] += c.embedding[i];
  let mag = 0;
  for (let i = 0; i < dim; i++) mag += centroid[i] * centroid[i];
  mag = Math.sqrt(mag) || 1;

  let best = null, bestSim = -Infinity;
  for (const c of chunks) {
    let sim = 0;
    for (let i = 0; i < dim; i++) sim += c.embedding[i] * (centroid[i] / mag);
    if (sim > bestSim) { bestSim = sim; best = c; }
  }
  const words = (best.text || '').split(/\s+/);
  return words.length > maxWords ? words.slice(0, maxWords).join(' ') + ' …' : best.text;
}

// ---------------------------------------------------------------------------
// Phi-4 prompt + call
// ---------------------------------------------------------------------------

function buildPrompt(ctx, perCategory) {
  const titles    = ctx.titles.map((t) => `- ${t}`).join('\n');
  const abstracts = ctx.abstracts.filter(Boolean).map((a, i) => `[${i + 1}] ${a}`).join('\n');
  const headings  = ctx.headings.length ? ctx.headings.join('; ') : '(none)';
  const excerpt   = ctx.excerpt ? `\nRepresentative excerpt:\n"""${ctx.excerpt}"""\n` : '';

  return `You generate realistic search queries that a researcher might type into a document-intelligence system indexing the academic papers described below.

Papers in this topic cluster:
${titles}

Top keywords (BM25): ${ctx.keywords.join(', ')}

Notable section headings: ${headings}

Abstracts:
${abstracts}
${excerpt}
Generate exactly ${perCategory} diverse queries. Cover a mix of these types:
- factual: asks for a specific fact, number, or result reported in one paper
- definition: asks what a term or concept from the keywords means
- comparison: contrasts approaches/results across two or more of these papers
- synthesis: a broad question whose answer must combine several papers
- methodology: asks how an experiment, dataset, or technique was set up

Rules:
- Phrase them like real user queries (mixed style: some keyword-ish, some full questions).
- Ground every query in the context above; do not invent papers or topics not present.
- Return ONLY a JSON array, no explanation, no markdown fences:
[{"query": "...", "type": "factual|definition|comparison|synthesis|methodology"}, ...]`;
}

async function generateQueries(prompt, model) {
  const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      options: { temperature: 0.8 }, // want diversity, unlike citation parsing
    }),
  });
  if (!resp.ok) throw new Error(`Ollama HTTP ${resp.status}`);
  const raw = ((await resp.json()).response || '').trim();

  // Same defensive parse as heuristic.py: grab the outermost JSON array
  const match = raw.match(/\[[\s\S]*\]/);
  const parsed = JSON.parse(match ? match[0] : raw);
  if (!Array.isArray(parsed)) throw new Error('model did not return a JSON array');

  return parsed
    .filter((q) => q && typeof q.query === 'string' && q.query.trim())
    .map((q) => ({
      query: q.query.trim(),
      type: QUERY_TYPES.includes(q.type) ? q.type : 'factual',
    }));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export async function bootstrapQueries({ perCategory = PER_CATEGORY, model = MODEL } = {}) {
  let categories, doclings, embedStore = null;

  try {
    categories = JSON.parse(await fs.readFile(CATEGORIES_PATH, 'utf-8'));
  } catch {
    throw new Error('data/categories.json not found — run generate_categories.js first');
  }
  try {
    doclings = JSON.parse(await fs.readFile(DOCLINGS_PATH, 'utf-8'));
  } catch {
    throw new Error('data/doclings.json not found — run extract.py first');
  }
  try {
    embedStore = JSON.parse(await fs.readFile(EMBEDDINGS_PATH, 'utf-8'));
  } catch {
    console.warn('[bootstrap] embeddings.json not found — skipping medoid excerpts');
  }

  // Corpus-wide IDF once; reused for every cluster (same as heuristic.py)
  const docIds = Object.keys(doclings);
  const tokenised = new Map(docIds.map((d) => [d, tokenise(doclings[d].text || '')]));
  const idf = buildIdf([...tokenised.values()]);

  const results = [];
  const cats = categories.categories || [];

  for (let ci = 0; ci < cats.length; ci++) {
    const memberIds = (cats[ci].members || [])
      .map((m) => m.docId)
      .filter((d) => doclings[d]);
    if (memberIds.length === 0) continue;

    const entries = memberIds.map((d) => doclings[d]);
    const ctx = {
      keywords:  clusterKeywords(memberIds.map((d) => tokenised.get(d) || []), idf),
      titles:    entries.map(docTitle),
      abstracts: entries.map((e) => docAbstract(e)).slice(0, 5), // cap prompt size
      headings:  distinctiveHeadings(entries),
      excerpt:   medoidExcerpt(memberIds, embedStore),
    };

    console.log(`[bootstrap] Category ${ci + 1}/${cats.length} (${memberIds.length} docs): ${ctx.keywords.slice(0, 5).join(', ')} ...`);

    let queries = [];
    try {
      queries = await generateQueries(buildPrompt(ctx, perCategory), model);
    } catch (err) {
      console.error(`[bootstrap]   Phi-4 generation failed: ${err.message} — skipping category`);
    }

    results.push({
      categoryIndex: ci,
      members: memberIds.map((d) => ({ docId: d, filename: doclings[d].filename })),
      keywords: ctx.keywords,
      queries,
    });
  }

  const output = {
    generatedAt: new Date().toISOString(),
    model,
    perCategory,
    threshold: categories.threshold ?? null,
    categories: results,
  };

  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.writeFile(OUTPUT_PATH, JSON.stringify(output, null, 2), 'utf-8');

  const total = results.reduce((s, c) => s + c.queries.length, 0);
  console.log(`[bootstrap] ${total} queries across ${results.length} categories → ${OUTPUT_PATH}`);
  return output;
}

// Run directly: node backend/extraction/bootstrap_queries.js --per-category 8 --model phi4
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv;
  const flag = (name, fallback) => {
    const i = argv.indexOf(name);
    return i !== -1 ? argv[i + 1] : fallback;
  };
  bootstrapQueries({
    perCategory: parseInt(flag('--per-category', PER_CATEGORY), 10),
    model: flag('--model', MODEL),
  }).catch((err) => {
    console.error('[bootstrap]', err.message);
    process.exit(1);
  });
}