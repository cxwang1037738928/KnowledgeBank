/**
 * bootstrap_queries.js — synthesize plausible user queries per category
 * (Ollama edition — targets Ministral 3B via local Ollama)
 *
 * For every cluster in data/categories.json, assembles a context pack and
 * prompts the model to generate the kinds of questions a user might ask
 * against that slice of the corpus.
 *
 * Context pack per category (all computed locally, no LLM needed):
 *   - top BM25-IDF keywords   (same summed-IDF-over-members logic as heuristic.py)
 *   - member titles           (GROBID metadata.title, falling back to filename)
 *   - abstracts               (GROBID metadata.abstract paired with its title;
 *                              docs with real abstracts are preferred, first
 *                              non-trivial section as last resort)
 *   - distinctive headings    (recurring/specific, boilerplate filtered)
 *   - medoid excerpt          (chunk closest to the category centroid)
 *
 * ---------------------------------------------------------------------------
 * Ollama configuration (.env):
 *   OLLAMA_URL           Ollama base URL (default http://localhost:11434)
 *   BOOTSTRAP_MODEL      Ollama model tag (default ministral:3b)
 *   BOOTSTRAP_MAX_TOKENS output cap per call — passed as num_predict (default 2000)
 *   BOOTSTRAP_INPUT_BUDGET prompt budget in tokens (default 6000)
 *
 * Context-length safety: prompt size is estimated (~4 chars/token) and the
 * context pack is trimmed in stages until it fits BOOTSTRAP_INPUT_BUDGET:
 *   1. drop the medoid excerpt
 *   2. shrink abstracts (fewer docs, fewer words each)
 *   3. shrink headings + keywords + title list
 * Ministral 3B has a 128K context window, but a small model reasons better
 * over a lean prompt anyway, so the default budget is deliberately modest.
 *
 * Reads:  data/categories.json, data/doclings.json, data/embeddings.json (optional)
 * Writes: data/bootstrap_queries.json
 *
 * Run: node backend/extraction/bootstrap_queries.js --per-category 8
 */

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { tokenise, REF_HEADINGS, normHeading } from './regex_utils.js';

const ROOT            = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DATA_DIR        = path.resolve(ROOT, process.env.DATA_DIR || 'data');
const CATEGORIES_PATH = path.join(DATA_DIR, 'categories.json');
const DOCLINGS_PATH   = path.join(DATA_DIR, 'doclings.json');
const EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.json');
const OUTPUT_PATH     = path.join(DATA_DIR, 'bootstrap_queries.json');

const OLLAMA_URL     = process.env.OLLAMA_URL || 'http://localhost:11434';
const MODEL          = process.env.BOOTSTRAP_MODEL || 'ministral:3b';
const MAX_TOKENS     = parseInt(process.env.BOOTSTRAP_MAX_TOKENS || '2000', 10);
const INPUT_BUDGET   = parseInt(process.env.BOOTSTRAP_INPUT_BUDGET || '6000', 10);
const PER_CATEGORY   = parseInt(process.env.BOOTSTRAP_PER_CATEGORY || '8', 10);
const KEYWORDS_N     = parseInt(process.env.KEYWORDS_N || '20', 10);

const QUERY_TYPES = ['factual', 'definition', 'comparison', 'synthesis', 'methodology'];

const BOILERPLATE_HEADINGS = new Set([
  'abstract', 'introduction', 'conclusion', 'conclusions', 'references',
  'acknowledgments', 'acknowledgements', 'appendix', 'related work', 'discussion',
]);

// ---------------------------------------------------------------------------
// Token estimation (~4 chars/token for English — coarse but adequate for
// budget enforcement; err low on the budget, not high on the estimate)
// ---------------------------------------------------------------------------

const estTokens = (s) => Math.ceil((s || '').length / 4);
const capWords  = (s, n) => {
  const words = (s || '').split(/\s+/);
  return words.length > n ? words.slice(0, n).join(' ') + ' …' : (s || '');
};

// ---------------------------------------------------------------------------
// IDF over body text (tokenise/REF_HEADINGS/normHeading live in
// regex_utils.js and mirror heuristic.py so keywords agree across stages)
// ---------------------------------------------------------------------------

function bodyText(entry) {
  // normHeading strips leading numbering ('7. References') so numbered
  // bibliography headings don't leak reference text into the keyword pool.
  const sections = (entry?.sections || []).filter(
    (s) => !REF_HEADINGS.has(normHeading(s.heading)) && (s.text || '').trim()
  );
  return sections.length ? sections.map((s) => s.text).join(' ') : (entry?.text || '');
}

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

function clusterKeywords(memberTokens, idf, n = KEYWORDS_N) {
  const scores = new Map();
  for (const tokens of memberTokens) {
    for (const t of new Set(tokens)) scores.set(t, (scores.get(t) || 0) + idf(t));
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([t]) => t);
}

// ---------------------------------------------------------------------------
// Context assembly
// ---------------------------------------------------------------------------

// metadata.title / metadata.abstract are produced by GROBID's CRF header
// model at extraction time (extract.py) — clean, reliable fields. The
// fallbacks below only fire for docs extracted while GROBID was down.
function docTitle(entry) {
  return entry?.metadata?.title?.trim() || entry?.filename || 'untitled';
}

function docAbstract(entry, maxWords = 120) {
  let abs = entry?.metadata?.abstract?.trim();
  if (!abs) {
    const sec = (entry?.sections || []).find((s) => (s.text || '').trim().length > 200);
    abs = sec?.text?.trim() || '';
  }
  return capWords(abs, maxWords);
}

/**
 * Pick up to `max` {title, text} abstract entries for the context pack.
 * Docs with a real GROBID abstract are preferred over section-text
 * fallbacks, so the prompt is grounded in actual paper summaries whenever
 * the cluster has them.
 */
function abstractEntries(entries, max = 5) {
  const withReal = entries.filter((e) => e?.metadata?.abstract?.trim());
  const without  = entries.filter((e) => !e?.metadata?.abstract?.trim());
  return [...withReal, ...without]
    .slice(0, max)
    .map((e) => ({ title: docTitle(e), text: docAbstract(e) }))
    .filter((a) => a.text);
}

function distinctiveHeadings(entries, maxHeadings = 8) {
  const counts = new Map();
  for (const entry of entries) {
    for (const sec of entry?.sections || []) {
      const h = (sec.heading || '').trim();
      if (!h) continue;
      const key = h.toLowerCase().replace(/^[0-9.\s]+/, '');
      if (!key || BOILERPLATE_HEADINGS.has(key) || key.length < 4) continue;
      counts.set(h, (counts.get(h) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => (b[1] - a[1]) || (b[0].length - a[0].length))
    .slice(0, maxHeadings)
    .map(([h]) => h);
}

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
  return capWords(best.text, maxWords);
}

// ---------------------------------------------------------------------------
// Prompt building with staged truncation to fit the input budget
// ---------------------------------------------------------------------------

function renderPrompt(ctx, perCategory) {
  const titles    = ctx.titles.map((t) => `- ${t}`).join('\n');
  const abstracts = ctx.abstracts
    .map((a) => `"${a.title}":\n${a.text}`)
    .join('\n\n');
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
- Respond in JSON only. Return a single JSON object of this exact shape, with no other text:
{"queries": [{"query": "...", "type": "factual|definition|comparison|synthesis|methodology"}]}`;
}

/**
 * Build the prompt, then — if it overshoots the token budget — trim the
 * context pack in stages, cheapest information first:
 *   1. drop the medoid excerpt
 *   2. shrink abstracts (cap at 3 docs × 60 words, then 2 × 40)
 *   3. shrink headings→4, keywords→10, titles→10
 * Returns { prompt, trimmed } where trimmed lists the stages applied.
 */
function buildPromptWithinBudget(ctx, perCategory, budget = INPUT_BUDGET) {
  const trimmed = [];
  let working = { ...ctx };
  let prompt = renderPrompt(working, perCategory);
  if (estTokens(prompt) <= budget) return { prompt, trimmed };

  // Stage 1 — drop excerpt
  working = { ...working, excerpt: null };
  trimmed.push('excerpt');
  prompt = renderPrompt(working, perCategory);
  if (estTokens(prompt) <= budget) return { prompt, trimmed };

  // Stage 2 — shrink abstracts progressively
  for (const [count, words] of [[3, 60], [2, 40], [1, 30]]) {
    working = {
      ...working,
      abstracts: working.abstracts
        .slice(0, count)
        .map((a) => ({ ...a, text: capWords(a.text, words) })),
    };
    trimmed.push(`abstracts→${count}x${words}w`);
    prompt = renderPrompt(working, perCategory);
    if (estTokens(prompt) <= budget) return { prompt, trimmed };
  }

  // Stage 3 — shrink lists
  working = {
    ...working,
    headings: working.headings.slice(0, 4),
    keywords: working.keywords.slice(0, 10),
    titles:   working.titles.slice(0, 10),
  };
  trimmed.push('lists');
  prompt = renderPrompt(working, perCategory);
  return { prompt, trimmed }; // send even if still over budget
}

// ---------------------------------------------------------------------------
// Ollama generate call
// ---------------------------------------------------------------------------

async function ollamaGenerate(prompt) {
  const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model:   MODEL,
      system:  'You are a precise assistant that responds only with valid JSON.',
      prompt,
      stream:  false,
      options: { temperature: 0.8, num_predict: MAX_TOKENS },
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => '');
    throw new Error(`Ollama HTTP ${resp.status}: ${errText.slice(0, 300)}`);
  }

  return ((await resp.json()).response || '').trim();
}

// ---------------------------------------------------------------------------
// Output parsing + shape validation
// ---------------------------------------------------------------------------

function parseQueries(raw) {
  const text = (raw || '').trim();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    // Fallbacks: fenced block, then outermost object/array
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    const blob   = fenced?.[1] || text.match(/\{[\s\S]*\}|\[[\s\S]*\]/)?.[0];
    if (!blob) throw new Error('no JSON found in model output');
    parsed = JSON.parse(blob);
  }

  const arr = Array.isArray(parsed) ? parsed : parsed?.queries;
  if (!Array.isArray(arr)) throw new Error('output JSON has no "queries" array');

  const queries = arr
    .filter((q) => q && typeof q.query === 'string' && q.query.trim())
    .map((q) => ({
      query: q.query.trim(),
      type: QUERY_TYPES.includes(q.type) ? q.type : 'factual',
    }));
  if (queries.length === 0) throw new Error('queries array was empty or malformed');
  return queries;
}

async function generateQueries(prompt) {
  try {
    return parseQueries(await ollamaGenerate(prompt));
  } catch (err) {
    // One shape-retry: small models occasionally wrap or narrate; a stricter
    // nudge usually fixes it.
    console.warn(`[bootstrap]   parse failed (${err.message}) — retrying once with stricter instruction`);
    const strict = prompt + '\n\nIMPORTANT: Your previous attempt was invalid. Output ONLY the JSON object, starting with { and ending with }.';
    return parseQueries(await ollamaGenerate(strict));
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export async function bootstrapQueries({ perCategory = PER_CATEGORY } = {}) {
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

  const docIds = Object.keys(doclings);
  const tokenised = new Map(docIds.map((d) => [d, tokenise(bodyText(doclings[d]))]));
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
      abstracts: abstractEntries(entries),
      headings:  distinctiveHeadings(entries),
      excerpt:   medoidExcerpt(memberIds, embedStore),
    };

    const { prompt, trimmed } = buildPromptWithinBudget(ctx, perCategory);
    const trimNote = trimmed.length ? ` [trimmed: ${trimmed.join(', ')}]` : '';
    console.log(`[bootstrap] Category ${ci + 1}/${cats.length} (${memberIds.length} docs, ~${estTokens(prompt)} tok${trimNote}): ${ctx.keywords.slice(0, 5).join(', ')} ...`);

    let queries = [];
    try {
      queries = await generateQueries(prompt);
    } catch (err) {
      console.error(`[bootstrap]   generation failed: ${err.message} — skipping category`);
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
    model: MODEL,
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

// Run directly: node backend/extraction/bootstrap_queries.js --per-category 8
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const argv = process.argv;
  const flag = (name, fallback) => {
    const i = argv.indexOf(name);
    return i !== -1 ? argv[i + 1] : fallback;
  };
  bootstrapQueries({
    perCategory: parseInt(flag('--per-category', PER_CATEGORY), 10),
  }).catch((err) => {
    console.error('[bootstrap]', err.message);
    process.exit(1);
  });
}
