// Grounded multi-note SYNTHESIS with citations — "what do I think about X across
// everything I've written?". This is the app's synthesis core (vs. plain search).
//
// Why it isn't just "search + summarize": plain vector/FTS over all-MiniLM-L6-v2 is
// register-clustered — a philosophy query returns WORK notes, and the app's recency
// re-ranking makes it worse. So synthesis retrieval deliberately differs from search:
//   1. query expansion (LLM → paraphrases) + multi-probe union, no recency bias
//   2. true relevance = cosine(query-centroid, candidate-chunk embedding)
//   3. relevance gate to drop off-topic notes
//   4. MMR diversification so the evidence spans the topic, not 12 near-duplicates
//   5. synthesis prompt forces inline [n] citations; we post-check provenance
// Embeddings are injected so we reuse the server's already-loaded model.

type EmbedOne = (s: string) => Promise<number[]>;
type EmbedBatch = (s: string[]) => Promise<number[][]>;

export type Synthesis = {
  topic: string;
  queries: string[];
  synthesis: string;
  sources: { n: number; title: string; folder: string; modified: string; rel: number; excerpt: string }[];
  provenance: { used: number[]; hallucinated: number[]; uncited: number[] };
  lowConfidence: boolean;
  ms: number;
  cost: number;
};

const cos = (a: number[], b: number[]) => { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; };

// OpenAI-compatible chat. baseURL lets this target a LOCAL model (LM Studio
// http://localhost:1234/v1, Ollama http://localhost:11434/v1) — keeping notes
// fully private — or real OpenAI. Default: OpenAI.
async function chat(baseURL: string, apiKey: string, model: string, sys: string, usr: string, temp = 0.3) {
  const r = await fetch(`${baseURL.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, temperature: temp, messages: [{ role: "system", content: sys }, { role: "user", content: usr }] }),
  });
  const j: any = await r.json();
  if (!r.ok) throw new Error(j?.error?.message || `LLM HTTP ${r.status}`);
  return { text: j.choices[0].message.content as string, usage: j.usage };
}

export async function synthesize(opts: {
  table: any; topic: string; embedOne: EmbedOne; embedBatch: EmbedBatch;
  apiKey: string; baseURL?: string; model?: string; n?: number;
}): Promise<Synthesis> {
  const { table, topic, embedOne, embedBatch, apiKey } = opts;
  const baseURL = opts.baseURL ?? "https://api.openai.com/v1";
  const model = opts.model ?? "gpt-4o-mini";
  const N = opts.n ?? 12;
  const t0 = performance.now();

  // 1. query expansion → diverse retrieval probes
  const exp = await chat(baseURL, apiKey, model,
    "You expand a search topic into diverse retrieval queries for a personal-notes corpus. Output 5 short, varied paraphrases/related concepts, one per line, no numbering.",
    `TOPIC: ${topic}`, 0.5);
  const queries = [topic, ...exp.text.split("\n").map(s => s.replace(/^[-*\d.\s]+/, "").trim()).filter(Boolean)].slice(0, 6);

  // query centroid for the relevance gate
  const qVecs = await Promise.all(queries.map(embedOne));
  const dim = qVecs[0].length;
  const centroid = new Array(dim).fill(0);
  for (const v of qVecs) for (let i = 0; i < dim; i++) centroid[i] += v[i] / qVecs.length;

  // 2. multi-probe: union candidate notes across vector probes + FTS (no recency)
  const CAND = 40;
  const noteChunk = new Map<string, { title: string; content: string; folder: string; mod: string }>();
  const keep = (r: any) => {
    const cur = noteChunk.get(r.title);
    // keep the longest chunk per note as its representative excerpt
    if (!cur || (r.content?.length ?? 0) > cur.content.length)
      noteChunk.set(r.title, { title: r.title, content: r.content ?? "", folder: r.folder ?? "", mod: r.modification_date ?? "" });
  };
  for (const qv of qVecs) (await table.search(qv, "vector").limit(CAND).toArray()).forEach(keep);
  for (const q of queries) (await table.search(q, "fts", "content").limit(CAND).toArray().catch(() => [])).forEach(keep);

  // 3. true relevance: embed each candidate's representative text, cos vs centroid
  const cands = [...noteChunk.values()];
  if (!cands.length) {
    return { topic, queries, synthesis: "", sources: [], provenance: { used: [], hallucinated: [], uncited: [] }, lowConfidence: true, ms: Math.round(performance.now() - t0), cost: 0 };
  }
  const mat = await embedBatch(cands.map(c => (c.title + ". " + c.content).slice(0, 1200)));
  const scored = cands.map((c, i) => ({ c, rel: cos(centroid, mat[i]), vec: mat[i] })).sort((a, b) => b.rel - a.rel);

  // 4. relevance gate — relative, not a fixed cosine (MiniLM tops out ~0.3 on-topic).
  // Keep notes within 60% of the best score, floored at 0.15. If too few clear it,
  // fall back to the top candidates and flag low confidence.
  const top = scored[0].rel;
  const floor = Math.max(0.15, top * 0.6);
  let gated = scored.filter(s => s.rel >= floor);
  let lowConfidence = false;
  if (gated.length < 4) { gated = scored.slice(0, 8); lowConfidence = true; }

  // 5. MMR diversification → final evidence set
  const LAMBDA = 0.7, picked: typeof gated = [], pool = [...gated];
  while (picked.length < N && pool.length) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      let maxSim = 0; for (const p of picked) maxSim = Math.max(maxSim, cos(pool[i].vec, p.vec));
      const val = LAMBDA * pool[i].rel - (1 - LAMBDA) * maxSim;
      if (val > bv) { bv = val; bi = i; }
    }
    picked.push(pool.splice(bi, 1)[0]);
  }

  const evidence = picked.map((p, i) => ({ n: i + 1, title: p.c.title, folder: p.c.folder, modified: p.c.mod, rel: p.rel, excerpt: p.c.content.slice(0, 900) }));
  const sourcesTxt = evidence.map(e => `[${e.n}] "${e.title}" (${e.folder || "—"}; ${e.modified ? e.modified.slice(0, 10) : "?"})\n${e.excerpt}`).join("\n\n");

  const system = `You are a synthesis engine over a person's PRIVATE notes; write in second person ("you").
RULES:
- Use ONLY the numbered excerpts. No outside knowledge, no invented specifics.
- EVERY substantive claim ends with citation(s) like [3] or [3][7].
- If sources conflict or evolve over time, say so and cite both with dates.
- If the sources don't really address the topic, say so plainly instead of fabricating.
- 3-5 paragraphs. Surface the through-line, the tensions, and what's notably absent.`;
  const syn = await chat(baseURL, apiKey, model, system, `TOPIC: ${topic}\n\nSOURCE EXCERPTS:\n\n${sourcesTxt}\n\nWrite the grounded synthesis with inline [n] citations.`, 0.3);

  const cited = new Set([...syn.text.matchAll(/\[(\d+)\]/g)].map(x => Number(x[1])));
  const used = [...cited].filter(n => n >= 1 && n <= evidence.length).sort((a, b) => a - b);
  const hallucinated = [...cited].filter(n => n < 1 || n > evidence.length);
  const uncited = evidence.filter(e => !cited.has(e.n)).map(e => e.n);
  const totIn = (exp.usage?.prompt_tokens ?? 0) + (syn.usage?.prompt_tokens ?? 0);
  const totOut = (exp.usage?.completion_tokens ?? 0) + (syn.usage?.completion_tokens ?? 0);

  return {
    topic, queries: queries.slice(1),
    synthesis: syn.text,
    sources: evidence.map(({ excerpt, ...s }) => ({ ...s, excerpt: excerpt.slice(0, 280) })),
    provenance: { used, hallucinated, uncited },
    lowConfidence,
    ms: Math.round(performance.now() - t0),
    cost: +(totIn / 1e6 * 0.15 + totOut / 1e6 * 0.60).toFixed(5),
  };
}
