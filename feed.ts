// The FEED: one ranked, evidence-first stream mixing three item kinds —
//   bridge            A ↔ C via B, with the connecting phrase from B as evidence
//   abstraction_pair  two notes whose LLM-consolidated abstractions rhyme
//   entity_overlap    an entity that threads many notes across many folders
//
// Ranking is w·x through a sigmoid — a tiny online logistic regression, pure TS,
// no deps. Hand-tuned prior weights; every 👍/👎 takes a few SGD steps. Votes are
// appended to ~/.mcp-apple-notes/votes.jsonl (with the item's feature vector) and
// replayed on boot, so the ranking survives restarts.

import { Database } from "bun:sqlite";
import fs from "node:fs";
import type { Bridge } from "./bridges";

export type FeedItem = {
  id: string;
  kind: "bridge" | "abstraction_pair" | "entity_overlap";
  headline: string;
  evidence: string;            // plain-text WHY, always present
  notes: { title: string; folder: string }[];
  raw: number;                 // kind-native score (bridge det / abs sim / entity reach)
  features: number[];          // FEATURE_DIM-long input to the ranker
  score?: number;              // ranker output, filled at serve time
  voted?: 1 | -1;              // present if the user already voted on this item
};

export type Vote = { ts: string; id: string; kind: string; vote: 1 | -1; features: number[] };

// ── feature vector ────────────────────────────────────────────────────────────
// x = [bias, isBridge, isAbsPair, isEntity, kindScoreNorm, folderDiversity,
//      noteReach, evidenceRichness]
export const FEATURE_DIM = 8;
// Hand prior: bridges slightly ahead (rarest signal), abstraction pairs next,
// entity overlaps last (cheapest to produce); strong weight on the kind-native
// score so within-kind ordering starts sane.
export const DEFAULT_WEIGHTS = [-0.2, 0.7, 0.55, 0.15, 1.3, 0.45, 0.3, 0.25];

const dot = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const scoreItem = (w: number[], x: number[]) => sigmoid(dot(w, x));

// A few SGD steps per vote — enough to visibly move the ranking without
// letting one vote dominate. Deterministic (no shuffling).
export function sgdStep(w: number[], x: number[], y: 0 | 1, lr = 0.35, steps = 4) {
  for (let s = 0; s < steps; s++) {
    const g = scoreItem(w, x) - y;
    for (let i = 0; i < w.length; i++) w[i] -= lr * g * x[i];
  }
}

export function replayVotes(w: number[], votes: Vote[]) {
  for (const v of votes) {
    if (Array.isArray(v.features) && v.features.length === FEATURE_DIM)
      sgdStep(w, v.features, v.vote > 0 ? 1 : 0);
  }
}

export function loadVotes(file: string): Vote[] {
  try {
    return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l));
  } catch { return []; }
}

// ── evidence: TF-IDF sentence scoring (deterministic, no LLM) ────────────────
const tokenize = (s: string): string[] =>
  (s.toLowerCase().match(/[a-z][a-z0-9'-]{2,}/g) ?? []).filter(w => w.length > 2);

const splitSentences = (text: string): string[] =>
  text.split(/(?<=[.!?])\s+|\n+/).map(s => s.trim().replace(/^[#>*\-\s]+/, ""))
    .filter(s => s.length >= 25 && s.length <= 300);

// The "connecting phrase": the sentence in B whose vocabulary overlaps BOTH A
// and C the most (IDF-weighted) — the strongest deterministic one-liner for why
// B ties the two together.
export function connectingPhrase(bText: string, aText: string, cText: string, idf: (t: string) => number): string {
  const aSet = new Set(tokenize(aText));
  const cSet = new Set(tokenize(cText));
  let best = "", bestScore = 0;
  for (const sent of splitSentences(bText).slice(0, 120)) {
    const toks = new Set(tokenize(sent));
    let sa = 0, sc = 0;
    for (const t of toks) { if (aSet.has(t)) sa += idf(t); if (cSet.has(t)) sc += idf(t); }
    const score = sa * sc; // must connect to BOTH sides
    if (score > bestScore) { bestScore = score; best = sent; }
  }
  return best;
}

// ── building the items ───────────────────────────────────────────────────────
const STOPLIST = new Set(["i", "you", "me", "user", "world", "project", "time", "way"]);

const round = (x: number) => Math.round(x * 10000) / 10000;
const cap = (x: number) => Math.max(0, Math.min(1, x));

export function buildFeedItems(opts: {
  bridges: Bridge[];                       // pre-selected pool (hub-penalized)
  noteText: Map<string, string>;           // title → text, for bridge evidence
  consolidatedPath: string;                // jsonl {rel_path, claims, abstraction}
  layeredDbPath: string;                   // sqlite nodes/edges graph
}): FeedItem[] {
  const items: FeedItem[] = [];

  // Corpus IDF over note texts (for connecting-phrase scoring)
  const df = new Map<string, number>();
  for (const text of opts.noteText.values()) {
    for (const t of new Set(tokenize(text.slice(0, 4000)))) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const N = Math.max(1, opts.noteText.size);
  const idf = (t: string) => Math.log(N / (1 + (df.get(t) ?? 0)));

  // 1) bridges — evidence = connecting phrase mined from B's text
  for (const b of opts.bridges) {
    const phrase = connectingPhrase(
      opts.noteText.get(b.b.title) ?? "",
      opts.noteText.get(b.a.title) ?? b.a.title,
      opts.noteText.get(b.c.title) ?? b.c.title,
      idf,
    );
    const evidence = (phrase
      ? `“${phrase}” — the line in “${b.b.title}” that touches both sides.`
      : `“${b.b.title}” is strongly similar to both (${b.sims.ab} / ${b.sims.bc}) while they barely relate to each other (${b.sims.ac}).`)
      + ` sim A↔B ${b.sims.ab} · B↔C ${b.sims.bc} · A↔C ${b.sims.ac}`;
    const folders = new Set([b.a.folder, b.b.folder, b.c.folder].filter(Boolean));
    items.push({
      id: `bridge:${b.a.title}|${b.b.title}|${b.c.title}`,
      kind: "bridge",
      headline: `${b.a.title} ↔ ${b.c.title} · via ${b.b.title}`,
      evidence,
      notes: [b.a, b.b, b.c],
      raw: b.score,
      features: [1, 1, 0, 0, cap(b.score / 0.35), cap(folders.size / 3), cap(3 / 10), cap(evidence.length / 280)].map(round),
    });
  }

  // 2) abstraction pairs — graph edges kind='abstraction_pair' joined to the
  //    consolidated jsonl abstraction strings (the evidence IS the two strings)
  try {
    const absByPath = new Map<string, string>();
    for (const line of fs.readFileSync(opts.consolidatedPath, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try { const r = JSON.parse(line); if (r.rel_path && r.abstraction) absByPath.set(r.rel_path, r.abstraction); } catch {}
    }
    const db = new Database(opts.layeredDbPath, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        SELECT e.src, e.dst, e.weight,
               ns.label AS srcTitle, ns.folder AS srcFolder,
               nd.label AS dstTitle, nd.folder AS dstFolder
        FROM edges e JOIN nodes ns ON ns.id = e.src JOIN nodes nd ON nd.id = e.dst
        WHERE e.kind = 'abstraction_pair' ORDER BY e.weight DESC
      `).all();
      for (const r of rows) {
        const absA = absByPath.get(String(r.src).replace(/^note:/, ""));
        const absB = absByPath.get(String(r.dst).replace(/^note:/, ""));
        if (!absA || !absB) continue;
        const evidence = `A: ${absA}\nB: ${absB}`;
        const folders = new Set([r.srcFolder, r.dstFolder].filter(Boolean));
        items.push({
          id: `abs:${r.src}|${r.dst}`,
          kind: "abstraction_pair",
          headline: `${r.srcTitle} ↔ ${r.dstTitle} — same underlying move`,
          evidence,
          notes: [{ title: r.srcTitle, folder: r.srcFolder }, { title: r.dstTitle, folder: r.dstFolder }],
          raw: r.weight,
          features: [1, 0, 1, 0, cap(r.weight / 0.6), cap(folders.size / 3), cap(2 / 10), cap(evidence.length / 280)].map(round),
        });
      }
    } finally { db.close(); }
  } catch { /* consolidated.jsonl or graph db absent — feed degrades gracefully */ }

  // 3) entity overlaps — non-junk entities threading notes across folders
  try {
    const db = new Database(opts.layeredDbPath, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        SELECT n.label, n.folder AS type, COUNT(DISTINCT e.src) AS noteCount,
               COUNT(DISTINCT src_n.folder) AS folderCount
        FROM nodes n
        JOIN edges e ON e.dst = n.id AND e.kind = 'mentions'
        JOIN nodes src_n ON src_n.id = e.src
        WHERE n.kind = 'entity' AND LENGTH(n.label) > 3
        GROUP BY n.id
        HAVING noteCount >= 4 AND folderCount >= 2
        ORDER BY noteCount DESC LIMIT 60
      `).all();
      const sample = db.query<any, [string]>(`
        SELECT src_n.label AS title, src_n.folder AS folder
        FROM edges e JOIN nodes n ON n.id = e.dst JOIN nodes src_n ON src_n.id = e.src
        WHERE e.kind = 'mentions' AND n.kind = 'entity' AND n.label = ?
        ORDER BY e.weight DESC LIMIT 4
      `);
      for (const r of rows) {
        if (STOPLIST.has(String(r.label).toLowerCase())) continue;
        const notes = sample.all(r.label).map((s: any) => ({ title: s.title, folder: s.folder ?? "" }));
        const evidence = `${r.noteCount} notes across ${r.folderCount} folders mention “${r.label}” (${r.type})` +
          (notes.length ? ` — e.g. ${notes.map((n: any) => `“${n.title}”`).join(", ")}.` : ".");
        const reach = r.noteCount * Math.sqrt(r.folderCount);
        items.push({
          id: `ent:${r.label}`,
          kind: "entity_overlap",
          headline: `“${r.label}” threads ${r.noteCount} notes in ${r.folderCount} folders`,
          evidence,
          notes,
          raw: round(reach),
          features: [1, 0, 0, 1, cap(reach / 90), cap(r.folderCount / 8), cap(r.noteCount / 25), cap(evidence.length / 280)].map(round),
        });
      }
    } finally { db.close(); }
  } catch { /* graph db absent */ }

  return items;
}

// ── ranking + diversity arrangement ──────────────────────────────────────────
// Sort by ranker score, then greedily arrange with constraints:
//   · at most `maxRun` consecutive items of the same kind
//   · no note title appearing more than `maxNotePerPage` times per page
export function rankAndArrange(items: FeedItem[], weights: number[], opts: {
  pageSize?: number; maxRun?: number; maxNotePerPage?: number;
} = {}): FeedItem[] {
  const { pageSize = 20, maxRun = 2, maxNotePerPage = 2 } = opts;
  const sorted = items
    .map(it => ({ ...it, score: round(scoreItem(weights, it.features)) }))
    .sort((a, b) => (b.score! - a.score!) || a.id.localeCompare(b.id)); // deterministic ties

  const out: FeedItem[] = [];
  const remaining = [...sorted];
  let noteCounts = new Map<string, number>();
  while (remaining.length) {
    if (out.length % pageSize === 0) noteCounts = new Map(); // per-page constraint
    const runKind = out.length >= maxRun &&
      out.slice(-maxRun).every(x => x.kind === out[out.length - 1].kind)
      ? out[out.length - 1].kind : null;
    let pick = remaining.findIndex(it =>
      it.kind !== runKind &&
      it.notes.every(n => (noteCounts.get(n.title) ?? 0) < maxNotePerPage));
    if (pick < 0) pick = remaining.findIndex(it => it.kind !== runKind);
    if (pick < 0) pick = 0; // relax fully rather than drop items
    const [it] = remaining.splice(pick, 1);
    for (const n of it.notes) noteCounts.set(n.title, (noteCounts.get(n.title) ?? 0) + 1);
    out.push(it);
  }
  return out;
}
