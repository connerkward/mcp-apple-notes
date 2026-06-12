// Swanson-ABC "bridge" mining over the note embeddings already stored in LanceDB.
// A bridge is a triple (A, B, C): A and C are each strongly similar to a shared
// note B but NOT to each other, live in different folders, and share no tags or
// wikilinks — two notes the corpus only connects through an intermediary.
// Deterministic score sim(A,B)·sim(B,C)·(1−sim(A,C)); no LLM, $0.
//
// Note-level vectors are embedded on demand (title + first ~2000 chars) via the
// embedBatch the caller provides (same pattern as synthesize.ts), NOT averaged
// from the stored chunk vectors: the table's stored `vector` column embeds the
// wrong source field (same-folder notes have identical vectors — see report on
// the bridges branch), so it carries no per-note meaning. Mined candidates are
// cached in memory by rowCount; only the first call pays the embedding cost.

export type Bridge = {
  a: { title: string; folder: string };
  b: { title: string; folder: string };
  c: { title: string; folder: string };
  score: number;
  sims: { ab: number; bc: number; ac: number };
};

export type BridgeResult = {
  notes: number;
  candidates: number;
  tHigh: number;
  tLow: number;
  ms: number;
  bridges: Bridge[];
};

type MinedNote = { title: string; folder: string; titleLc: string; markers: Set<string> };
type Candidate = { det: number; a: number; b: number; c: number; ab: number; bc: number; ac: number };

let cache: { key: string; notes: number; tHigh: number; tLow: number; all: Bridge[] } | null = null;

// Same patterns as extractTags/extractWikilinks in index.ts (kept local so this
// module stays import-free of the entrypoint, like clustering.ts).
const TAG_RE = /#([A-Za-z][A-Za-z0-9_-]*)/g;
const WIKI_RE = /\[\[([^\]]+)\]\]/g;

const noteMarkers = (text: string): Set<string> => {
  const m = new Set<string>();
  for (const t of text.matchAll(TAG_RE)) m.add(t[1].toLowerCase());
  for (const w of text.matchAll(WIKI_RE)) m.add(w[1].split(/[|#]/)[0].trim().toLowerCase());
  return m;
};

const shareMarker = (a: Set<string>, b: Set<string>): boolean => {
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  for (const m of small) if (big.has(m)) return true;
  return false;
};

function normalize(v: number[]): number[] {
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
  return v.map(x => x / n);
}

// Triples (a, b, c): sims[b][a]>=tHigh, sims[b][c]>=tHigh, sims[a][c]<=tLow,
// folder(a)!=folder(c), no shared tags/wikilinks, no direct wikilink a<->c.
// Dedupe per unordered (a, c) pair keeping the best bridge b.
function mineCandidates(notes: MinedNote[], sims: Float32Array[], tHigh: number, tLow: number, maxNeighbors = 30): Candidate[] {
  const n = notes.length;
  const best = new Map<number, Candidate>();
  for (let b = 0; b < n; b++) {
    const row = sims[b];
    let nbr: number[] = [];
    for (let i = 0; i < n; i++) if (i !== b && row[i] >= tHigh) nbr.push(i);
    if (nbr.length < 2) continue;
    if (nbr.length > maxNeighbors) nbr = nbr.sort((x, y) => row[y] - row[x]).slice(0, maxNeighbors);
    for (let i = 0; i < nbr.length; i++) {
      const a = nbr[i];
      for (let j = i + 1; j < nbr.length; j++) {
        const c = nbr[j];
        const ac = sims[a][c];
        if (ac > tLow) continue;
        if (notes[a].folder === notes[c].folder) continue;
        if (shareMarker(notes[a].markers, notes[c].markers)) continue;
        if (notes[a].markers.has(notes[c].titleLc) || notes[c].markers.has(notes[a].titleLc)) continue;
        const det = row[a] * row[c] * (1 - ac);
        const key = a < c ? a * n + c : c * n + a;
        const prev = best.get(key);
        if (!prev || det > prev.det) best.set(key, { det, a, b, c, ab: row[a], bc: row[c], ac });
      }
    }
  }
  return [...best.values()].sort((x, y) => y.det - x.det);
}

// Relax/tighten thresholds until the candidate pool lands in [lo, hi].
function adaptThresholds(notes: MinedNote[], sims: Float32Array[], tHigh = 0.48, tLow = 0.23, lo = 200, hi = 2000, maxIter = 12) {
  for (let it = 0; it < maxIter; it++) {
    const cands = mineCandidates(notes, sims, tHigh, tLow);
    if (cands.length >= lo && cands.length <= hi) return { cands, tHigh, tLow };
    if (cands.length < lo) {
      tHigh = Math.max(0.30, tHigh - 0.03);
      tLow = Math.min(0.45, tLow + 0.02);
      if (tHigh === 0.30 && tLow === 0.45) break;
    } else {
      tHigh = Math.min(0.80, tHigh + 0.03);
      tLow = Math.max(0.05, tLow - 0.02);
    }
  }
  return { cands: mineCandidates(notes, sims, tHigh, tLow), tHigh, tLow };
}

async function mineAll(notesTable: any, embedBatch: (arr: string[]) => Promise<number[][]>) {
  // Pull every chunk, fold chunks → notes (first ~2000 chars of text), then
  // embed one vector per note. Markers (tags/wikilinks) come from the same text.
  const rows: any[] = await notesTable.query()
    .select(["title", "folder", "content"])
    .limit(200000).toArray();
  const byNote = new Map<string, { note: MinedNote; text: string }>();
  for (const r of rows) {
    const id = (r.title ?? "") + " " + (r.folder ?? "");
    let e = byNote.get(id);
    if (!e) {
      e = {
        note: { title: r.title ?? "", folder: r.folder ?? "", titleLc: (r.title ?? "").trim().toLowerCase(), markers: new Set() },
        text: "",
      };
      byNote.set(id, e);
    }
    if (e.text.length < 2000) e.text += " " + (r.content ?? "");
  }
  const notes: MinedNote[] = [];
  const texts: string[] = [];
  for (const e of byNote.values()) {
    e.note.markers = noteMarkers(e.note.title + " " + e.text);
    notes.push(e.note);
    texts.push((e.note.title + " " + e.text).slice(0, 2000));
  }
  if (notes.length < 3) return { notes: notes.length, tHigh: 0.48, tLow: 0.23, all: [] as Bridge[] };

  const vecs: number[][] = [];
  const BATCH = 32;
  for (let i = 0; i < texts.length; i += BATCH) {
    for (const v of await embedBatch(texts.slice(i, i + BATCH))) vecs.push(normalize(v));
    if (i % (BATCH * 8) === 0) console.error(`[bridges] embedded ${Math.min(i + BATCH, texts.length)}/${texts.length} notes`);
  }

  // Full cosine matrix (vectors normalized → dot = cosine). ~1.8k notes → ~13 MB.
  const n = notes.length;
  const sims: Float32Array[] = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    const vi = vecs[i]; const ri = sims[i];
    for (let j = i + 1; j < n; j++) {
      let s = 0; const vj = vecs[j];
      for (let d = 0; d < vi.length; d++) s += vi[d] * vj[d];
      ri[j] = s; sims[j][i] = s;
    }
  }

  const { cands, tHigh, tLow } = adaptThresholds(notes, sims);
  const round = (x: number) => Math.round(x * 10000) / 10000;
  const all: Bridge[] = cands.map(({ det, a, b, c, ab, bc, ac }) => ({
    a: { title: notes[a].title, folder: notes[a].folder },
    b: { title: notes[b].title, folder: notes[b].folder },
    c: { title: notes[c].title, folder: notes[c].folder },
    score: round(det),
    sims: { ab: round(ab), bc: round(bc), ac: round(ac) },
  }));
  return { notes: n, tHigh: round(tHigh), tLow: round(tLow), all };
}

export async function computeBridges({ table, embedBatch, limit = 20, folder }: {
  table: any;
  embedBatch: (arr: string[]) => Promise<number[][]>;
  limit?: number;
  folder?: string;
}): Promise<BridgeResult> {
  const t0 = performance.now();
  const total = await table.countRows().catch(() => 0);
  const key = String(total);
  if (!cache || cache.key !== key) cache = { key, ...(await mineAll(table, embedBatch)) };

  // Folder segment matcher — same semantics as searchAndCombineResults in index.ts
  const matchesFolder = folder
    ? (p: string) => p === folder || p.startsWith(folder + "/") || p.endsWith("/" + folder) || p.includes("/" + folder + "/")
    : null;
  const pool = matchesFolder
    ? cache.all.filter(b => matchesFolder(b.a.folder) || matchesFolder(b.c.folder))
    : cache.all;

  return {
    notes: cache.notes,
    candidates: cache.all.length,
    tHigh: cache.tHigh,
    tLow: cache.tLow,
    ms: Math.round(performance.now() - t0),
    bridges: selectWithHubPenalty(pool, limit),
  };
}

// Greedy hub-penalized selection (benchmark finding: top-by-raw-score lets one
// "hub" note appear as an endpoint in 4+ bridges, crowding out variety).
// Each round, pick the candidate maximizing det / sqrt(1 + appearances(A) + appearances(C)),
// and hard-cap any note at 2 appearances as an endpoint in the returned list.
export function selectWithHubPenalty(pool: Bridge[], limit: number, cap = 2): Bridge[] {
  const appearances = new Map<string, number>();
  const used = new Set<number>();
  const out: Bridge[] = [];
  while (out.length < limit) {
    let bestIdx = -1, bestScore = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      if (used.has(i)) continue;
      const b = pool[i];
      const aN = appearances.get(b.a.title) ?? 0;
      const cN = appearances.get(b.c.title) ?? 0;
      if (aN >= cap || cN >= cap) continue;
      const s = b.score / Math.sqrt(1 + aN + cN);
      if (s > bestScore) { bestScore = s; bestIdx = i; }
    }
    if (bestIdx < 0) break;
    used.add(bestIdx);
    const b = pool[bestIdx];
    appearances.set(b.a.title, (appearances.get(b.a.title) ?? 0) + 1);
    appearances.set(b.c.title, (appearances.get(b.c.title) ?? 0) + 1);
    out.push(b);
  }
  return out;
}
