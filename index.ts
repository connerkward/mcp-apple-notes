import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { Database } from "bun:sqlite";
import { gunzipSync } from "node:zlib";
import { execSync } from "node:child_process";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
import http from "node:http";
import { computeClusters, type ClusterResult } from "./clustering";
import { mineBridges, selectBridges, selectWithHubPenalty, type MinedBridges } from "./bridges";
import { synthesize } from "./synthesize";
import {
  buildFeedItems, rankAndArrange, loadVotes, sgdStep, replayVotes,
  DEFAULT_WEIGHTS, type FeedItem, type Vote,
} from "./feed";

// Entity layer (optional, zero deps — bun:sqlite). A sibling benchmark harness
// (exp-notes-indexing/layered_graph.py) writes ~/.mcp-apple-notes/layered_graph.db:
//   nodes(id, kind, label, folder)  — kinds: entity/note/folder/tag/theme;
//                                     for entity nodes, `folder` holds the entity TYPE
//   edges(src, dst, kind, weight)   — mentions edges are src=note → dst=entity
// If the file is absent the entity tools degrade to a helpful message.
const LAYERED_DB = process.env.LAYERED_DB || path.join(os.homedir(), ".mcp-apple-notes", "layered_graph.db");
const NO_GRAPH_DB_MSG = `No entity graph db at ${LAYERED_DB} — generate it with the exp-notes-indexing benchmark harness (layered_graph.py), or set LAYERED_DB.`;

const withLayeredDb = <T>(fn: (db: Database) => T): T | null => {
  if (!require("node:fs").existsSync(LAYERED_DB)) return null;
  const db = new Database(LAYERED_DB, { readonly: true });
  try { return fn(db); } finally { db.close(); }
};

// Resolve a user-supplied entity name → best matching entity node.
// Exact (case-insensitive) → prefix → contains; ties broken by mention count.
const resolveEntity = (db: Database, name: string): { id: string; label: string; type: string } | null => {
  const pick = (where: string) => db.query<any, [string]>(`
    SELECT n.id, n.label, n.folder AS type, COUNT(e.src) AS cnt
    FROM nodes n LEFT JOIN edges e ON e.dst = n.id AND e.kind = 'mentions'
    WHERE n.kind = 'entity' AND ${where}
    GROUP BY n.id ORDER BY cnt DESC LIMIT 1
  `).get(name);
  return pick("lower(n.label) = lower(?)")
      ?? pick("n.label LIKE ? || '%'")
      ?? pick("n.label LIKE '%' || ? || '%'")
      ?? null;
};

const entityNotes = (entity: string) => withLayeredDb((db) => {
  const ent = resolveEntity(db, entity);
  if (!ent) return { error: `No entity matching "${entity}" in the graph.` };
  const notes = db.query<any, [string]>(`
    SELECT n.label AS title, n.folder AS folder, ROUND(e.weight, 4) AS weight
    FROM edges e JOIN nodes n ON n.id = e.src
    WHERE e.kind = 'mentions' AND e.dst = ?
    ORDER BY e.weight DESC
  `).all(ent.id);
  return { entity: ent.label, type: ent.type, notes };
});

const listEntities = (query?: string, limit = 30) => withLayeredDb((db) => {
  const filter = query ? "AND n.label LIKE '%' || ? || '%'" : "";
  const params: any[] = query ? [query, limit] : [limit];
  const entities = db.query<any, any[]>(`
    SELECT n.label, n.folder AS type, COUNT(e.src) AS count
    FROM nodes n JOIN edges e ON e.dst = n.id AND e.kind = 'mentions'
    WHERE n.kind = 'entity' AND LENGTH(n.label) > 3 ${filter}
    GROUP BY n.id ORDER BY count DESC LIMIT ?
  `).all(...params);
  return { entities };
});

// LLM config for synthesis (optional). Read once; key is never logged.
// Works with a LOCAL OpenAI-compatible server (LM Studio / Ollama) or real OpenAI:
//   SYNTH_BASE_URL   default https://api.openai.com/v1
//                    LM Studio: http://localhost:1234/v1 · Ollama: http://localhost:11434/v1
//   SYNTH_MODEL      default gpt-4o-mini
//   OPENAI_API_KEY   from env (any non-empty string for local servers)
let _llm: { key: string; baseURL: string; model: string } | null | undefined;
const getLlmConfig = () => {
  if (_llm !== undefined) return _llm;
  const key = process.env.OPENAI_API_KEY?.trim() || "";
  _llm = key ? {
    key,
    baseURL: process.env.SYNTH_BASE_URL?.trim() || "https://api.openai.com/v1",
    model: process.env.SYNTH_MODEL?.trim() || "gpt-4o-mini",
  } : null;
  return _llm;
};
import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";

/**
 * IMPORTANT: Claude Desktop calls "list tools" during startup.
 * If we eagerly import heavy native deps (LanceDB) or load ML models here,
 * Claude will show "Loading tools" and appear hung.
 *
 * So: everything heavyweight is lazy-loaded.
 */
let dbPromise: Promise<any> | null = null;
let extractorPromise: Promise<any> | null = null;
let extractorIdleTimer: ReturnType<typeof setTimeout> | null = null;
const EXTRACTOR_IDLE_MS = 10 * 60 * 1000; // 10 minutes
let turndownPromise: Promise<(html: string) => string> | null = null;
let notesTableSchemaPromise: Promise<any> | null = null;

const getDb = async () => {
  if (!dbPromise) {
    dbPromise = (async () => {
      const lancedb = await import("@lancedb/lancedb");
      return await lancedb.connect(
        path.join(os.homedir(), ".mcp-apple-notes", "data")
      );
    })();
  }
  return await dbPromise;
};

const touchExtractor = () => {
  if (extractorIdleTimer) clearTimeout(extractorIdleTimer);
  extractorIdleTimer = setTimeout(() => {
    extractorPromise = null;
    extractorIdleTimer = null;
    console.error("Embedding model unloaded after 10m idle");
  }, EXTRACTOR_IDLE_MS);
};

const getExtractor = async () => {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline } = await import("@huggingface/transformers");
      return await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    })();
  }
  touchExtractor();
  return await extractorPromise;
};

const getTurndown = async () => {
  if (!turndownPromise) {
    turndownPromise = (async () => {
      const { default: TurndownService } = await import("turndown");
      const service = new TurndownService();
      return (html: string) => service.turndown(html);
    })();
  }
  return await turndownPromise;
};

const getNotesTableSchema = async () => {
  if (!notesTableSchemaPromise) {
    notesTableSchemaPromise = (async () => {
      const { EmbeddingFunction, LanceSchema, register } = await import(
        "@lancedb/lancedb/embedding"
      );
      const { Float32, Utf8 } = await import("apache-arrow");

      @register("openai")
      class OnDeviceEmbeddingFunction extends (EmbeddingFunction as any)<string> {
        toJSON(): object {
          return {};
        }
        ndims() {
          return 384;
        }
        embeddingDataType() {
          return new Float32();
        }
        async computeQueryEmbeddings(data: string) {
          const extractor = await getExtractor();
          const output = await extractor(data, { pooling: "mean", normalize: true });
          return Array.from(output.data) as number[];
        }
        async computeSourceEmbeddings(data: string[]) {
          const extractor = await getExtractor();
          // Pass the whole array — single ONNX forward pass instead of N sequential ones
          const output = await extractor(data, { pooling: "mean", normalize: true });
          return output.tolist() as number[][];
        }
      }

      const func = new OnDeviceEmbeddingFunction();
      // ONE sourceField only. Multiple sourceFields made LanceDB embed the LAST
      // one (folder) — every vector in a folder was identical, so the vector arm
      // of search/related-notes/clustering was folder-identity noise. content is
      // the only field that should drive the embedding; title relevance is
      // handled by the title FTS index + title boost at query time.
      return LanceSchema({
        title: new Utf8(),
        content: func.sourceField(new Utf8()),
        creation_date: new Utf8(),
        modification_date: new Utf8(),
        folder: new Utf8(),
        vector: func.vectorField(),
      });
    })();
  }
  return await notesTableSchemaPromise;
};

const QueryNotesSchema = z.object({
  query: z.string(),
  folder: z.string().optional().describe("Filter by folder name (exact path segment match). E.g. 'Notes' or 'ACADEMIA'."),
  modifiedAfter: z.string().optional().describe("ISO date — only notes modified after this date"),
  modifiedBefore: z.string().optional().describe("ISO date — only notes modified before this date"),
});

const GetNoteSchema = z.object({
  title: z.string().describe("Exact or partial note title. Falls back to semantic search if no exact match."),
});

const UpdateNoteSchema = z.object({
  title: z.string().describe("Exact title of the note to update"),
  content: z.string().describe("New full content for the note"),
});

const CreateNoteSchema = z.object({
  title: z.string(),
  content: z.string(),
});

const StartIndexNotesSchema = z.object({
  tableName: z.string().optional(),
});

const JobIdSchema = z.object({
  jobId: z.string(),
});

const JobLogsSchema = z.object({
  jobId: z.string(),
  offset: z.number().int().min(0).optional(),
});

type IndexJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  total: number;
  message?: string;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
  cancelled?: boolean;
  logs: string[];
};

const indexJobs = new Map<string, IndexJob>();

// LanceDB applies a DEFAULT LIMIT (10) to plain query() scans on tables opened
// through createEmptyTable — an unlimited-looking scan silently returns 10 rows.
// This was the index storm's root cause: the incremental indexer "saw" only 10
// existing rows, judged the whole corpus new, and re-embedded everything on
// every search. EVERY full scan must set an explicit limit.
const FULL_SCAN_LIMIT = 200000;

const DATA_DIR = path.join(os.homedir(), ".mcp-apple-notes");
const INDEX_STATE_FILE = path.join(DATA_DIR, "index_state.json");

// Track the max modification_date at the time of last successful index.
// Persisted to disk so a fresh boot doesn't treat the whole corpus as changed.
let lastIndexedModDate: number | null = (() => {
  try {
    return JSON.parse(require("node:fs").readFileSync(INDEX_STATE_FILE, "utf8")).lastIndexedModDate ?? null;
  } catch { return null; }
})();

const markIndexed = () => {
  lastIndexedModDate = getNotesMaxModDate();
  try {
    require("node:fs").mkdirSync(DATA_DIR, { recursive: true });
    require("node:fs").writeFileSync(INDEX_STATE_FILE, JSON.stringify({ lastIndexedModDate }));
  } catch { /* best-effort */ }
};

const getNotesMaxModDate = (): number | null => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<{ d: number | null }, []>(
        `SELECT MAX(ZMODIFICATIONDATE1) AS d FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = ${noteEnt(db)} AND ZMARKEDFORDELETION = 0`
      ).get();
      return row?.d ?? null;
    } finally { db.close(); }
  } catch { return null; }
};

const isActiveIndexJob = () =>
  [...indexJobs.values()].some((j) => j.status === "running" || j.status === "queued");

// Call before search: if notes have changed, kick ONE background re-index and
// return immediately. Search must NEVER block on bulk re-indexing — it serves
// from the current index; results catch up once the background job finishes.
// Single-flight: duplicate triggers (rapid concurrent searches) are dropped.
let reindexInFlight = false;
const kickReindexIfNeeded = () => {
  const maxMod = getNotesMaxModDate();
  if (maxMod === null) return;
  if (lastIndexedModDate !== null && maxMod <= lastIndexedModDate) return;
  if (reindexInFlight || isActiveIndexJob()) return;
  reindexInFlight = true;
  void (async () => {
    try {
      const { notesTable } = await createNotesTable();
      await indexNotes(notesTable);
      markIndexed();
    } catch (err) {
      console.error("Background re-index failed:", err);
    } finally {
      reindexInFlight = false;
    }
  })();
};

// ── precompute-once disk caches (bridges, clusters) ──────────────────────────
// Keyed by a corpus fingerprint (note count + max modification date). Fresh →
// serve from disk instantly. Stale → serve the stale copy instantly AND kick a
// background recompute (single-flight per entry). Absent → compute inline once.
const BRIDGES_CACHE_FILE = path.join(DATA_DIR, "bridges_cache.json");
const CLUSTERS_CACHE_FILE = path.join(DATA_DIR, "clusters_cache.json");
const VOTES_FILE = path.join(DATA_DIR, "votes.jsonl");
const CONSOLIDATED_FILE = path.join(DATA_DIR, "consolidated.jsonl");

const fsSync = require("node:fs") as typeof import("node:fs");

const corpusFingerprint = (): string => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<{ n: number; m: number | null }, []>(
        `SELECT COUNT(*) AS n, MAX(ZMODIFICATIONDATE1) AS m FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = ${noteEnt(db)} AND ZMARKEDFORDELETION = 0`
      ).get();
      return `${row?.n ?? 0}:${row?.m ?? 0}`;
    } finally { db.close(); }
  } catch { return "unknown"; }
};

type CacheFile = { fingerprint: string; computedAt: string; entries: Record<string, any> };
const readCacheFile = (file: string): CacheFile | null => {
  try { return JSON.parse(fsSync.readFileSync(file, "utf8")); } catch { return null; }
};
const writeCacheEntry = (file: string, fingerprint: string, subKey: string, result: any) => {
  try {
    const cur = readCacheFile(file);
    const entries = cur && cur.fingerprint === fingerprint ? cur.entries : {};
    entries[subKey] = result;
    fsSync.mkdirSync(DATA_DIR, { recursive: true });
    fsSync.writeFileSync(file, JSON.stringify({ fingerprint, computedAt: new Date().toISOString(), entries }));
  } catch (e) { console.error(`cache write ${file} failed:`, e); }
};

const recomputing = new Set<string>(); // single-flight keys: `${file}#${subKey}`

async function cachedCompute<T>(
  file: string,
  subKey: string,
  compute: () => Promise<T>
): Promise<{ result: T; stale: boolean; recomputing: boolean }> {
  const fp = corpusFingerprint();
  const flightKey = `${file}#${subKey}`;
  const cached = readCacheFile(file);
  const hasEntry = cached?.entries?.[subKey] !== undefined;

  if (cached && hasEntry && cached.fingerprint === fp) {
    return { result: cached.entries[subKey], stale: false, recomputing: recomputing.has(flightKey) };
  }
  if (cached && hasEntry) {
    // Stale: serve instantly, recompute in background (drop duplicate triggers)
    if (!recomputing.has(flightKey)) {
      recomputing.add(flightKey);
      void (async () => {
        try {
          const startFp = corpusFingerprint();
          writeCacheEntry(file, startFp, subKey, await compute());
        } catch (e) { console.error(`background recompute ${flightKey} failed:`, e); }
        finally { recomputing.delete(flightKey); }
      })();
    }
    return { result: cached.entries[subKey], stale: true, recomputing: true };
  }
  // First ever: compute inline, persist
  const result = await compute();
  writeCacheEntry(file, fp, subKey, result);
  return { result, stale: false, recomputing: false };
}

const getMinedBridges = () => cachedCompute<MinedBridges>(BRIDGES_CACHE_FILE, "mined", async () => {
  const { notesTable } = await createNotesTable();
  const extractor = await getExtractor();
  const embedBatch = async (arr: string[]) => (await extractor(arr, { pooling: "mean", normalize: true })).tolist() as number[][];
  return await mineBridges(notesTable, embedBatch);
});

const getClusters = (k: number) => cachedCompute<ClusterResult>(CLUSTERS_CACHE_FILE, `k${k}`, async () => {
  const { notesTable } = await createNotesTable();
  return await computeClusters(notesTable, k);
});

// ── the FEED: ranked evidence-first stream + online logistic ranker ──────────
const feedWeights: number[] = [...DEFAULT_WEIGHTS];
const votedMap = new Map<string, 1 | -1>(); // itemId → last vote
{ // votes persist and re-apply on boot
  const votes = loadVotes(VOTES_FILE);
  for (const v of votes) votedMap.set(v.id, v.vote);
  replayVotes(feedWeights, votes);
}

let feedItemsCache: { fp: string; items: FeedItem[] } | null = null;

async function getFeedItems(): Promise<FeedItem[]> {
  const fp = corpusFingerprint();
  if (feedItemsCache?.fp === fp) return feedItemsCache.items;

  // Bridges come from the disk cache; if never mined, don't block the feed for
  // ~2 min — serve abstraction pairs + entity overlaps now, kick the mine, and
  // bridges appear on a later load.
  let bridgePool: MinedBridges | null = null;
  const cached = readCacheFile(BRIDGES_CACHE_FILE);
  if (cached?.entries?.mined !== undefined) {
    bridgePool = cached.entries.mined;
    if (cached.fingerprint !== fp) void getMinedBridges(); // background refresh
  } else {
    void getMinedBridges(); // first-ever mine in background
  }

  // title → text map for bridge evidence (one table scan)
  const noteText = new Map<string, string>();
  if (bridgePool) {
    try {
      const { notesTable } = await createNotesTable();
      const rows: any[] = await notesTable.query().select(["title", "content"]).limit(200000).toArray();
      for (const r of rows) {
        const cur = noteText.get(r.title) ?? "";
        if (cur.length < 4000) noteText.set(r.title, cur + " " + (r.content ?? ""));
      }
    } catch { /* feed degrades to no bridge evidence */ }
  }

  const items = buildFeedItems({
    bridges: bridgePool ? selectWithHubPenalty(bridgePool.all, 60) : [],
    noteText,
    consolidatedPath: CONSOLIDATED_FILE,
    layeredDbPath: LAYERED_DB,
  });
  feedItemsCache = { fp, items };
  return items;
}

async function serveFeed(offset: number, limit: number) {
  const items = await getFeedItems();
  const arranged = rankAndArrange(items, feedWeights);
  const page = arranged.slice(offset, offset + limit).map(it => ({
    ...it,
    voted: votedMap.get(it.id),
  }));
  return { items: page, total: arranged.length, offset, limit };
}

function recordVote(id: string, vote: 1 | -1): { ok: true } | { error: string } {
  const item = feedItemsCache?.items.find(it => it.id === id);
  if (!item) return { error: `unknown feed item id: ${id}` };
  const v: Vote = { ts: new Date().toISOString(), id, kind: item.kind, vote, features: item.features };
  try {
    fsSync.mkdirSync(DATA_DIR, { recursive: true });
    fsSync.appendFileSync(VOTES_FILE, JSON.stringify(v) + "\n");
  } catch (e) { return { error: String(e) }; }
  sgdStep(feedWeights, item.features, vote > 0 ? 1 : 0);
  votedMap.set(id, vote);
  return { ok: true };
}

const INDEXER_RESOURCE_URI = "ui://apple-notes/indexer.html";

function createServer(): McpServer {
const server = new McpServer({
  name: "my-apple-notes-mcp",
  version: "1.0.0",
});

registerAppResource(
  server,
  INDEXER_RESOURCE_URI,
  INDEXER_RESOURCE_URI,
  { mimeType: RESOURCE_MIME_TYPE },
  async () => {
    const htmlPath = path.join(import.meta.dirname, "ui-dist", "ui", "indexer.html");
    let html: string;
    try {
      html = await fs.readFile(htmlPath, "utf-8");
    } catch {
      html = `<html><body><pre style="font:12px ui-monospace;padding:16px">Missing ui-dist/indexer.html. Run: bun run build:ui</pre></body></html>`;
    }
    return {
      contents: [
        { uri: INDEXER_RESOURCE_URI, mimeType: RESOURCE_MIME_TYPE, text: html },
      ],
    };
  }
);

registerTools(server);
return server;
} // end createServer

const NOTES_DB = path.join(
  os.homedir(),
  "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"
);

// CoreData timestamps: seconds since 2001-01-01
const CF_EPOCH = 978307200;
const cfDate = (cf: number) => new Date((cf + CF_EPOCH) * 1000).toISOString();

// Z_ENT entity numbers are NOT stable across Notes.app/macOS schema versions
// (ICNote was 11 on older builds, 12 on newer; ICFolder 14 → 15). They depend on
// the order entities were added to the Core Data model. Resolve them from
// Z_PRIMARYKEY at runtime — hardcoding silently matches the wrong entity
// (ICMedia/ICAccount on this build) and returns zero notes.
export function entId(db: Database, name: string): number {
  const row = db
    .query<{ e: number }, [string]>("SELECT Z_ENT AS e FROM Z_PRIMARYKEY WHERE Z_NAME = ?")
    .get(name);
  if (!row) throw new Error(`Entity "${name}" not found in Z_PRIMARYKEY`);
  return row.e;
}
export const noteEnt = (db: Database) => entId(db, "ICNote");
export const folderEnt = (db: Database) => entId(db, "ICFolder");

// Safe escaping for LanceDB filter strings (Arrow SQL syntax)
export function escapeForFilter(s: string): string {
  return s.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

export function calcEntropy(s: string): number {
  const freq = new Map<string, number>();
  for (const c of s) freq.set(c, (freq.get(c) ?? 0) + 1);
  let e = 0;
  for (const n of freq.values()) { const p = n / s.length; e -= p * Math.log2(p); }
  return e;
}

export function filterContent(text: string): string {
  return text
    .replace(/[A-Za-z0-9+/]{60,}={0,2}/g, (m) => calcEntropy(m) > 4.5 ? "" : m)
    .replace(/\b(AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{36}|-----BEGIN [A-Z ]+ KEY-----[\s\S]*?-----END [A-Z ]+ KEY-----)\b/g, "[redacted]");
}

const HASHTAG_RE = /#([A-Za-z][A-Za-z0-9_-]*)/g;
const WIKILINK_RE = /\[\[([^\]]+)\]\]/g;

export function extractTags(text: string): string[] {
  return [...text.matchAll(HASHTAG_RE)].map(m => m[1].toLowerCase());
}

export function extractWikilinks(text: string): string[] {
  return [...text.matchAll(WIKILINK_RE)].map(m => m[1]);
}

export function extractTablesFromText(text: string): string[][][] {
  const tables: string[][][] = [];
  const lines = text.split("\n");
  let cur: string[][] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      const cells = trimmed.split("|").slice(1, -1).map(c => c.trim());
      if (!cells.every(c => /^[-: ]+$/.test(c))) cur.push(cells); // skip separator rows
    } else if (line.includes("\t")) {
      const cells = line.split("\t").map(c => c.trim());
      if (cells.length >= 2) { cur.push(cells); continue; }
      if (cur.length >= 2) { tables.push(cur); } cur = [];
    } else {
      if (cur.length >= 2) tables.push(cur);
      cur = [];
    }
  }
  if (cur.length >= 2) tables.push(cur);
  return tables;
}

// Minimal inline protobuf decoder — no external library.
// Apple Notes ZDATA layout (reverse-engineered): outer.field2 → middle.field3 → field2 = plain text.
function readVarint(buf: Uint8Array, pos: number): [bigint, number] {
  let result = 0n, shift = 0n;
  while (pos < buf.length) {
    const b = buf[pos++];
    result |= BigInt(b & 0x7f) << shift;
    shift += 7n;
    if (!(b & 0x80)) break;
  }
  return [result, pos];
}

function walkProto(buf: Uint8Array, cb: (field: number, wire: number, val: bigint | Uint8Array) => void) {
  let pos = 0;
  while (pos < buf.length) {
    let tag: bigint;
    [tag, pos] = readVarint(buf, pos);
    const field = Number(tag >> 3n), wire = Number(tag & 7n);
    if (wire === 0) { let v: bigint; [v, pos] = readVarint(buf, pos); cb(field, wire, v); }
    else if (wire === 1) { pos += 8; }
    else if (wire === 2) { let len: bigint; [len, pos] = readVarint(buf, pos); cb(field, wire, buf.slice(pos, pos + Number(len))); pos += Number(len); }
    else if (wire === 5) { pos += 4; }
    else break;
  }
}

function extractText(data: Uint8Array): string {
  try {
    const buf = gunzipSync(Buffer.from(data));
    let text = "";
    walkProto(buf, (f1, w1, v1) => {
      if (f1 === 2 && w1 === 2) walkProto(v1 as Uint8Array, (f2, w2, v2) => {
        if (f2 === 3 && w2 === 2) walkProto(v2 as Uint8Array, (f3, w3, v3) => {
          if (f3 === 2 && w3 === 2 && !text) text = Buffer.from(v3 as Uint8Array).toString("utf8");
        });
      });
    });
    return text;
  } catch { return ""; }
}

function openFDASettings() {
  try {
    execSync(`open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"`);
  } catch { /* best-effort */ }
}

function handleDbError(err: unknown): never {
  const msg = String(err instanceof Error ? err.message : err);
  if (msg.includes("authorization denied") || msg.includes("unable to open")) {
    openFDASettings();
    throw new Error(
      "Full Disk Access required. System Settings just opened — add Claude.app (or bun) under Full Disk Access, then restart Claude Desktop."
    );
  }
  throw err;
}

export const folderCte = (db: Database) => {
  const fe = folderEnt(db);
  return `
  WITH RECURSIVE folder_path(id, path) AS (
    SELECT Z_PK, ZTITLE2
    FROM ZICCLOUDSYNCINGOBJECT
    WHERE Z_ENT = ${fe} AND ZPARENT IS NULL AND ZTITLE2 IS NOT NULL
    UNION ALL
    SELECT f.Z_PK, fp.path || '/' || f.ZTITLE2
    FROM ZICCLOUDSYNCINGOBJECT f
    JOIN folder_path fp ON f.ZPARENT = fp.id
    WHERE f.Z_ENT = ${fe} AND f.ZTITLE2 IS NOT NULL
  )
`;
};

const getAllNoteDetails = async () => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        ${folderCte(db)}
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0 AND n.Z_ENT = ${noteEnt(db)}
      `).all();
      return rows.map((r: any) => ({
        title: r.title as string,
        content: r.data ? extractText(r.data) : "",
        creation_date: r.created ? cfDate(r.created) : "",
        modification_date: r.modified ? cfDate(r.modified) : "",
        folder: r.folder as string,
      }));
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
};

const getNoteDetailsByTitle = async (title: string) => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<any, [string]>(`
        ${folderCte(db)}
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.ZTITLE1 = ? AND n.ZMARKEDFORDELETION = 0 AND n.Z_ENT = ${noteEnt(db)}
        LIMIT 1
      `).get(title);
      if (!row) return {} as any;
      return {
        title: row.title as string,
        content: row.data ? extractText(row.data) : "",
        creation_date: row.created ? cfDate(row.created) : "",
        modification_date: row.modified ? cfDate(row.modified) : "",
        folder: row.folder as string,
      };
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
};

const maybeSendProgress = async (
  extra: any | undefined,
  progress: number,
  total: number,
  message?: string
) => {
  const token = extra?._meta?.progressToken;
  if (token === undefined) return;
  await extra.sendNotification({
    method: "notifications/progress",
    params: {
      progressToken: token,
      progress,
      total,
      message,
    },
  });
};

const maybeSendMessage = async (extra: any | undefined, message: string) => {
  if (!extra?.sendNotification) return;
  await extra.sendNotification({
    method: "notifications/message",
    params: {
      level: "info",
      data: message,
    },
  });
};

const fmtDuration = (ms: number) => {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
};

const appendJobLog = (job: IndexJob, line: string) => {
  job.logs.push(line);
  // keep memory bounded
  if (job.logs.length > 2000) job.logs.splice(0, job.logs.length - 2000);
};

const updateJob = (job: IndexJob, patch: Partial<IndexJob>) => {
  Object.assign(job, patch);
};

export const indexNotes = async (
  notesTable: any,
  extra?: any,
  job?: IndexJob
) => {
  const start = performance.now();

  // 1. Fetch all notes from Apple Notes in one JXA call
  if (job) appendJobLog(job, "Fetching notes from Apple Notes…");
  const allNotes = await getAllNoteDetails();

  // 2. Query existing index to find what's already embedded.
  // Titles are NOT unique in Apple Notes ("TODO" ×10, "New Note" ×16, …), so a
  // title → single-mod-date map can never converge: for a duplicated title at
  // least one of the notes always compares unequal, which re-embedded the same
  // ~90 notes on EVERY change check, forever (the index storm). Compare the
  // per-title SET of modification dates instead — a title is dirty iff the set
  // in the index differs from the set in Apple Notes.
  const existingDates = new Map<string, Set<string>>(); // title → set of modification_dates
  try {
    const rows = await notesTable.query().select(["title", "modification_date"]).limit(FULL_SCAN_LIMIT).toArray();
    for (const row of rows) {
      let s = existingDates.get(row.title);
      if (!s) existingDates.set(row.title, (s = new Set()));
      s.add(row.modification_date);
    }
  } catch { /* empty table */ }

  const appleByTitle = new Map<string, typeof allNotes>();
  for (const n of allNotes) {
    if (!n.title) continue;
    const arr = appleByTitle.get(n.title);
    if (arr) arr.push(n); else appleByTitle.set(n.title, [n]);
  }

  // 3. Remove notes deleted from Apple Notes
  const deletedTitles = [...existingDates.keys()].filter((t) => !appleByTitle.has(t));
  for (const title of deletedTitles) {
    await notesTable.delete(`title = '${escapeForFilter(title)}'`);
  }

  // 4. Find dirty titles (new, modified, or duplicate-set changed)
  const setsEqual = (a: Set<string>, b: Set<string>) =>
    a.size === b.size && [...a].every((x) => b.has(x));
  const dirtyTitles: string[] = [];
  for (const [title, notes] of appleByTitle) {
    const want = new Set(notes.map((n) => n.modification_date));
    const have = existingDates.get(title);
    if (!have || !setsEqual(want, have)) dirtyTitles.push(title);
  }
  const toIndex = dirtyTitles.flatMap((t) => appleByTitle.get(t)!);
  const skipped = allNotes.length - toIndex.length;

  const summary = [
    toIndex.length > 0 ? `${toIndex.length} to embed` : null,
    skipped > 0 ? `${skipped} unchanged` : null,
    deletedTitles.length > 0 ? `${deletedTitles.length} removed` : null,
  ].filter(Boolean).join(", ");

  if (job) appendJobLog(job, summary || "Nothing to do.");

  if (toIndex.length === 0) {
    if (job) updateJob(job, { progress: 1, total: 1, message: "All notes up to date." });
    await maybeSendProgress(extra, 1, 1, "All notes up to date.");
    return { chunks: 0, report: "", allNotes: allNotes.length, time: performance.now() - start };
  }

  // 5. Delete stale versions of dirty titles before re-embedding (once per
  // title — duplicated titles share rows, so all their notes are re-added below)
  for (const title of dirtyTitles) {
    if (existingDates.has(title)) {
      await notesTable.delete(`title = '${escapeForFilter(title)}'`);
    }
  }

  const overallTotal = toIndex.length;
  if (job) updateJob(job, { progress: 0, total: overallTotal, message: `Embedding ${toIndex.length} notes…` });
  await maybeSendProgress(extra, 0, overallTotal, `Embedding ${toIndex.length} notes…`);
  await maybeSendMessage(extra, `Embedding ${toIndex.length} notes…`);

  // 6. Build chunks: convert HTML → Markdown, then split large notes
  const CHUNK_SIZE = 1500; // chars (~300 tokens)
  const CHUNK_OVERLAP = 150;
  // NOTE: chunk objects must match the Lance schema exactly (no extra `id`
  // field — the table was created without one, and add() rejects extras).
  const chunks: Array<{ title: string; content: string; creation_date: string; modification_date: string; folder: string }> = [];

  let td: ((html: string) => string) | null = null;
  try { td = await getTurndown(); } catch { /* ignore */ }

  const { RecursiveCharacterTextSplitter } = await import("@langchain/textsplitters");
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP });

  for (const note of toIndex) {
    let text = note.content ?? "";
    if (td) { try { text = td(text); } catch { /* keep original */ } }
    text = filterContent(text);
    // Normalize "AR-15" → "AR15": removes hyphens between letter-digit pairs so FTS
    // indexes as a single token, matching queries like "ar15" directly.
    text = text.replace(/\b([A-Za-z]+)-(\d+)\b/g, "$1$2").replace(/\b(\d+)-([A-Za-z]+)\b/g, "$1$2");

    const splits = text.length > CHUNK_SIZE
      ? await splitter.splitText(text)
      : [text];

    splits.forEach((chunk) => {
      chunks.push({
        title: note.title,
        content: chunk,
        creation_date: note.creation_date,
        modification_date: note.modification_date,
        folder: note.folder ?? "",
      });
    });
  }

  // 7. Embed and write in batches, reporting ETA after first batch
  // Larger batches = more efficient ONNX matrix ops; scale with CPU count
  const batchSize = Math.max(8, Math.min(64, os.cpus().length * 4));
  let written = 0;
  const embedStart = performance.now();

  for (let i = 0; i < chunks.length; i += batchSize) {
    if (job?.cancelled) {
      appendJobLog(job, "Cancelled.");
      throw new Error("Indexing cancelled");
    }
    const batch = chunks.slice(i, i + batchSize);
    await notesTable.add(batch);
    written += batch.length;

    const elapsed = performance.now() - embedStart;
    const remaining = (chunks.length - written) * (elapsed / written);
    const etaStr = remaining > 1000 ? ` — ~${fmtDuration(remaining)} remaining` : "";
    const msg = `Embedded ${written}/${chunks.length} notes${etaStr}`;

    await maybeSendProgress(extra, written, overallTotal, msg);
    if (job) {
      updateJob(job, { progress: written, total: overallTotal, message: msg });
      if (written === chunks.length || written % (batchSize * 4) === 0) appendJobLog(job, msg);
    }
  }

  const msg = `Done. Indexed ${chunks.length} notes in ${fmtDuration(performance.now() - start)}.`;
  await maybeSendProgress(extra, overallTotal, overallTotal, msg);
  await maybeSendMessage(extra, msg);
  if (job) {
    updateJob(job, { progress: overallTotal, total: overallTotal, message: msg });
    appendJobLog(job, msg);
  }

  return { chunks: chunks.length, report: "", allNotes: allNotes.length, time: performance.now() - start };
};

const purgeDb = async () => {
  dbPromise = null;
  // Only the Lance index + its watermark — NOT the whole ~/.mcp-apple-notes dir,
  // which also holds votes.jsonl, consolidated.jsonl, layered_graph.db, caches.
  await fs.rm(path.join(DATA_DIR, "data"), { recursive: true, force: true });
  await fs.rm(INDEX_STATE_FILE, { force: true });
};

const createNotesTableInner = async (overrideName?: string) => {
  const db = await getDb();
  const notesTableSchema = await getNotesTableSchema();
  const notesTable = await db.createEmptyTable(
    overrideName || "notes",
    notesTableSchema,
    { mode: "create", existOk: true }
  );
  // The cached connection pins tables to the version it knew at open — without
  // this, a long-lived process reads STALE data after its own writes (verified:
  // a second indexNotes pass in one process re-embedded the entire corpus and
  // duplicated every row, because it couldn't see the first pass's commits).
  if (typeof notesTable.checkoutLatest === "function") await notesTable.checkoutLatest();
  const indices = await notesTable.listIndices();
  const lancedb = await import("@lancedb/lancedb");
  if (!indices.find((index: any) => index.name === "content_idx")) {
    await notesTable.createIndex("content", { config: lancedb.Index.fts(), replace: true });
  }
  if (!indices.find((index: any) => index.name === "title_idx")) {
    await notesTable.createIndex("title", { config: lancedb.Index.fts(), replace: true });
  }
  return notesTable;
};

export const createNotesTable = async (overrideName?: string) => {
  const start = performance.now();
  try {
    const notesTable = await createNotesTableInner(overrideName);
    return { notesTable, time: performance.now() - start };
  } catch (err: any) {
    const msg = String(err?.message ?? err);
    // Corrupted index files — purge and rebuild from scratch
    if (msg.includes("Not found") && (msg.includes(".lance") || msg.includes("tokens.lance"))) {
      await purgeDb();
      const notesTable = await createNotesTableInner(overrideName);
      return { notesTable, time: performance.now() - start };
    }
    throw err;
  }
};

const createNote = async (title: string, content: string) => {
  // osascript has its own bundle ID so the Automation dialog fires correctly
  const script = `
    const app = Application('Notes');
    app.make({new: 'note', withProperties: {
      name: ${JSON.stringify(title)},
      body: ${JSON.stringify(content)}
    }});
  `;
  execSync(`osascript -l JavaScript -e ${JSON.stringify(script)}`);
  return true;
};

const createTextResponse = (text: string) => ({ content: [{ type: "text", text }] });

const startIndexJob = async (notesTable: any) => {
  const jobId = `${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
  const job: IndexJob = {
    id: jobId,
    status: "queued",
    progress: 0,
    total: 0,
    logs: [],
  };
  indexJobs.set(jobId, job);
  updateJob(job, { status: "running", startedAt: Date.now() });
  appendJobLog(job, `Job ${jobId} started.`);

  void (async () => {
    try {
      const result = await indexNotes(notesTable, undefined, job);
      markIndexed();
      updateJob(job, {
        status: "completed",
        finishedAt: Date.now(),
        message: `Indexed ${result.chunks} chunks.`,
      });
      appendJobLog(job, `Completed. Indexed ${result.chunks} chunks.`);
    } catch (err: any) {
      const cancelled = job.cancelled === true;
      updateJob(job, {
        status: cancelled ? "cancelled" : "failed",
        finishedAt: Date.now(),
        error: err?.message ?? String(err),
        message: cancelled ? "Cancelled." : "Failed.",
      });
      appendJobLog(
        job,
        cancelled ? "Cancelled." : `Failed: ${err?.message ?? String(err)}`
      );
    }
  })();

  return jobId;
};

const createIndexJobShell = () => {
  const jobId = `${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
  const job: IndexJob = {
    id: jobId,
    status: "queued",
    progress: 0,
    total: 0,
    logs: [],
  };
  indexJobs.set(jobId, job);
  updateJob(job, { status: "running", startedAt: Date.now() });
  appendJobLog(job, `Job ${jobId} started.`);
  return job;
};

const startIndexJobLazy = async (tableName?: string) => {
  const job = createIndexJobShell();

  // Fire-and-forget: do heavy work after returning jobId so the MCP App UI can render instantly.
  void (async () => {
    try {
      appendJobLog(job, "Loading embedding model…");
      await getExtractor();
      appendJobLog(job, "Model ready. Connecting to database…");
      const { notesTable } = await createNotesTable(tableName);
      appendJobLog(job, "Database ready.");
      const result = await indexNotes(notesTable, undefined, job);
      markIndexed();
      updateJob(job, {
        status: "completed",
        finishedAt: Date.now(),
        message: `Indexed ${result.chunks} chunks.`,
      });
      appendJobLog(job, `Completed. Indexed ${result.chunks} chunks.`);
    } catch (err: any) {
      const cancelled = job.cancelled === true;
      updateJob(job, {
        status: cancelled ? "cancelled" : "failed",
        finishedAt: Date.now(),
        error: err?.message ?? String(err),
        message: cancelled ? "Cancelled." : "Failed.",
      });
      appendJobLog(
        job,
        cancelled ? "Cancelled." : `Failed: ${err?.message ?? String(err)}`
      );
    }
  })();

  return job.id;
};

function registerTools(server: McpServer) {

server.tool("create-note", CreateNoteSchema.shape, async ({ title, content }) => {
  await createNote(title, content);
  return createTextResponse(`Created note "${title}" successfully.`);
});

const ListNotesSchema = z.object({
  folder: z.string().optional().describe("Filter by folder name (exact path segment)"),
  modifiedAfter: z.string().optional().describe("ISO date string — only notes modified after this date"),
  modifiedBefore: z.string().optional().describe("ISO date string — only notes modified before this date"),
  limit: z.number().int().min(1).max(500).optional().describe("Max results (default 50)"),
});

server.tool("list-notes", ListNotesSchema.shape, async ({ folder, modifiedAfter, modifiedBefore, limit = 50 }) => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const afterCf = modifiedAfter ? (new Date(modifiedAfter).getTime() / 1000 - CF_EPOCH) : null;
      const beforeCf = modifiedBefore ? (new Date(modifiedBefore).getTime() / 1000 - CF_EPOCH) : null;

      const rows = db.query<any, []>(`
        ${folderCte(db)}
        SELECT n.ZTITLE1 AS title,
               datetime(n.ZMODIFICATIONDATE1 + ${CF_EPOCH}, 'unixepoch') AS modified,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.Z_ENT = ${noteEnt(db)} AND n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0
          ${afterCf !== null ? `AND n.ZMODIFICATIONDATE1 > ${afterCf}` : ""}
          ${beforeCf !== null ? `AND n.ZMODIFICATIONDATE1 < ${beforeCf}` : ""}
        ORDER BY n.ZMODIFICATIONDATE1 DESC
        LIMIT ${limit}
      `).all();

      const filtered = folder
        ? rows.filter((r: any) => {
            const p = r.folder as string;
            return p === folder || p.startsWith(folder + "/") ||
                   p.endsWith("/" + folder) || p.includes("/" + folder + "/");
          })
        : rows;

      return createTextResponse(JSON.stringify(filtered));
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
});

server.tool("get-note", GetNoteSchema.shape, async ({ title }) => {
  // Try exact match first, fall back to semantic search
  const note = await getNoteDetailsByTitle(title);
  if (note && note.title) return createTextResponse(JSON.stringify(note));

  // Fuzzy fallback via search index
  kickReindexIfNeeded();
  const { notesTable } = await createNotesTable();
  const results = await searchAndCombineResults(notesTable, title, 1);
  if (!results.length) return createTextResponse(JSON.stringify({ error: `No note found matching "${title}"` }));
  const best = await getNoteDetailsByTitle(results[0].title);
  return createTextResponse(JSON.stringify(best));
});

server.tool("update-note", UpdateNoteSchema.shape, async ({ title, content }) => {
  const script = `
    const app = Application('Notes');
    const matches = app.notes.whose({name: ${JSON.stringify(title)}});
    if (matches.length === 0) throw new Error(${JSON.stringify("Note not found: " + title)});
    matches[0].body = ${JSON.stringify(content)};
  `;
  try {
    execSync(`osascript -l JavaScript -e ${JSON.stringify(script)}`);
  } catch (err: any) {
    return createTextResponse(JSON.stringify({ error: err.message }));
  }
  return createTextResponse(`Updated note "${title}".`);
});

server.tool("list-folders", {}, async () => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const rows = db.query<{ path: string; noteCount: number }, []>(`
        ${folderCte(db)}
        SELECT fp.path, COUNT(n.Z_PK) AS noteCount
        FROM folder_path fp
        LEFT JOIN ZICCLOUDSYNCINGOBJECT n ON n.ZFOLDER = fp.id AND n.Z_ENT = ${noteEnt(db)} AND n.ZMARKEDFORDELETION = 0
        GROUP BY fp.id, fp.path
        ORDER BY fp.path
      `).all();
      return createTextResponse(JSON.stringify(rows));
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
});

server.tool("list-tags", {}, async () => {
  const { notesTable } = await createNotesTable();
  const rows = await notesTable.query().select(["title", "content"]).limit(FULL_SCAN_LIMIT).toArray();
  const tagCounts = new Map<string, number>();
  const seen = new Set<string>();
  for (const row of rows) {
    if (seen.has(row.title)) continue;
    seen.add(row.title);
    for (const tag of extractTags(row.content ?? "")) {
      tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
    }
  }
  const sorted = [...tagCounts.entries()]
    .sort(([, a], [, b]) => b - a)
    .map(([tag, count]) => ({ tag: `#${tag}`, count }));
  return createTextResponse(JSON.stringify(sorted));
});

server.tool("search-by-tag", { tag: z.string().describe("Hashtag to search for, with or without leading #") }, async ({ tag }) => {
  const { notesTable } = await createNotesTable();
  const normalized = tag.replace(/^#/, "").toLowerCase();
  const rows = await notesTable.query().select(["title", "folder", "modification_date", "content"]).limit(FULL_SCAN_LIMIT).toArray();
  const seen = new Set<string>();
  const results: any[] = [];
  for (const row of rows) {
    if (seen.has(row.title)) continue;
    seen.add(row.title);
    if (extractTags(row.content ?? "").includes(normalized)) {
      results.push({ title: row.title, folder: row.folder, modified: row.modification_date });
    }
  }
  return createTextResponse(JSON.stringify(results));
});

server.tool("related-notes", {
  title: z.string().describe("Exact title of the source note"),
  limit: z.number().int().min(1).max(20).optional().describe("Max results (default 10)"),
}, async ({ title, limit = 10 }) => {
  kickReindexIfNeeded();
  const note = await getNoteDetailsByTitle(title);
  if (!note?.title) return createTextResponse(JSON.stringify({ error: `Note not found: "${title}"` }));

  const sourceTags = new Set(extractTags(note.content ?? ""));
  const sourceLinks = new Set(extractWikilinks(note.content ?? "").map(l => l.toLowerCase()));

  const { notesTable } = await createNotesTable();
  // Vector similarity over title + first 500 chars
  const [vectorResults, allRows] = await Promise.all([
    notesTable.search(`${note.title} ${(note.content ?? "").slice(0, 500)}`, "vector").limit(50).toArray(),
    notesTable.query().select(["title", "folder", "modification_date", "content"]).limit(FULL_SCAN_LIMIT).toArray(),
  ]);

  const scores = new Map<string, number>();
  const metaMap = new Map<string, { folder: string; modified: string }>();

  const seen = new Set<string>();
  for (const row of allRows) {
    if (row.title === title || seen.has(row.title)) continue;
    seen.add(row.title);
    metaMap.set(row.title, { folder: row.folder, modified: row.modification_date });

    let score = 0;
    for (const t of extractTags(row.content ?? "")) if (sourceTags.has(t)) score += 0.8;
    if (sourceLinks.has(row.title.toLowerCase())) score += 1.0; // outgoing wikilink
    const rowLinks = new Set(extractWikilinks(row.content ?? "").map(l => l.toLowerCase()));
    if (rowLinks.has(title.toLowerCase())) score += 1.0; // incoming backlink
    if (score > 0) scores.set(row.title, score);
  }

  vectorResults.forEach((r: any, idx: number) => {
    if (r.title === title) return;
    scores.set(r.title, (scores.get(r.title) ?? 0) + 0.5 / (60 + idx));
    if (!metaMap.has(r.title)) metaMap.set(r.title, { folder: r.folder, modified: r.modification_date });
  });

  const results = [...scores.entries()]
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
    .map(([t, score]) => ({ title: t, score: Math.round(score * 100) / 100, ...metaMap.get(t) }));

  return createTextResponse(JSON.stringify(results));
});

server.tool("bridge-notes", {
  limit: z.number().int().min(1).max(200).optional().describe("Max bridges (default 20)"),
  folder: z.string().optional().describe("Only bridges where A or C is in this folder (exact path segment match)"),
}, async ({ limit = 20, folder }) => {
  const { result: mined, stale } = await getMinedBridges();
  return createTextResponse(JSON.stringify({ ...selectBridges(mined, { limit, folder }), stale }));
});

server.tool("feed", {
  limit: z.number().int().min(1).max(100).optional().describe("Max feed items (default 20)"),
}, async ({ limit = 20 }) => {
  const result = await serveFeed(0, limit);
  return createTextResponse(JSON.stringify(result));
});

server.tool("entity-notes", {
  entity: z.string().describe("Entity name, e.g. 'Mercedes' — case-insensitive, prefix/contains fallback"),
}, async ({ entity }) => {
  const result = entityNotes(entity);
  if (result === null) return createTextResponse(NO_GRAPH_DB_MSG);
  return createTextResponse(JSON.stringify(result));
});

server.tool("list-entities", {
  query: z.string().optional().describe("Substring filter on entity labels"),
  limit: z.number().int().min(1).max(200).optional().describe("Max entities (default 30)"),
}, async ({ query, limit = 30 }) => {
  const result = listEntities(query, limit);
  if (result === null) return createTextResponse(NO_GRAPH_DB_MSG);
  return createTextResponse(JSON.stringify(result));
});

server.tool("get-tables", { title: z.string().describe("Exact note title") }, async ({ title }) => {
  const note = await getNoteDetailsByTitle(title);
  if (!note?.title) return createTextResponse(JSON.stringify({ error: `Note not found: "${title}"` }));
  const tables = extractTablesFromText(note.content ?? "");
  return createTextResponse(JSON.stringify({ title: note.title, tables, count: tables.length }));
});

server.tool("check-changes", {}, async () => {
  const currentMax = getNotesMaxModDate();
  const hasChanges = currentMax !== null && (lastIndexedModDate === null || currentMax > lastIndexedModDate);
  let changedCount = 0;
  if (hasChanges && lastIndexedModDate !== null) {
    try {
      const db = new Database(NOTES_DB, { readonly: true });
      try {
        const row = db.query<{ n: number }, [number]>(
          `SELECT COUNT(*) AS n FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = ${noteEnt(db)} AND ZMARKEDFORDELETION = 0 AND ZMODIFICATIONDATE1 > ?`
        ).get(lastIndexedModDate);
        changedCount = row?.n ?? 0;
      } finally { db.close(); }
    } catch { /* ignore */ }
  }
  return createTextResponse(JSON.stringify({
    hasChanges,
    changedCount: hasChanges ? changedCount : 0,
    lastIndexedAt: lastIndexedModDate ? new Date((lastIndexedModDate + CF_EPOCH) * 1000).toISOString() : null,
    currentMaxModDate: currentMax ? new Date((currentMax + CF_EPOCH) * 1000).toISOString() : null,
  }));
});

server.tool("search-notes", QueryNotesSchema.shape, async ({ query, folder, modifiedAfter, modifiedBefore }) => {
  kickReindexIfNeeded();
  const { notesTable } = await createNotesTable();
  const combinedResults = await searchAndCombineResults(notesTable, query, 20, folder, modifiedAfter, modifiedBefore);
  return createTextResponse(JSON.stringify(combinedResults));
});

server.tool("find-notes", QueryNotesSchema.shape, async ({ query, folder, modifiedAfter, modifiedBefore }) => {
  kickReindexIfNeeded();
  const { notesTable } = await createNotesTable();

  const queryL = query.toLowerCase();
  const variants = [queryL];
  // Also search hyphenated/de-hyphenated variants (e.g. "ar15" ↔ "ar-15")
  const hyph = queryL.replace(/([a-zA-Z])(\d)/g, "$1-$2").replace(/(\d)([a-zA-Z])/g, "$1-$2");
  const dehyph = queryL.replace(/([a-zA-Z])-(\d)/g, "$1$2").replace(/(\d)-([a-zA-Z])/g, "$1$2");
  if (hyph !== queryL) variants.push(hyph);
  if (dehyph !== queryL && dehyph !== hyph) variants.push(dehyph);

  const conditions = variants
    .flatMap(v => {
      const esc = v.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
      return [`lower(title) LIKE '%${esc}%'`, `lower(content) LIKE '%${esc}%'`];
    })
    .join(" OR ");

  const afterMs = modifiedAfter ? new Date(modifiedAfter).getTime() : null;
  const beforeMs = modifiedBefore ? new Date(modifiedBefore).getTime() : null;

  const rows = await notesTable.query()
    .where(conditions)
    .select(["title", "content", "folder", "modification_date"])
    .limit(FULL_SCAN_LIMIT)
    .toArray()
    .catch(() => [] as any[]);

  // Deduplicate by title (multiple chunks per note), apply filters, build snippets
  const seen = new Set<string>();
  const results: any[] = [];
  for (const row of rows) {
    if (seen.has(row.title)) continue;
    seen.add(row.title);
    if (folder) {
      const p = row.folder ?? "";
      if (!(p === folder || p.startsWith(folder + "/") || p.endsWith("/" + folder) || p.includes("/" + folder + "/"))) continue;
    }
    if (afterMs || beforeMs) {
      const ms = row.modification_date ? new Date(row.modification_date).getTime() : null;
      if (ms !== null) {
        if (afterMs && ms < afterMs) continue;
        if (beforeMs && ms > beforeMs) continue;
      }
    }
    const content = (row.content ?? "").toLowerCase();
    const matchVariant = variants.find(v => content.includes(v) || row.title.toLowerCase().includes(v)) ?? queryL;
    const idx = content.indexOf(matchVariant);
    const snippet = idx >= 0
      ? row.content.slice(Math.max(0, idx - 60), idx + 240).trim()
      : row.content.slice(0, 300).trim();
    results.push({ title: row.title, folder: row.folder, modification_date: row.modification_date, snippet });
  }

  return createTextResponse(JSON.stringify(results));
});

server.tool("index-health", {}, async () => {
  const maxMod = getNotesMaxModDate();
  const totalNotes = (() => {
    try {
      const db = new Database(NOTES_DB, { readonly: true });
      try {
        return (db.query<{ n: number }, []>(`SELECT COUNT(*) AS n FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = ${noteEnt(db)} AND ZMARKEDFORDELETION = 0`).get()?.n ?? 0);
      } finally { db.close(); }
    } catch { return null; }
  })();
  return createTextResponse(JSON.stringify({
    lastIndexedAt: lastIndexedModDate ? new Date((lastIndexedModDate + CF_EPOCH) * 1000).toISOString() : null,
    currentMaxModDate: maxMod ? new Date((maxMod + CF_EPOCH) * 1000).toISOString() : null,
    inSync: lastIndexedModDate !== null && maxMod !== null && maxMod <= lastIndexedModDate,
    activeJob: isActiveIndexJob(),
    totalNotesInApple: totalNotes,
  }));
});

server.tool("index-notes-blocking", {}, async (_args, extra) => {
  const { notesTable } = await createNotesTable();
  const { time, chunks } = await indexNotes(notesTable, extra);
  markIndexed();
  return createTextResponse(
    `Indexed ${chunks} notes chunks in ${time}ms. You can now search for them using the "search-notes" tool.`
  );
});

registerAppTool(
  server,
  "index-notes",
  {
    title: "Index Apple Notes",
    description:
      "Starts indexing Apple Notes in the background and opens an MCP App UI that shows progress and logs without blocking conversation.",
    inputSchema: {},
    _meta: { ui: { resourceUri: INDEXER_RESOURCE_URI } },
  },
  async () => {
    const jobId = await startIndexJobLazy();
    return createTextResponse(JSON.stringify({ jobId }));
  }
);

server.tool("start-index-notes", StartIndexNotesSchema.shape, async ({ tableName }) => {
  const jobId = await startIndexJobLazy(tableName);
  return createTextResponse(JSON.stringify({ jobId }));
});

server.tool("index-notes-status", JobIdSchema.shape, async ({ jobId }) => {
  const job = indexJobs.get(jobId);
  if (!job) return createTextResponse(`Unknown jobId: ${jobId}`);
  return createTextResponse(
    JSON.stringify({
      jobId: job.id,
      status: job.status,
      progress: job.progress,
      total: job.total,
      message: job.message,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      error: job.error,
    })
  );
});

server.tool("index-notes-logs", JobLogsSchema.shape, async ({ jobId, offset }) => {
  const job = indexJobs.get(jobId);
  if (!job) return createTextResponse(`Unknown jobId: ${jobId}`);
  const start = offset ?? 0;
  const lines = job.logs.slice(start);
  return createTextResponse(
    JSON.stringify({
      jobId: job.id,
      nextOffset: job.logs.length,
      lines,
    })
  );
});

server.tool("cancel-index-notes", JobIdSchema.shape, async ({ jobId }) => {
  const job = indexJobs.get(jobId);
  if (!job) return createTextResponse(`Unknown jobId: ${jobId}`);
  job.cancelled = true;
  appendJobLog(job, "Cancellation requested.");
  return createTextResponse(JSON.stringify({ jobId: job.id, cancelled: true }));
});

} // end registerTools

// Start the server — only when run directly, not when imported by tests
if (import.meta.main) {

const PORT = parseInt(process.env.MCP_PORT ?? "3741");

if (process.argv.includes("--stdio")) {
  const server = createServer();
  await server.connect(new StdioServerTransport());
  console.error("MCP server running on stdio");
  // Pre-warm the ONNX model in the background so it's ready before the user calls index-notes
  void getExtractor().then(() => console.error("Model pre-warm complete")).catch(() => {});
} else {
  const httpServer = http.createServer(async (req, res) => {
    // Same-machine only: reflect a localhost origin, never wildcard (the server binds 127.0.0.1
    // and has no auth, so a wildcard would let any website read your notes via the browser).
    const origin = req.headers.origin;
    if (origin && /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
    }
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id, Mcp-Protocol-Version");
    res.setHeader("Access-Control-Expose-Headers", "Mcp-Session-Id");
    if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

    const u = new URL(req.url ?? "/", `http://localhost:${PORT}`);
    const sendJson = (code: number, obj: unknown) => {
      res.writeHead(code, { "Content-Type": "application/json" });
      res.end(JSON.stringify(obj));
    };

    // ---- muser-style local web UI (standalone, no MCP client needed) ----
    if (req.method === "GET" && (u.pathname === "/" || u.pathname === "/index.html")) {
      try {
        const html = await fs.readFile(path.join(import.meta.dirname, "web", "search.html"), "utf-8");
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(html);
      } catch {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end("Missing web/search.html");
      }
      return;
    }

    // ---- JSON search API (mirrors muser: /api/status, /api/search, /api/index) ----
    if (req.method === "GET" && u.pathname === "/api/status") {
      try {
        const { notesTable } = await createNotesTable();
        const indexed = await notesTable.countRows().catch(() => 0);
        sendJson(200, { model: "all-MiniLM-L6-v2", indexed, db: path.join(os.homedir(), ".mcp-apple-notes", "data") });
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/search") {
      const q = u.searchParams.get("q") ?? "";
      if (!q.trim()) { sendJson(200, { results: [], ms: 0 }); return; }
      const k = Math.min(parseInt(u.searchParams.get("k") ?? "25") || 25, 100);
      const folder = u.searchParams.get("folder") ?? undefined;
      try {
        kickReindexIfNeeded();
        const { notesTable } = await createNotesTable();
        const t0 = performance.now();
        const results = await searchAndCombineResults(notesTable, q, k, folder);
        sendJson(200, { results, ms: Math.round(performance.now() - t0) });
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e), results: [] });
      }
      return;
    }

    // ---- entity layer (bun:sqlite over the optional layered graph db) ----
    if (req.method === "GET" && u.pathname === "/api/entities") {
      const q = u.searchParams.get("q")?.trim() || undefined;
      const limit = Math.min(Math.max(parseInt(u.searchParams.get("limit") ?? "30") || 30, 1), 200);
      try {
        const result = listEntities(q, limit);
        if (result === null) { sendJson(503, { error: NO_GRAPH_DB_MSG }); return; }
        sendJson(200, result);
      } catch (e: any) { sendJson(500, { error: String(e?.message ?? e) }); }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/entity-notes") {
      const entity = u.searchParams.get("entity")?.trim();
      if (!entity) { sendJson(400, { error: "missing entity" }); return; }
      try {
        const result = entityNotes(entity);
        if (result === null) { sendJson(503, { error: NO_GRAPH_DB_MSG }); return; }
        sendJson(200, result);
      } catch (e: any) { sendJson(500, { error: String(e?.message ?? e) }); }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/synthesize") {
      const q = (u.searchParams.get("q") ?? "").trim();
      if (!q) { sendJson(400, { error: "missing q" }); return; }
      const llm = getLlmConfig();
      if (!llm) { sendJson(503, { error: "synthesis unavailable: set OPENAI_API_KEY (and optionally SYNTH_BASE_URL / SYNTH_MODEL)" }); return; }
      try {
        const { notesTable } = await createNotesTable();
        const extractor = await getExtractor();
        const embedOne = async (s: string) => Array.from((await extractor(s, { pooling: "mean", normalize: true })).data) as number[];
        const embedBatch = async (arr: string[]) => (await extractor(arr, { pooling: "mean", normalize: true })).tolist() as number[][];
        const result = await synthesize({ table: notesTable, topic: q, embedOne, embedBatch, apiKey: llm.key, baseURL: llm.baseURL, model: llm.model });
        sendJson(200, result);
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/clusters") {
      const k = Math.min(Math.max(parseInt(u.searchParams.get("k") ?? "12") || 12, 2), 40);
      try {
        const { result, stale, recomputing: rec } = await getClusters(k);
        sendJson(200, { ...result, stale, recomputing: rec });
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/bridges") {
      const limit = Math.min(Math.max(parseInt(u.searchParams.get("limit") ?? "40") || 40, 1), 200);
      const folder = u.searchParams.get("folder") ?? undefined;
      try {
        const { result: mined, stale, recomputing: rec } = await getMinedBridges();
        sendJson(200, { ...selectBridges(mined, { limit, folder }), stale, recomputing: rec });
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/feed") {
      const limit = Math.min(Math.max(parseInt(u.searchParams.get("limit") ?? "20") || 20, 1), 100);
      const offset = Math.max(parseInt(u.searchParams.get("offset") ?? "0") || 0, 0);
      try {
        sendJson(200, await serveFeed(offset, limit));
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "POST" && u.pathname === "/api/vote") {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", async () => {
        try {
          const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
          const id = String(body.id ?? "");
          const vote = body.vote === 1 || body.vote === -1 ? body.vote : null;
          if (!id || vote === null) { sendJson(400, { error: "expected {id, vote: 1|-1}" }); return; }
          await getFeedItems(); // make sure items (and their features) are loaded
          const r = recordVote(id, vote);
          if ("error" in r) { sendJson(404, r); return; }
          sendJson(200, { ok: true, weights: feedWeights.map(w => Math.round(w * 10000) / 10000) });
        } catch (e: any) {
          sendJson(500, { error: String(e?.message ?? e) });
        }
      });
      return;
    }

    if (req.method === "POST" && u.pathname === "/api/index") {
      try {
        const jobId = await startIndexJobLazy();
        sendJson(200, { jobId });
      } catch (e: any) {
        sendJson(500, { error: String(e?.message ?? e) });
      }
      return;
    }

    if (req.method === "GET" && u.pathname === "/api/index-status") {
      const jobId = u.searchParams.get("jobId") ?? "";
      const job = indexJobs.get(jobId);
      if (!job) { sendJson(404, { error: "unknown jobId" }); return; }
      sendJson(200, {
        jobId: job.id, status: job.status, progress: job.progress,
        total: job.total, message: job.message, error: job.error,
      });
      return;
    }

    // ---- MCP-over-HTTP transport ----
    if (u.pathname === "/mcp") {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", async () => {
        const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
        const server = createServer();
        const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
        res.on("close", () => { transport.close(); server.close(); });
        await server.connect(transport);
        await transport.handleRequest(req, res, body);
      });
      return;
    }

    res.writeHead(404); res.end();
  });
  // Bind loopback only — the HTTP UI and /mcp endpoint are unauthenticated and can read AND
  // write Apple Notes (via osascript), so they must never be exposed to the LAN or a tunnel.
  httpServer.listen(PORT, "127.0.0.1", () => {
    console.error(`Apple Notes search UI:  http://127.0.0.1:${PORT}/`);
    console.error(`MCP endpoint:           http://127.0.0.1:${PORT}/mcp`);
  });
}

} // end import.meta.main

/**
 * Search for notes by title or content using both vector and FTS search.
 * The results are combined using RRF
 */
const TEMPORAL_RE = /\b(recent|latest|newest|last|today|this week|new|current)\b/i;

export const searchAndCombineResults = async (
  notesTable: any,
  query: string,
  limit = 20,
  folder?: string,
  modifiedAfter?: string,
  modifiedBefore?: string,
) => {
  const isTemporalQuery = TEMPORAL_RE.test(query);
  const candidateLimit = Math.max(limit * 3, 60);

  // Folder segment matcher — checks if `name` is an exact segment in the path
  const matchesFolder = folder
    ? (path: string) => {
        const seg = folder;
        return path === seg || path.startsWith(seg + "/") ||
               path.endsWith("/" + seg) || path.includes("/" + seg + "/");
      }
    : null;

  // Build query variants to match both "AR15" and "AR-15" in content:
  //   - Original: matches content already normalized to "AR15"
  //   - Hyphenated: "ar15" → "ar-15" matches content with "AR-15"
  const queryL = query.toLowerCase();
  const queryHyphenated = queryL.replace(/([a-zA-Z])(\d)/g, "$1-$2").replace(/(\d)([a-zA-Z])/g, "$1-$2");
  const queryVariants = new Set([queryL, queryHyphenated,
    queryL.replace(/([a-zA-Z])-(\d)/g, "$1$2").replace(/(\d)-([a-zA-Z])/g, "$1$2")]);

  // FTS returns garbage results when a query term has no index matches.
  // Filter: only keep FTS results where the actual text contains the query (or a variant).
  const isFtsRelevant = (r: any) => {
    const text = (r.title + " " + r.content).toLowerCase();
    return [...queryVariants].some(v => text.includes(v));
  };

  // Exact substring search: what Apple Notes does natively. Catches tokenization gaps
  // (e.g. "AR-15" found by variant "ar-15", "AR15" found by variant "ar15").
  const substringConditions = [...queryVariants]
    .flatMap(v => {
      const esc = v.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
      return [`lower(title) LIKE '%${esc}%'`, `lower(content) LIKE '%${esc}%'`];
    })
    .join(" OR ");

  // Fuzzy FTS: tantivy ~1 edit distance on terms ≥4 chars — catches typos.
  // Phrase FTS: wrap multi-word queries in quotes for adjacency matching.
  const words = queryL.split(/\s+/).filter(Boolean);
  const fuzzyQuery = words.map(w => w.length >= 4 ? `${w}~1` : w).join(" ");
  const phraseQuery = words.length > 1 ? `"${queryL}"` : null;

  const [vectorResults, ftsContentResults, ftsTitleResults, substringResults, fuzzyResults, phraseResults] = await Promise.all([
    notesTable.search(query, "vector").limit(candidateLimit).toArray(),
    notesTable.search(query, "fts", "content").limit(candidateLimit).toArray().catch(() => [] as any[]),
    notesTable.search(query, "fts", "title").limit(candidateLimit).toArray().catch(() => [] as any[]),
    notesTable.query().where(substringConditions).select(["title", "content", "folder", "modification_date"]).limit(FULL_SCAN_LIMIT).toArray().catch(() => [] as any[]),
    notesTable.search(fuzzyQuery, "fts", "content").limit(candidateLimit).toArray().catch(() => [] as any[]),
    phraseQuery ? notesTable.search(phraseQuery, "fts", "content").limit(candidateLimit).toArray().catch(() => [] as any[]) : Promise.resolve([] as any[]),
  ]);

  const k = 60;
  const scores = new Map<string, number>();

  // Store full content + folder + modification_date in side maps; key is title
  const contentMap = new Map<string, string>();
  const folderMap = new Map<string, string>();
  const modDateMap = new Map<string, string>();

  // Deduplicate chunks per note within each result set so a chunky note can't
  // accumulate 10× score. Only the highest-ranked chunk (lowest idx) counts.
  const processResults = (results: any[], weight = 1, relevanceFilter = false) => {
    const seen = new Set<string>();
    const filtered = relevanceFilter ? results.filter(isFtsRelevant) : results;
    filtered.forEach((result, idx) => {
      const key = result.title;
      if (seen.has(key)) return;
      seen.add(key);
      if (!contentMap.has(key)) contentMap.set(key, result.content ?? "");
      if (!folderMap.has(key)) folderMap.set(key, result.folder ?? "");
      if (!modDateMap.has(key)) modDateMap.set(key, result.modification_date ?? "");
      const score = weight / (k + idx);
      scores.set(key, (scores.get(key) || 0) + score);
    });
  };

  processResults(vectorResults);
  processResults(ftsContentResults, 1, true);   // filter garbage FTS results
  processResults(ftsTitleResults, 3, true);      // title match = 3× weight, filter garbage
  processResults(substringResults, 2, false);    // exact substring — high weight, catches tokenization gaps
  processResults(fuzzyResults, 0.8, true);       // fuzzy ~1 edit distance — typo tolerance
  processResults(phraseResults, 1.5, true);      // phrase match — exact adjacency for multi-word queries

  // --- Re-ranking: multiplicative combination (IR best practice) ---
  // final_score = rrf_score * title_boost * recency_decay^(recency_alpha)
  // Relevance (RRF) stays primary; recency and title modulate but can't override.

  const now = Date.now();
  // Temporal queries: 1-day half-life, strong recency alpha.
  // Normal queries: 90-day half-life, weak recency alpha (just a tiebreaker).
  const HALF_LIFE_MS = isTemporalQuery ? 1 * 24 * 60 * 60 * 1000 : 90 * 24 * 60 * 60 * 1000;
  const recencyAlpha = isTemporalQuery ? 0.7 : 0.1; // how much recency shifts the final score

  const queryWords = query.toLowerCase().split(/\W+/).filter(w => w.length > 2);

  for (const [title, rrf] of scores) {
    // Title field boost: 2x per matching query word, capped at 3x total (standard Elasticsearch default)
    const tl = title.toLowerCase();
    const titleMatches = queryWords.filter(w => tl.includes(w)).length;
    const titleBoost = Math.min(1 + titleMatches * 0.5, 3.0);

    // Recency decay: [0, 1] — 1.0 = just modified, approaches 0 for old notes
    const modDate = modDateMap.get(title);
    const ageMs = modDate ? now - new Date(modDate).getTime() : Infinity;
    const recencyDecay = Math.exp(-Math.LN2 * ageMs / HALF_LIFE_MS);

    // Multiplicative blend: relevance * title * lerp(1, recencyDecay, alpha)
    const recencyFactor = 1 - recencyAlpha + recencyAlpha * recencyDecay;
    scores.set(title, rrf * titleBoost * recencyFactor);
  }

  const afterMs = modifiedAfter ? new Date(modifiedAfter).getTime() : null;
  const beforeMs = modifiedBefore ? new Date(modifiedBefore).getTime() : null;

  const SNIPPET_LEN = 300;
  const results = Array.from(scores.entries())
    .sort(([, a], [, b]) => b - a)
    .filter(([title]) => {
      if (matchesFolder && !matchesFolder(folderMap.get(title) ?? "")) return false;
      const mod = modDateMap.get(title);
      if (mod) {
        const ms = new Date(mod).getTime();
        if (afterMs !== null && ms < afterMs) return false;
        if (beforeMs !== null && ms > beforeMs) return false;
      }
      return true;
    })
    .slice(0, limit)
    .map(([title]) => ({
      title,
      folder: folderMap.get(title) ?? "",
      modified: modDateMap.get(title) ?? "",
      snippet: (contentMap.get(title) ?? "").slice(0, SNIPPET_LEN),
    }));

  return results;
};
