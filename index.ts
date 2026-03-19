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

const getExtractor = async () => {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline } = await import("@huggingface/transformers");
      return await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    })();
  }
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
      return LanceSchema({
        title: func.sourceField(new Utf8()),
        content: func.sourceField(new Utf8()),
        creation_date: func.sourceField(new Utf8()),
        modification_date: func.sourceField(new Utf8()),
        folder: func.sourceField(new Utf8()),
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

// Track the max modification_date at the time of last successful index
let lastIndexedModDate: number | null = null;

const getNotesMaxModDate = (): number | null => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<{ d: number | null }, []>(
        "SELECT MAX(ZMODIFICATIONDATE1) AS d FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 11 AND ZMARKEDFORDELETION = 0"
      ).get();
      return row?.d ?? null;
    } finally { db.close(); }
  } catch { return null; }
};

const isActiveIndexJob = () =>
  [...indexJobs.values()].some((j) => j.status === "running" || j.status === "queued");

// Call before search: if notes have changed, re-index synchronously before returning.
// Incremental, so usually < 1s for a few changed notes.
const syncReindexIfNeeded = async () => {
  const maxMod = getNotesMaxModDate();
  if (maxMod === null) return;
  if (lastIndexedModDate !== null && maxMod <= lastIndexedModDate) return;

  // If a job is already running (e.g. manual index-notes), wait for it to finish
  if (isActiveIndexJob()) {
    await new Promise<void>((resolve) => {
      const iv = setInterval(() => {
        if (!isActiveIndexJob()) { clearInterval(iv); resolve(); }
      }, 200);
    });
    lastIndexedModDate = getNotesMaxModDate();
    return;
  }

  // Run inline incremental re-index — blocks until done
  const { notesTable } = await createNotesTable();
  await indexNotes(notesTable);
  lastIndexedModDate = getNotesMaxModDate();
};

const server = new McpServer({
  name: "my-apple-notes-mcp",
  version: "1.0.0",
});

const INDEXER_RESOURCE_URI = "ui://apple-notes/indexer.html";

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


const NOTES_DB = path.join(
  os.homedir(),
  "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"
);

// CoreData timestamps: seconds since 2001-01-01
const CF_EPOCH = 978307200;
const cfDate = (cf: number) => new Date((cf + CF_EPOCH) * 1000).toISOString();

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

const FOLDER_CTE = `
  WITH RECURSIVE folder_path(id, path) AS (
    SELECT Z_PK, ZTITLE2
    FROM ZICCLOUDSYNCINGOBJECT
    WHERE Z_ENT = 14 AND ZPARENT IS NULL AND ZTITLE2 IS NOT NULL
    UNION ALL
    SELECT f.Z_PK, fp.path || '/' || f.ZTITLE2
    FROM ZICCLOUDSYNCINGOBJECT f
    JOIN folder_path fp ON f.ZPARENT = fp.id
    WHERE f.Z_ENT = 14 AND f.ZTITLE2 IS NOT NULL
  )
`;

const getAllNoteDetails = async () => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        ${FOLDER_CTE}
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0 AND n.Z_ENT = 11
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
        ${FOLDER_CTE}
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.ZTITLE1 = ? AND n.ZMARKEDFORDELETION = 0 AND n.Z_ENT = 11
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

  // 2. Query existing index to find what's already embedded
  const existingMap = new Map<string, string>(); // title → modification_date
  try {
    const rows = await notesTable.query().select(["title", "modification_date"]).toArray();
    for (const row of rows) existingMap.set(row.title, row.modification_date);
  } catch { /* empty table */ }

  // 3. Remove notes deleted from Apple Notes
  const allTitles = new Set(allNotes.map((n) => n.title));
  const deletedTitles = [...existingMap.keys()].filter((t) => !allTitles.has(t));
  for (const title of deletedTitles) {
    await notesTable.delete(`title = '${escapeForFilter(title)}'`);
  }

  // 4. Filter to only new or modified notes
  const toIndex = allNotes.filter((n) => {
    if (!n.title) return false;
    return existingMap.get(n.title) !== n.modification_date;
  });
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

  // 5. Delete stale versions of modified notes before re-embedding
  for (const note of toIndex) {
    if (existingMap.has(note.title)) {
      await notesTable.delete(`title = '${escapeForFilter(note.title)}'`);
    }
  }

  const overallTotal = toIndex.length;
  if (job) updateJob(job, { progress: 0, total: overallTotal, message: `Embedding ${toIndex.length} notes…` });
  await maybeSendProgress(extra, 0, overallTotal, `Embedding ${toIndex.length} notes…`);
  await maybeSendMessage(extra, `Embedding ${toIndex.length} notes…`);

  // 6. Build chunks: convert HTML → Markdown, then split large notes
  const CHUNK_SIZE = 1500; // chars (~300 tokens)
  const CHUNK_OVERLAP = 150;
  const chunks: Array<{ id: string; title: string; content: string; creation_date: string; modification_date: string; folder: string }> = [];

  let td: ((html: string) => string) | null = null;
  try { td = await getTurndown(); } catch { /* ignore */ }

  const { RecursiveCharacterTextSplitter } = await import("@langchain/textsplitters");
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP });

  for (const note of toIndex) {
    let text = note.content ?? "";
    if (td) { try { text = td(text); } catch { /* keep original */ } }
    text = filterContent(text);

    const splits = text.length > CHUNK_SIZE
      ? await splitter.splitText(text)
      : [text];

    splits.forEach((chunk, ci) => {
      chunks.push({
        id: `${note.title}::${ci}`,
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
  await fs.rm(path.join(os.homedir(), ".mcp-apple-notes"), { recursive: true, force: true });
};

const createNotesTableInner = async (overrideName?: string) => {
  const db = await getDb();
  const notesTableSchema = await getNotesTableSchema();
  const notesTable = await db.createEmptyTable(
    overrideName || "notes",
    notesTableSchema,
    { mode: "create", existOk: true }
  );
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
      lastIndexedModDate = getNotesMaxModDate();
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
      lastIndexedModDate = getNotesMaxModDate();
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
        ${FOLDER_CTE}
        SELECT n.ZTITLE1 AS title,
               datetime(n.ZMODIFICATIONDATE1 + ${CF_EPOCH}, 'unixepoch') AS modified,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.Z_ENT = 11 AND n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0
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
  await syncReindexIfNeeded();
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
    if (matches.length === 0) throw new Error('Note not found: ${title.replace(/'/g, "\\'")}');
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
        ${FOLDER_CTE}
        SELECT fp.path, COUNT(n.Z_PK) AS noteCount
        FROM folder_path fp
        LEFT JOIN ZICCLOUDSYNCINGOBJECT n ON n.ZFOLDER = fp.id AND n.Z_ENT = 11 AND n.ZMARKEDFORDELETION = 0
        GROUP BY fp.id, fp.path
        ORDER BY fp.path
      `).all();
      return createTextResponse(JSON.stringify(rows));
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
});

server.tool("list-tags", {}, async () => {
  const { notesTable } = await createNotesTable();
  const rows = await notesTable.query().select(["title", "content"]).toArray();
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
  const rows = await notesTable.query().select(["title", "folder", "modification_date", "content"]).toArray();
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
  await syncReindexIfNeeded();
  const note = await getNoteDetailsByTitle(title);
  if (!note?.title) return createTextResponse(JSON.stringify({ error: `Note not found: "${title}"` }));

  const sourceTags = new Set(extractTags(note.content ?? ""));
  const sourceLinks = new Set(extractWikilinks(note.content ?? "").map(l => l.toLowerCase()));

  const { notesTable } = await createNotesTable();
  // Vector similarity over title + first 500 chars
  const [vectorResults, allRows] = await Promise.all([
    notesTable.search(`${note.title} ${(note.content ?? "").slice(0, 500)}`, "vector").limit(50).toArray(),
    notesTable.query().select(["title", "folder", "modification_date", "content"]).toArray(),
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
          "SELECT COUNT(*) AS n FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 11 AND ZMARKEDFORDELETION = 0 AND ZMODIFICATIONDATE1 > ?"
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
  await syncReindexIfNeeded();
  const { notesTable } = await createNotesTable();
  const combinedResults = await searchAndCombineResults(notesTable, query, 20, folder, modifiedAfter, modifiedBefore);
  return createTextResponse(JSON.stringify(combinedResults));
});

server.tool("index-health", {}, async () => {
  const maxMod = getNotesMaxModDate();
  const totalNotes = (() => {
    try {
      const db = new Database(NOTES_DB, { readonly: true });
      try {
        return (db.query<{ n: number }, []>("SELECT COUNT(*) AS n FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 11 AND ZMARKEDFORDELETION = 0").get()?.n ?? 0);
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

// Start the server — only when run directly, not when imported by tests
if (import.meta.main) {

const PORT = parseInt(process.env.MCP_PORT ?? "3741");

if (process.argv.includes("--stdio")) {
  await server.connect(new StdioServerTransport());
  console.error("MCP server running on stdio");
  // Pre-warm the ONNX model in the background so it's ready before the user calls index-notes
  void getExtractor().then(() => console.error("Model pre-warm complete")).catch(() => {});
} else {
  const httpServer = http.createServer(async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id");
    if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
    if (req.url === "/mcp") {
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", async () => {
        const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
        const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
        res.on("close", () => { transport.close(); server.close(); });
        await server.connect(transport);
        await transport.handleRequest(req, res, body);
      });
    } else {
      res.writeHead(404); res.end();
    }
  });
  httpServer.listen(PORT, () => {
    console.error(`MCP server running on http://localhost:${PORT}/mcp`);
    console.error(`Expose with: npx cloudflared tunnel --url http://localhost:${PORT}`);
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

  // Normalize query for FTS: "AR-15" → "AR15", "wi-fi" → "wifi" so hyphenated
  // and non-hyphenated variants match. Tantivy tokenizes on hyphens so "AR-15"
  // indexes as ["ar","15"] but "AR15" indexes as ["ar15"] — they never intersect.
  const ftsQuery = query.replace(/([a-zA-Z])-(\d)|(\d)-([a-zA-Z])|([a-zA-Z])-([a-zA-Z])/g,
    (_, a, b, c, d, e, f) => (a && b) ? a+b : (c && d) ? c+d : e+f);

  const [vectorResults, ftsContentResults, ftsTitleResults] = await Promise.all([
    notesTable.search(query, "vector").limit(candidateLimit).toArray(),
    notesTable.search(ftsQuery, "fts", "content").limit(candidateLimit).toArray().catch(() => [] as any[]),
    notesTable.search(ftsQuery, "fts", "title").limit(candidateLimit).toArray().catch(() => [] as any[]),
  ]);

  const k = 60;
  const scores = new Map<string, number>();

  // Store full content + folder + modification_date in side maps; key is title
  const contentMap = new Map<string, string>();
  const folderMap = new Map<string, string>();
  const modDateMap = new Map<string, string>();
  const processResults = (results: any[], weight = 1) => {
    results.forEach((result, idx) => {
      const key = result.title;
      if (!contentMap.has(key)) contentMap.set(key, result.content ?? "");
      if (!folderMap.has(key)) folderMap.set(key, result.folder ?? "");
      if (!modDateMap.has(key)) modDateMap.set(key, result.modification_date ?? "");
      const score = weight / (k + idx);
      scores.set(key, (scores.get(key) || 0) + score);
    });
  };

  processResults(vectorResults);
  processResults(ftsContentResults);
  // Title FTS gets 2× weight — an exact keyword match in the title is very high signal
  processResults(ftsTitleResults, 2);

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
