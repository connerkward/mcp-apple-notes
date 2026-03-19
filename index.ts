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
        vector: func.vectorField(),
      });
    })();
  }
  return await notesTableSchemaPromise;
};

const QueryNotesSchema = z.object({
  query: z.string(),
});

const GetNoteSchema = z.object({
  title: z.string(),
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
const cfDate = (cf: number) => new Date((cf + CF_EPOCH) * 1000).toLocaleString();

function extractText(data: Uint8Array): string {
  try {
    const buf = gunzipSync(Buffer.from(data));
    return buf.toString("utf8")
      .replace(/[^\x20-\x7E\x0A\x0D\u00A0-\uFFFF]/g, " ")
      .replace(/ {3,}/g, " ")
      .trim();
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

const getAllNoteDetails = async () => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        WHERE n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0
      `).all();
      return rows.map((r: any) => ({
        title: r.title as string,
        content: r.data ? extractText(r.data) : "",
        creation_date: r.created ? cfDate(r.created) : "",
        modification_date: r.modified ? cfDate(r.modified) : "",
      }));
    } finally { db.close(); }
  } catch (err) { handleDbError(err); }
};

const getNoteDetailsByTitle = async (title: string) => {
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<any, [string]>(`
        SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
               n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
        WHERE n.ZTITLE1 = ? AND n.ZMARKEDFORDELETION = 0
        LIMIT 1
      `).get(title);
      if (!row) return {} as any;
      return {
        title: row.title as string,
        content: row.data ? extractText(row.data) : "",
        creation_date: row.created ? cfDate(row.created) : "",
        modification_date: row.modified ? cfDate(row.modified) : "",
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
    await notesTable.delete(`title = '${title.replace(/'/g, "''")}'`);
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
      await notesTable.delete(`title = '${note.title.replace(/'/g, "''")}'`);
    }
  }

  const overallTotal = toIndex.length;
  if (job) updateJob(job, { progress: 0, total: overallTotal, message: `Embedding ${toIndex.length} notes…` });
  await maybeSendProgress(extra, 0, overallTotal, `Embedding ${toIndex.length} notes…`);
  await maybeSendMessage(extra, `Embedding ${toIndex.length} notes…`);

  // 6. Build chunks and convert HTML → Markdown
  const chunks = toIndex.map((note, i) => ({
    id: i.toString(),
    title: note.title,
    content: note.content,
    creation_date: note.creation_date,
    modification_date: note.modification_date,
  }));
  try {
    const td = await getTurndown();
    for (const chunk of chunks) {
      try { chunk.content = td(chunk.content || ""); } catch { /* keep original */ }
    }
  } catch { /* ignore turndown load errors */ }

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
  if (!indices.find((index: any) => index.name === "content_idx")) {
    const lancedb = await import("@lancedb/lancedb");
    await notesTable.createIndex("content", {
      config: lancedb.Index.fts(),
      replace: true,
    });
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

server.tool("list-notes", {}, async () => {
  const { notesTable } = await createNotesTable();
  return createTextResponse(
    `There are ${await notesTable.countRows()} notes in your Apple Notes database.`
  );
});

server.tool("get-note", GetNoteSchema.shape, async ({ title }) => {
  const note = await getNoteDetailsByTitle(title);
  return createTextResponse(JSON.stringify(note));
});

server.tool("search-notes", QueryNotesSchema.shape, async ({ query }) => {
  const { notesTable } = await createNotesTable();
  const combinedResults = await searchAndCombineResults(notesTable, query);
  return createTextResponse(JSON.stringify(combinedResults));
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

// Start the server — HTTP mode for MCP Apps UI, stdio as fallback
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
        res.on("close", () => transport.close());
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

/**
 * Search for notes by title or content using both vector and FTS search.
 * The results are combined using RRF
 */
export const searchAndCombineResults = async (
  notesTable: any,
  query: string,
  limit = 20
) => {
  const [vectorResults, ftsSearchResults] = await Promise.all([
    (async () => {
      const results = await notesTable
        .search(query, "vector")
        .limit(limit)
        .toArray();
      return results;
    })(),
    (async () => {
      const results = await notesTable
        .search(query, "fts", "content")
        .limit(limit)
        .toArray();
      return results;
    })(),
  ]);

  const k = 60;
  const scores = new Map<string, number>();

  const processResults = (results: any[], startRank: number) => {
    results.forEach((result, idx) => {
      const key = `${result.title}::${result.content}`;
      const score = 1 / (k + startRank + idx);
      scores.set(key, (scores.get(key) || 0) + score);
    });
  };

  processResults(vectorResults, 0);
  processResults(ftsSearchResults, 0);

  const results = Array.from(scores.entries())
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
    .map(([key]) => {
      const [title, content] = key.split("::");
      return { title, content };
    });

  return results;
};
