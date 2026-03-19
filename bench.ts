/**
 * Benchmark: JXA vs SQLite for fetching Apple Notes data
 *
 * Discovery: app.notes.body() bulk call fails with error -1741 on macOS Notes.
 * Individual note.body() calls work but cost ~1.5s each (IPC overhead per call),
 * making full-corpus JXA body fetch infeasible at scale (1844 notes ≈ 46 min).
 *
 * This benchmark therefore measures:
 *   A1. JXA metadata-only bulk fetch (name + dates) — what actually works fast
 *   A2. JXA body fetch for a small sample (10 notes) — to characterize per-note cost
 *   B.  SQLite direct read — full fetch (title + body + dates)
 */

import { runJxa } from "run-jxa";
import { Database } from "bun:sqlite";
import { gunzipSync } from "node:zlib";
import os from "node:os";
import path from "node:path";

const NOTES_DB = path.join(
  os.homedir(),
  "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"
);
const CF_EPOCH = 978307200;
const cfDate = (cf: number) => new Date((cf + CF_EPOCH) * 1000).toLocaleString();

function extractText(data: Uint8Array): string {
  try {
    const buf = gunzipSync(Buffer.from(data));
    return buf.toString("utf8").replace(/[^\x20-\x7E\x0A\x0D\u00A0-\uFFFF]/g, " ").replace(/ {3,}/g, " ").trim();
  } catch {
    return "";
  }
}

// A1: JXA metadata bulk fetch (works fine)
async function jxaMetadata(): Promise<{ count: number; ms: number }> {
  const t0 = performance.now();
  const result = await runJxa(`
    const app = Application('Notes');
    const titles = app.notes.name();
    const created = app.notes.creationDate();
    const modified = app.notes.modificationDate();
    return JSON.stringify({
      count: titles.length,
      titles,
      created: created.map(d => d.toLocaleString()),
      modified: modified.map(d => d.toLocaleString())
    });
  `) as string;
  const ms = performance.now() - t0;
  const data = JSON.parse(result);
  return { count: data.count, ms };
}

// A2: JXA body fetch for N notes (to characterize per-note cost)
const SAMPLE_N = 10;
async function jxaBodySample(): Promise<{ count: number; ms: number }> {
  const t0 = performance.now();
  const result = await runJxa(`
    const app = Application('Notes');
    const n = ${SAMPLE_N};
    const bodies = [];
    for (let i = 0; i < n; i++) {
      bodies.push(app.notes[i].body());
    }
    return JSON.stringify({ count: n, bodies });
  `) as string;
  const ms = performance.now() - t0;
  const data = JSON.parse(result);
  return { count: data.count, ms };
}

// B: SQLite full fetch
function sqliteFull(): { count: number; ms: number } | { error: string } {
  const t0 = performance.now();
  try {
    const db = new Database(NOTES_DB, { readonly: true });
    const rows = db.query(`
      SELECT n.ZTITLE1 AS title, n.ZCREATIONDATE1 AS created,
             n.ZMODIFICATIONDATE1 AS modified, d.ZDATA AS data
      FROM ZICCLOUDSYNCINGOBJECT n
      JOIN ZICNOTEDATA d ON d.ZNOTE = n.Z_PK
      WHERE n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0
    `).all() as Array<{ title: string; created: number; modified: number; data: Uint8Array }>;
    db.close();
    const notes = rows.map(r => ({
      title: r.title,
      body: extractText(r.data),
      created: cfDate(r.created),
      modified: cfDate(r.modified),
    }));
    const ms = performance.now() - t0;
    return { count: notes.length, ms };
  } catch (e: any) {
    if (e?.message?.includes("authorization") || e?.message?.includes("denied")) {
      return { error: "Authorization denied — Full Disk Access required for SQLite" };
    }
    return { error: `SQLite error: ${e?.message}` };
  }
}

function fmt(ms: number, pad = 10) {
  return ms.toFixed(1).padStart(pad);
}

async function runN<T>(label: string, n: number, fn: () => Promise<T> | T): Promise<T[]> {
  const results: T[] = [];
  for (let i = 0; i < n; i++) {
    process.stdout.write(`  ${label} run ${i + 1}/${n}... `);
    const r = await fn();
    if (r && typeof r === "object" && "ms" in r) {
      console.log(`${(r as any).ms.toFixed(1)} ms  (${(r as any).count} notes)`);
    } else if (r && typeof r === "object" && "error" in r) {
      console.log(`ERROR: ${(r as any).error}`);
    }
    results.push(r);
  }
  return results;
}

function avg(times: number[]) {
  return times.reduce((a, b) => a + b, 0) / times.length;
}

async function main() {
  const RUNS = 3;
  console.log("=== Apple Notes Fetch Benchmark ===");
  console.log(`Runs: ${RUNS}  (run 1 = warmup; avg of runs 2-${RUNS} reported)\n`);

  // --- B: SQLite ---
  console.log("B — SQLite full fetch (title + body + dates):");
  const sqliteResults = await runN("SQLite", RUNS, sqliteFull);
  const sqliteOk = sqliteResults.filter(r => !("error" in r)) as Array<{ count: number; ms: number }>;
  const sqliteError = (sqliteResults.find(r => "error" in r) as any)?.error ?? null;
  const sqliteAvg = sqliteOk.length >= 2 ? avg(sqliteOk.slice(1).map(r => r.ms)) : sqliteOk[0]?.ms ?? null;
  const sqliteCount = sqliteOk[0]?.count ?? 0;

  // --- A1: JXA metadata ---
  console.log("\nA1 — JXA metadata bulk fetch (name + dates, no body):");
  const metaResults = await runN("JXA-meta", RUNS, jxaMetadata);
  const metaTimes = metaResults.map(r => r.ms);
  const metaAvg = metaTimes.length >= 2 ? avg(metaTimes.slice(1)) : metaTimes[0];
  const metaCount = metaResults[0].count;

  // --- A2: JXA body sample ---
  console.log(`\nA2 — JXA body fetch sample (${SAMPLE_N} notes, individual IPC calls):`);
  const bodyResults = await runN("JXA-body", RUNS, jxaBodySample);
  const bodyTimes = bodyResults.map(r => r.ms);
  const bodyAvg = bodyTimes.length >= 2 ? avg(bodyTimes.slice(1)) : bodyTimes[0];
  const bodyMsPerNote = bodyAvg / SAMPLE_N;
  const projectedFullMs = bodyMsPerNote * metaCount;

  // --- Summary ---
  console.log("\n" + "=".repeat(70));
  console.log("RESULTS SUMMARY");
  console.log("=".repeat(70));
  console.log(
    `${"Approach".padEnd(22)} ${"Notes".padStart(6)} ${"Avg ms".padStart(10)} ${"ms/note".padStart(10)} ${"Covers body?".padStart(14)}`
  );
  console.log("-".repeat(64));
  console.log(
    `${"A1 JXA metadata-only".padEnd(22)} ${String(metaCount).padStart(6)} ${fmt(metaAvg)} ${fmt(metaAvg / metaCount)} ${"No".padStart(14)}`
  );
  console.log(
    `${"A2 JXA body (sample)".padEnd(22)} ${String(SAMPLE_N).padStart(6)} ${fmt(bodyAvg)} ${fmt(bodyMsPerNote)} ${"Yes".padStart(14)}`
  );
  if (sqliteError) {
    console.log(`${"B SQLite full".padEnd(22)} ${"N/A".padStart(6)} ${"N/A".padStart(10)} ${"N/A".padStart(10)} ${"Yes".padStart(14)}`);
    console.log(`\nSQLite skipped: ${sqliteError}`);
  } else if (sqliteAvg !== null) {
    console.log(
      `${"B SQLite full".padEnd(22)} ${String(sqliteCount).padStart(6)} ${fmt(sqliteAvg)} ${fmt(sqliteAvg / sqliteCount)} ${"Yes".padStart(14)}`
    );
  }

  console.log("=".repeat(70));
  console.log("\nPROJECTIONS for full corpus (" + metaCount + " notes):");
  console.log(`  A1 JXA metadata-only:   ${metaAvg.toFixed(0)} ms  (no body text)`);
  console.log(`  A2 JXA w/ body (extrap): ${(projectedFullMs / 1000 / 60).toFixed(0)} min  (${(projectedFullMs).toFixed(0)} ms — impractical)`);
  if (sqliteAvg !== null && !sqliteError) {
    console.log(`  B  SQLite full:          ${sqliteAvg.toFixed(0)} ms  (with full body text)`);
    const ratio = projectedFullMs / sqliteAvg;
    console.log(`\nSQLite is ${ratio.toFixed(0)}x faster than JXA for full fetch with body`);
    if (metaAvg > sqliteAvg) {
      console.log(`SQLite full fetch is even faster than JXA metadata-only by ${(metaAvg / sqliteAvg).toFixed(1)}x`);
    } else {
      console.log(`JXA metadata-only is ${(sqliteAvg / metaAvg).toFixed(1)}x faster than SQLite, but doesn't include body`);
    }
  }
}

main().catch(console.error);
