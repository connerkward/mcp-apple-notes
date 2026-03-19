/**
 * Test suite for mcp-apple-notes
 * Run: bun test test.ts
 *
 * Unit tests run instantly (pure functions).
 * Integration tests hit the real Apple Notes SQLite DB and LanceDB index.
 */
import { describe, it, expect, beforeAll } from "bun:test";
import { Database } from "bun:sqlite";
import path from "node:path";
import os from "node:os";
import {
  escapeForFilter,
  calcEntropy,
  filterContent,
  extractTags,
  extractWikilinks,
  extractTablesFromText,
  createNotesTable,
  indexNotes,
  searchAndCombineResults,
} from "./index";

const NOTES_DB = path.join(
  os.homedir(),
  "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"
);

const CF_EPOCH = 978307200;
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

// ─── Unit tests ───────────────────────────────────────────────────────────────

describe("escapeForFilter", () => {
  it("passes through plain strings unchanged", () => {
    expect(escapeForFilter("hello world")).toBe("hello world");
  });
  it("escapes single quotes", () => {
    expect(escapeForFilter("it's")).toBe("it\\'s");
  });
  it("escapes backslashes", () => {
    expect(escapeForFilter("a\\b")).toBe("a\\\\b");
  });
  it("escapes newlines and tabs", () => {
    expect(escapeForFilter("a\nb\tc")).toBe("a\\nb\\tc");
  });
  it("handles empty string", () => {
    expect(escapeForFilter("")).toBe("");
  });
});

describe("calcEntropy", () => {
  it("single char = 0", () => {
    expect(calcEntropy("aaaa")).toBe(0);
  });
  it("two equal chars = 1", () => {
    expect(calcEntropy("ab")).toBeCloseTo(1);
  });
  it("normal English text is low", () => {
    expect(calcEntropy("hello world this is normal text")).toBeLessThan(4.5);
  });
  it("base64-like string is high", () => {
    // Random-looking base64 with no spaces — high entropy
    const b64 = "aB3xK9mNpQ2rT7vW1yZ4cF6hJ8lE0uY5sD";
    expect(calcEntropy(b64)).toBeGreaterThan(4.0);
  });
});

describe("filterContent", () => {
  it("passes through normal text", () => {
    const text = "This is a regular note about groceries.";
    expect(filterContent(text)).toBe(text);
  });
  it("strips long base64 blobs", () => {
    // 80 chars of random-looking base64 chars with no spaces → high entropy → stripped
    const blob = "aB3xK9mNpQ2rT7vW1yZ4cF6hJ8lE0uY5sD".repeat(3).slice(0, 80);
    const result = filterContent("before " + blob + " after");
    expect(result).not.toContain(blob);
  });
  it("redacts AWS keys", () => {
    const text = "key=AKIAIOSFODNN7EXAMPLE and other stuff";
    expect(filterContent(text)).toContain("[redacted]");
  });
  it("redacts GitHub tokens", () => {
    const text = "token=ghp_" + "A".repeat(36);
    expect(filterContent(text)).toContain("[redacted]");
  });
});

describe("extractTags", () => {
  it("extracts hashtags", () => {
    expect(extractTags("Meeting #work #important today")).toEqual(["work", "important"]);
  });
  it("lowercases tags", () => {
    expect(extractTags("#Work #IMPORTANT")).toEqual(["work", "important"]);
  });
  it("ignores tags starting with digit", () => {
    expect(extractTags("#1foo")).toEqual([]);
  });
  it("returns empty for no tags", () => {
    expect(extractTags("no tags here")).toEqual([]);
  });
  it("handles tags with hyphens and underscores", () => {
    expect(extractTags("#my-tag #my_tag")).toEqual(["my-tag", "my_tag"]);
  });
});

describe("extractWikilinks", () => {
  it("extracts wikilinks", () => {
    expect(extractWikilinks("See [[Project Alpha]] and [[Meeting Notes]]")).toEqual(["Project Alpha", "Meeting Notes"]);
  });
  it("returns empty for none", () => {
    expect(extractWikilinks("no links here")).toEqual([]);
  });
  it("preserves original case", () => {
    expect(extractWikilinks("[[My Note]]")).toEqual(["My Note"]);
  });
});

describe("extractTablesFromText", () => {
  it("extracts pipe tables", () => {
    const text = "| Name | Age |\n| Alice | 30 |\n| Bob | 25 |";
    const tables = extractTablesFromText(text);
    expect(tables.length).toBe(1);
    expect(tables[0]).toEqual([["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]);
  });
  it("skips separator rows", () => {
    const text = "| Name | Age |\n|------|-----|\n| Alice | 30 |";
    const tables = extractTablesFromText(text);
    expect(tables.length).toBe(1);
    expect(tables[0].some(row => row.every(c => /^[-: ]+$/.test(c)))).toBe(false);
  });
  it("returns empty for no tables", () => {
    expect(extractTablesFromText("just normal text")).toEqual([]);
  });
  it("doesn't return single-row tables", () => {
    const text = "| only | one | row |";
    expect(extractTablesFromText(text)).toEqual([]);
  });
});

// ─── Integration tests (real Apple Notes SQLite) ──────────────────────────────

describe("SQLite — list-folders", () => {
  it("returns folders with note counts", () => {
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
      expect(rows.length).toBeGreaterThan(0);
      expect(rows[0]).toHaveProperty("path");
      expect(rows[0]).toHaveProperty("noteCount");
      console.log(`  Found ${rows.length} folders, e.g. "${rows[0].path}" (${rows[0].noteCount} notes)`);
    } finally { db.close(); }
  });
});

describe("SQLite — list-notes", () => {
  it("returns notes sorted by recency", () => {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const rows = db.query<any, []>(`
        ${FOLDER_CTE}
        SELECT n.ZTITLE1 AS title,
               datetime(n.ZMODIFICATIONDATE1 + ${CF_EPOCH}, 'unixepoch') AS modified,
               COALESCE(fp.path, '') AS folder
        FROM ZICCLOUDSYNCINGOBJECT n
        LEFT JOIN folder_path fp ON fp.id = n.ZFOLDER
        WHERE n.Z_ENT = 11 AND n.ZTITLE1 IS NOT NULL AND n.ZMARKEDFORDELETION = 0
        ORDER BY n.ZMODIFICATIONDATE1 DESC LIMIT 10
      `).all();
      expect(rows.length).toBeGreaterThan(0);
      expect(rows[0]).toHaveProperty("title");
      expect(rows[0]).toHaveProperty("modified");
      expect(rows[0]).toHaveProperty("folder");
      console.log(`  Most recent note: "${rows[0].title}" (${rows[0].modified}) in "${rows[0].folder}"`);
    } finally { db.close(); }
  });
});

describe("SQLite — check-changes logic", () => {
  it("can read current max modification date", () => {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<{ d: number | null }, []>(
        "SELECT MAX(ZMODIFICATIONDATE1) AS d FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 11 AND ZMARKEDFORDELETION = 0"
      ).get();
      expect(row).not.toBeNull();
      expect(typeof row!.d).toBe("number");
      const isoDate = new Date((row!.d! + CF_EPOCH) * 1000).toISOString();
      console.log(`  Current max mod date: ${isoDate}`);
    } finally { db.close(); }
  });
});

describe("SQLite — note count", () => {
  it("has notes in the database", () => {
    const db = new Database(NOTES_DB, { readonly: true });
    try {
      const row = db.query<{ n: number }, []>(
        "SELECT COUNT(*) AS n FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 11 AND ZMARKEDFORDELETION = 0"
      ).get();
      expect(row!.n).toBeGreaterThan(0);
      console.log(`  Total notes: ${row!.n}`);
    } finally { db.close(); }
  });
});

// ─── Index integration tests (requires existing LanceDB index) ────────────────

describe("LanceDB index — search", () => {
  let notesTable: any;
  beforeAll(async () => {
    const result = await createNotesTable();
    notesTable = result.notesTable;
  });

  it("has indexed notes", async () => {
    const count = await notesTable.countRows();
    expect(count).toBeGreaterThan(0);
    console.log(`  Indexed rows: ${count}`);
  });

  it("vector search returns results", async () => {
    const results = await notesTable.search("meeting notes", "vector").limit(5).toArray();
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty("title");
    expect(results[0]).toHaveProperty("content");
    console.log(`  Top vector result: "${results[0].title}"`);
  });

  it("FTS search returns results", async () => {
    const results = await notesTable.search("meeting", "fts", "content").limit(5).toArray();
    expect(results.length).toBeGreaterThan(0);
    console.log(`  Top FTS result: "${results[0].title}"`);
  });

  it("searchAndCombineResults returns ranked results", async () => {
    const results = await searchAndCombineResults(notesTable, "todo list", 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty("title");
    expect(results[0]).toHaveProperty("snippet");
    expect(results[0]).toHaveProperty("folder");
    console.log(`  Top combined result: "${results[0].title}" (folder: ${results[0].folder || "root"})`);
  });

  it("folder filter narrows results", async () => {
    // Get a real folder name from the DB first
    const db = new Database(NOTES_DB, { readonly: true });
    let folderName = "";
    try {
      const row = db.query<{ path: string }, []>(`
        ${FOLDER_CTE}
        SELECT fp.path FROM folder_path fp
        JOIN ZICCLOUDSYNCINGOBJECT n ON n.ZFOLDER = fp.id AND n.Z_ENT = 11 AND n.ZMARKEDFORDELETION = 0
        GROUP BY fp.path HAVING COUNT(*) > 2 LIMIT 1
      `).get();
      folderName = row?.path?.split("/").pop() ?? "";
    } finally { db.close(); }

    if (!folderName) { console.log("  Skipping — no multi-note folder found"); return; }

    const all = await searchAndCombineResults(notesTable, "note", 20);
    const filtered = await searchAndCombineResults(notesTable, "note", 20, folderName);
    expect(filtered.length).toBeLessThanOrEqual(all.length);
    console.log(`  Folder filter "${folderName}": ${all.length} → ${filtered.length} results`);
  });
});

describe("LanceDB index — list-tags", () => {
  it("extracts hashtags from indexed content", async () => {
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
    const sorted = [...tagCounts.entries()].sort(([, a], [, b]) => b - a);
    console.log(`  Found ${sorted.length} unique tags. Top 5: ${sorted.slice(0, 5).map(([t, n]) => `#${t}(${n})`).join(", ")}`);
    // It's fine if this user has no hashtags — just verify it doesn't throw
    expect(Array.isArray(sorted)).toBe(true);
  });
});
