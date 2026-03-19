import { App } from "@modelcontextprotocol/ext-apps";

type ToolText = { type: "text"; text: string };
type ToolResult = { content?: Array<ToolText | { type: string; [k: string]: any }> };

type JobStatus = {
  jobId: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  total: number;
  message?: string;
  startedAt?: number;
  finishedAt?: number;
  error?: string;
};

type LogsResp = { jobId: string; nextOffset: number; lines: string[] };

const $ = (id: string) => document.getElementById(id) as HTMLElement;
const statusEl = $("status");
const pillEl = $("pill");
const barEl = $("bar");
const logEl = $("log");
const cancelBtn = $("cancel") as HTMLButtonElement;
const retryBtn = $("retry") as HTMLButtonElement;

const app = new App({ name: "Apple Notes Indexer", version: "1.0.0" });

let jobId: string | null = null;
let logsOffset = 0;
let pollTimer: number | null = null;
let logsTimer: number | null = null;

function fmt(n: number) {
  return new Intl.NumberFormat().format(n);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function setUI(p: number, t: number, message?: string) {
  const pct = t > 0 ? clamp01(p / t) : 0;
  barEl.style.width = `${(pct * 100).toFixed(1)}%`;
  statusEl.textContent = message ?? "";
  pillEl.textContent = `${fmt(p)} / ${fmt(t)}`;
}

function appendLogs(lines: string[]) {
  if (!lines.length) return;
  const atBottom =
    Math.abs(logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight) < 8;
  logEl.textContent += lines.join("\n") + "\n";
  if (atBottom) logEl.scrollTop = logEl.scrollHeight;
}

function stopPolling() {
  if (pollTimer) window.clearInterval(pollTimer);
  if (logsTimer) window.clearInterval(logsTimer);
  pollTimer = null;
  logsTimer = null;
}

async function pollStatusOnce() {
  if (!jobId) return;
  const res = await app.callServerTool({
    name: "index-notes-status",
    arguments: { jobId },
  });
  const txt = (res as ToolResult).content?.find((c: any) => c.type === "text")
    ?.text;
  if (!txt) return;
  const s = JSON.parse(txt) as JobStatus;

  setUI(s.progress ?? 0, s.total ?? 0, s.message);
  cancelBtn.disabled = s.status !== "running";

  if (s.status === "failed") {
    appendLogs([`[ERROR] ${s.error ?? "Unknown error"}`]);
    stopPolling();
  }
  if (s.status === "completed" || s.status === "cancelled") {
    stopPolling();
  }
}

async function pollLogsOnce() {
  if (!jobId) return;
  const res = await app.callServerTool({
    name: "index-notes-logs",
    arguments: { jobId, offset: logsOffset },
  });
  const txt = (res as ToolResult).content?.find((c: any) => c.type === "text")
    ?.text;
  if (!txt) return;
  const l = JSON.parse(txt) as LogsResp;
  logsOffset = l.nextOffset ?? logsOffset;
  appendLogs(l.lines ?? []);
}

function startPolling() {
  stopPolling();
  pollTimer = window.setInterval(() => void pollStatusOnce(), 450);
  logsTimer = window.setInterval(() => void pollLogsOnce(), 450);
  void pollStatusOnce();
  void pollLogsOnce();
}

async function startJob() {
  setUI(0, 0, "Starting…");
  pillEl.textContent = "—";
  logEl.textContent = "";
  logsOffset = 0;
  cancelBtn.disabled = true;

  const res = await app.callServerTool({ name: "index-notes", arguments: {} });
  const txt = (res as ToolResult).content?.find((c: any) => c.type === "text")
    ?.text;
  if (!txt) {
    statusEl.textContent = "Failed to start job.";
    return;
  }
  const payload = JSON.parse(txt) as { jobId?: string; message?: string };
  jobId = payload.jobId ?? null;
  if (!jobId) {
    statusEl.textContent = payload.message ?? "Failed to start job.";
    return;
  }

  appendLogs([`Started job ${jobId}`]);
  startPolling();
}

cancelBtn.addEventListener("click", async () => {
  if (!jobId) return;
  cancelBtn.disabled = true;
  await app.callServerTool({
    name: "cancel-index-notes",
    arguments: { jobId },
  });
  appendLogs(["Cancellation requested."]);
});

retryBtn.addEventListener("click", () => void startJob());

app.ontoolresult = (result: ToolResult) => {
  // When this UI is opened by the host after calling a tool, the initial tool
  // result is delivered here. If it contains a jobId, attach; otherwise start.
  const txt = result.content?.find((c: any) => c.type === "text")?.text;
  if (txt) {
    try {
      const payload = JSON.parse(txt) as { jobId?: string };
      if (payload.jobId) {
        jobId = payload.jobId;
        appendLogs([`Attached to job ${jobId}`]);
        startPolling();
        return;
      }
    } catch {
      // ignore
    }
  }
  void startJob();
};

app.connect().catch((err: unknown) => {
  statusEl.textContent = `Connection error: ${err instanceof Error ? err.message : String(err)}`;
  logEl.textContent = `app.connect() failed:\n${err instanceof Error ? err.stack ?? err.message : String(err)}`;
});

