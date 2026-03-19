const el = (id) => document.getElementById(id);

const statusEl = el("status");
const pillEl = el("pill");
const barEl = el("bar");
const logEl = el("log");
const startBtn = el("start");
const stopBtn = el("stop");
const totalInput = el("total");
const tickInput = el("tick");

let timer = null;
let current = 0;
let total = 0;

function fmt(n) {
  return new Intl.NumberFormat().format(n);
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function setUI({ progress, total, message }) {
  const pct = total > 0 ? clamp01(progress / total) : 0;
  barEl.style.width = `${(pct * 100).toFixed(1)}%`;
  statusEl.textContent = message ?? "";
  pillEl.textContent = `${fmt(progress)} / ${fmt(total)}`;
}

function log(line) {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `[${ts}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

// This mimics the `onprogress` callback payload shape used by the MCP TS SDK.
function emitProgress(progress, total, message) {
  setUI({ progress, total, message });
  log(`${progress}/${total} ${message ?? ""}`.trim());
}

function stop() {
  if (timer) window.clearInterval(timer);
  timer = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

function start() {
  stop();
  logEl.textContent = "";

  total = Number(totalInput.value || "0");
  const tickMs = Number(tickInput.value || "0");
  if (!Number.isFinite(total) || total <= 0) return;
  if (!Number.isFinite(tickMs) || tickMs <= 0) return;

  current = 0;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  emitProgress(0, total, "Starting notes indexing…");

  timer = window.setInterval(() => {
    // Introduce some realistic variance (some notes take longer).
    const step = Math.random() < 0.06 ? 0 : 1;
    current = Math.min(total, current + step);

    if (current < total) {
      emitProgress(current, total, `Fetched ${current}/${total} notes…`);
      return;
    }

    // Two final phases similar to the server implementation.
    emitProgress(total, total, "Writing embeddings/index to database…");
    window.setTimeout(() => {
      emitProgress(total, total, "Indexing complete.");
      stop();
    }, Math.max(220, tickMs * 10));
  }, tickMs);
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

setUI({ progress: 0, total: 0, message: "Idle." });

