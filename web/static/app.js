const logEl = document.getElementById("log");
const formEl = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendEl = document.getElementById("send");
const modelSelectEl = document.getElementById("model-select");
const maxIterEl = document.getElementById("max-iterations");
const bannerEl = document.getElementById("key-banner");

let turns = []; // {question, out}
let currentModel = null;
let turnCounter = 0;

const STATUS_LABELS = {
  research: "Đang research...",
  verify: "Đang verify...",
  answer: "Đang trả lời...",
};

function safeHref(url) {
  try {
    const parsed = new URL(url, window.location.href);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "#";
  } catch {
    return "#";
  }
}

function addBubble(role, text, extraClass) {
  const div = document.createElement("div");
  div.className = `bubble ${role}${extraClass ? " " + extraClass : ""}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function renderResult(bubble, question, out) {
  bubble.textContent = "";
  bubble.classList.remove("error", "pending");

  const turnId = `t${turnCounter++}`;
  const sources = out.sources || [];
  const sourceIndexByUrl = new Map(sources.map((s, i) => [s.url, i]));

  const eyebrow = document.createElement("div");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Finding Report";
  bubble.appendChild(eyebrow);

  const answerText = document.createElement("div");
  answerText.className = "answer-text";
  answerText.textContent = out.answer || "";
  bubble.appendChild(answerText);

  if (out.findings && out.findings.length) {
    const findings = document.createElement("div");
    findings.className = "findings";
    const heading = document.createElement("div");
    heading.className = "ledger-heading";
    heading.textContent = "Findings";
    findings.appendChild(heading);
    out.findings.forEach((f) => {
      const conf = (f.confidence || "unknown").toLowerCase();
      const row = document.createElement("div");
      row.className = `finding-row confidence-${conf}`;
      const dot = document.createElement("span");
      dot.className = "confidence-dot";
      row.appendChild(dot);
      const claim = document.createElement("span");
      claim.className = "finding-claim";
      claim.textContent = f.claim || "";
      row.appendChild(claim);
      (f.source_urls || []).forEach((url) => {
        const idx = sourceIndexByUrl.get(url);
        if (idx === undefined) return;
        const tab = document.createElement("a");
        tab.className = "citation-tab";
        tab.href = `#${turnId}-source-${idx}`;
        tab.textContent = `S${idx + 1}`;
        tab.addEventListener("click", (e) => {
          e.preventDefault();
          const target = document.getElementById(`${turnId}-source-${idx}`);
          if (!target) return;
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add("flash");
          setTimeout(() => target.classList.remove("flash"), 900);
        });
        row.appendChild(tab);
      });
      findings.appendChild(row);
    });
    bubble.appendChild(findings);
  }

  if (sources.length) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";
    const heading = document.createElement("div");
    heading.className = "ledger-heading";
    heading.textContent = "Sources";
    sourcesEl.appendChild(heading);
    sources.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.id = `${turnId}-source-${i}`;
      const num = document.createElement("span");
      num.className = "source-num";
      num.textContent = `S${i + 1}`;
      item.appendChild(num);
      const link = document.createElement("a");
      link.href = safeHref(s.url);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = s.title || s.url;
      item.appendChild(link);
      sourcesEl.appendChild(item);
    });
    bubble.appendChild(sourcesEl);
  }

  const metrics = document.createElement("div");
  metrics.className = "metrics";
  metrics.textContent =
    `iterations: ${out.iteration ?? 0} | searches: ${out.search_calls ?? 0} | ` +
    `sources: ${(out.sources || []).length} | tokens: ${out.total_tokens ?? 0} | ` +
    `est_cost: $${(out.total_cost ?? 0).toFixed(4)}`;
  bubble.appendChild(metrics);

  const download = document.createElement("button");
  download.type = "button";
  download.className = "download";
  download.textContent = "Tải report.md";
  download.addEventListener("click", () => downloadReport(question, out));
  bubble.appendChild(download);
}

async function downloadReport(question, out) {
  const resp = await fetch("/api/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, out }),
  });
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "report.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function parseSseFrame(frame) {
  const lines = frame.split("\n");
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7);
    if (line.startsWith("data: ")) data = line.slice(6);
  }
  return data ? { event, data: JSON.parse(data) } : null;
}

async function sendQuestion(question) {
  addBubble("user", question);
  const thinking = addBubble("assistant", STATUS_LABELS.research, "pending");

  const history = turns.map((t) => ({ question: t.question, answer: t.out.answer }));
  const body = {
    question,
    history,
    model: modelSelectEl.value !== currentModel ? modelSelectEl.value : null,
    max_iterations: Number(maxIterEl.value) || null,
  };

  questionEl.disabled = true;
  sendEl.disabled = true;

  try {
    let resp;
    try {
      resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      thinking.textContent = `Lỗi kết nối: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!resp.ok) {
      let detail = "";
      try {
        detail = await resp.text();
      } catch {
        // best-effort only; fall back to the status line below
      }
      thinking.textContent = `Lỗi: ${resp.status} ${resp.statusText}${detail ? " — " + detail : ""}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawTerminalEvent = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop(); // keep the last, possibly incomplete frame
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;
          if (event === "status") {
            thinking.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
          } else if (event === "result") {
            renderResult(thinking, question, data);
            turns.push({ question, out: data });
            sawTerminalEvent = true;
          } else if (event === "error") {
            thinking.textContent = `Lỗi: ${data.message}`;
            thinking.classList.remove("pending");
            thinking.classList.add("error");
            sawTerminalEvent = true;
          }
        }
      }
    } catch (err) {
      thinking.textContent = `Lỗi: ${err.message}`;
      thinking.classList.remove("pending");
      thinking.classList.add("error");
      return;
    }

    if (!sawTerminalEvent) {
      thinking.textContent = "Lỗi: kết nối bị ngắt trước khi có kết quả.";
      thinking.classList.remove("pending");
      thinking.classList.add("error");
    }
  } finally {
    questionEl.disabled = false;
    sendEl.disabled = false;
    questionEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;
  questionEl.value = "";
  sendQuestion(question);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

async function loadModels() {
  const resp = await fetch("/api/models");
  const data = await resp.json();
  currentModel = data.current;
  modelSelectEl.innerHTML = "";
  for (const choice of data.choices) {
    const opt = document.createElement("option");
    opt.value = choice;
    opt.textContent = choice;
    if (choice === data.current) opt.selected = true;
    modelSelectEl.appendChild(opt);
  }
  if (!data.key_configured) {
    bannerEl.classList.remove("hidden");
    questionEl.disabled = true;
    sendEl.disabled = true;
  }
}

loadModels();
