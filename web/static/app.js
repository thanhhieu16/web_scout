const logEl = document.getElementById("log");
const formEl = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendEl = document.getElementById("send");
const modelSelectEl = document.getElementById("model-select");
const maxIterEl = document.getElementById("max-iterations");
const bannerEl = document.getElementById("key-banner");

let turns = []; // {question, out}
let currentModel = null;

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
  bubble.textContent = out.answer || "";
  bubble.classList.remove("error");

  if (out.sources && out.sources.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    const heading = document.createElement("strong");
    heading.textContent = "Sources";
    sources.appendChild(heading);
    out.sources.forEach((s, i) => {
      sources.appendChild(document.createElement("br"));
      const link = document.createElement("a");
      link.href = safeHref(s.url);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = `[${i + 1}] ${s.title || s.url}`;
      sources.appendChild(link);
    });
    bubble.appendChild(sources);
  }

  if (out.findings && out.findings.length) {
    const findings = document.createElement("div");
    findings.className = "findings";
    const heading = document.createElement("strong");
    heading.textContent = "Findings";
    findings.appendChild(heading);
    out.findings.forEach((f) => {
      findings.appendChild(document.createElement("br"));
      const line = document.createElement("span");
      line.textContent = `(${f.confidence || "?"}) ${f.claim || ""}`;
      findings.appendChild(line);
    });
    bubble.appendChild(findings);
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
  const thinking = addBubble("assistant", STATUS_LABELS.research);

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
            thinking.classList.add("error");
            sawTerminalEvent = true;
          }
        }
      }
    } catch (err) {
      thinking.textContent = `Lỗi: ${err.message}`;
      thinking.classList.add("error");
      return;
    }

    if (!sawTerminalEvent) {
      thinking.textContent = "Lỗi: kết nối bị ngắt trước khi có kết quả.";
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
