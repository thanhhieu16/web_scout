const logEl = document.getElementById("log");
const formEl = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendEl = document.getElementById("send");
const modelSelectEl = document.getElementById("model-select");
const maxIterEl = document.getElementById("max-iterations");
const bannerEl = document.getElementById("key-banner");
const newConversationEl = document.getElementById("new-conversation");
const conversationListEl = document.getElementById("conversation-list");
const themeToggleEl = document.getElementById("theme-toggle");

let currentModel = null;
let turnCounter = 0;
let activeConversationId = null;
let conversations = []; // [{id, title, updated_at}]

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

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggleEl.textContent = theme === "light" ? "☀" : "☾";
}

function initTheme() {
  const stored = localStorage.getItem("webscout-theme");
  if (stored === "light" || stored === "dark") {
    applyTheme(stored);
    return;
  }
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(prefersLight ? "light" : "dark");
}

themeToggleEl.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem("webscout-theme", next);
  applyTheme(next);
});

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

function renderConversationList() {
  conversationListEl.replaceChildren();
  conversations.forEach((c) => {
    const item = document.createElement("div");
    item.className = `conversation-item${c.id === activeConversationId ? " active" : ""}`;

    const title = document.createElement("span");
    title.className = "conversation-title";
    title.textContent = c.title;
    title.addEventListener("click", () => selectConversation(c.id));
    item.appendChild(title);

    const actions = document.createElement("span");
    actions.className = "conversation-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.textContent = "✎";
    renameBtn.title = "Đổi tên";
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      renameConversation(c.id, c.title);
    });
    actions.appendChild(renameBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.textContent = "×";
    deleteBtn.title = "Xóa";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    });
    actions.appendChild(deleteBtn);

    item.appendChild(actions);
    conversationListEl.appendChild(item);
  });
}

async function loadConversations() {
  const resp = await fetch("/api/conversations");
  conversations = await resp.json();
  renderConversationList();
}

async function createConversation() {
  const resp = await fetch("/api/conversations", { method: "POST" });
  const conv = await resp.json();
  conversations.unshift({ id: conv.id, title: conv.title, updated_at: null });
  await selectConversation(conv.id);
}

async function selectConversation(id) {
  activeConversationId = id;
  renderConversationList();
  const resp = await fetch(`/api/conversations/${id}`);
  const data = await resp.json();
  logEl.replaceChildren();
  data.messages.forEach((m) => {
    addBubble("user", m.question);
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    logEl.appendChild(bubble);
    renderResult(bubble, m.question, m.out);
  });
  logEl.scrollTop = logEl.scrollHeight;
}

async function renameConversation(id, currentTitle) {
  const next = window.prompt("Đổi tên hội thoại:", currentTitle);
  if (next === null) return;
  const title = next.trim();
  if (!title) return;
  const resp = await fetch(`/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) return;
  const updated = await resp.json();
  const conv = conversations.find((c) => c.id === id);
  if (conv) conv.title = updated.title;
  renderConversationList();
}

async function deleteConversation(id) {
  if (!window.confirm("Xóa hội thoại này?")) return;
  const resp = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (!resp.ok) return;
  conversations = conversations.filter((c) => c.id !== id);
  if (activeConversationId === id) {
    if (conversations.length) {
      await selectConversation(conversations[0].id);
    } else {
      await createConversation();
    }
  } else {
    renderConversationList();
  }
}

async function sendQuestion(question) {
  addBubble("user", question);
  const thinking = addBubble("assistant", STATUS_LABELS.research, "pending");

  const body = {
    conversation_id: activeConversationId,
    question,
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
        buffer = frames.pop();
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;
          const { event, data } = parsed;
          if (event === "status") {
            thinking.textContent = STATUS_LABELS[data.node] || `Đang ${data.node}...`;
          } else if (event === "result") {
            renderResult(thinking, question, data);
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
    } else {
      await loadConversations();
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
  if (!question || !activeConversationId) return;
  questionEl.value = "";
  sendQuestion(question);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

newConversationEl.addEventListener("click", () => {
  createConversation();
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

async function init() {
  try {
    await loadModels();
    await loadConversations();
    if (conversations.length) {
      await selectConversation(conversations[0].id);
    } else {
      await createConversation();
    }
  } catch (err) {
    bannerEl.textContent = "Không kết nối được máy chủ. Thử tải lại trang.";
    bannerEl.classList.remove("hidden");
    questionEl.disabled = true;
    sendEl.disabled = true;
  }
}

initTheme();
init();
