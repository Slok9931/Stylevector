const state = {
  task: null,
  userId: null,
  mode: "free", // "free" | "benchmark"
  users: [],
};

const TASK_PROMPT_META = {
  LaMP_4: {
    label: "Article text",
    hint: "Paste or write a short news article. StyleVector will generate a personalized headline for it, in this person's headline style.",
    placeholder: "e.g. City officials announced a new plan today to expand the downtown bike lane network by 2027...",
  },
  LaMP_5: {
    label: "Abstract text",
    hint: "Paste a paper abstract. StyleVector will generate a personalized title for it, in this person's titling style.",
    placeholder: "e.g. We propose a new method for approximating frequency moments of data streams under skewed distributions...",
  },
  LaMP_7: {
    label: "Tweet text",
    hint: "Type a tweet. StyleVector will paraphrase it in this person's own writing voice.",
    placeholder: "e.g. Really excited about the new season starting next week, can't wait!",
  },
};

const el = (id) => document.getElementById(id);

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

function setStatus(text) {
  el("status-text").textContent = text;
}

// --------------------------- Prompt box sync -----------------------------

// adjust this fallback chain if your /api/users/<task> objects expose
// the held-out test query under a different key. We try the most likely
// names in order and fall back to the old "type your own" placeholder if
// none of them are present, so nothing breaks if the key differs.
function getHeldOutQuery(user) {
  if (!user) return null;
  return user.held_out_query ?? user.query_input ?? user.input ?? null;
}

// Keeps the prompt textarea in sync with the current mode + selected user.
// In benchmark mode we now SHOW the actual held-out query (read-only, so it
// can still be selected/copied but not edited) instead of just disabling
// the box with an explanatory placeholder.
function applyModeToPrompt() {
  const textarea = el("prompt-input");
  const user = state.users.find((u) => u.user_id === state.userId);

  if (state.mode === "benchmark") {
    const query = getHeldOutQuery(user);
    textarea.value = query || "";
    textarea.readOnly = true;
    textarea.disabled = false; // readOnly (not disabled) keeps it legible + selectable
    textarea.placeholder = query;
  } else {
    textarea.readOnly = false;
    textarea.disabled = false;
    textarea.value = "";
    textarea.placeholder = TASK_PROMPT_META[state.task].placeholder;
  }
}

// --------------------------- Init: tasks -----------------------------

async function initTasks() {
  const tasks = await fetchJSON("/api/tasks");
  const select = el("task-select");
  select.innerHTML = "";
  tasks.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.name;
    select.appendChild(opt);
  });
  select.addEventListener("change", () => onTaskChange(select.value));
  await onTaskChange(tasks[0].id);
}

async function onTaskChange(task) {
  state.task = task;
  const meta = TASK_PROMPT_META[task];
  el("prompt-label").textContent = meta.label;
  el("prompt-hint").textContent = meta.hint;
  el("prompt-input").placeholder = meta.placeholder;
  el("prompt-input").value = "";
  el("comparison").style.display = "none";
  el("gold-strip-wrap").innerHTML = "";

  setStatus("Loading people for this task…");
  const users = await fetchJSON(`/api/users/${task}`);
  state.users = users;

  const select = el("user-select");
  select.innerHTML = "";
  users.forEach((u) => {
    const opt = document.createElement("option");
    opt.value = u.user_id;
    opt.textContent = formatUserOptionLabel(u);
    select.appendChild(opt);
  });
  select.onchange = () => onUserChange(select.value);
  setStatus("");
  await onUserChange(users[0].user_id);
}

function formatUserOptionLabel(u) {
  const star = u.starred ? "\u2605 " : "";
  let improvPart = "";
  if (u.improvement_pct !== null && u.improvement_pct !== undefined) {
    const sign = u.improvement_pct >= 0 ? "+" : "";
    improvPart = `  [${sign}${u.improvement_pct}% ROUGE-L]`;
  }
  return `${star}${u.user_id} (${u.history_size} past writings)${improvPart}`;
}

// --------------------------- User selection -----------------------------

async function onUserChange(userId) {
  state.userId = userId;
  const user = state.users.find((u) => u.user_id === userId);
  renderUserPreview(user);
  applyModeToPrompt();
  el("comparison").style.display = "none";
  el("gold-strip-wrap").innerHTML = "";

  el("vector-stats").innerHTML = `<div class="stat"><div class="num">…</div><div class="cap">building style vector</div></div>`;
  el("influences").innerHTML = "";

  try {
    const data = await fetchJSON("/api/style_vector", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: state.task, user_id: userId }),
    });
    renderVectorStats(data);
  } catch (e) {
    el("vector-stats").innerHTML = `<div class="stat"><div class="cap">Error: ${e.message}</div></div>`;
  }
}

function renderUserPreview(user) {
  if (!user) {
    el("user-preview").innerHTML = "";
    return;
  }
  let html = `Sample of their writing: "${escapeHTML(user.preview)}…"`;
  if (user.improvement_pct !== null && user.improvement_pct !== undefined) {
    const sign = user.improvement_pct >= 0 ? "+" : "";
    const cls = user.improvement_pct >= 0 ? "score-up" : "score-down";
    const star = user.starred ? "\u2605 " : "";
    html += `<br>${star}<span class="${cls}">Benchmark: ${sign}${user.improvement_pct}% ROUGE-L</span>`
      + ` (baseline ${user.rougeL_baseline} → steered ${user.rougeL_stylevector}) in our full evaluation.`;
  }
  el("user-preview").innerHTML = html;
}

function renderVectorStats(data) {
  el("vector-stats").innerHTML = `
    <div class="stat"><div class="num">${data.vector_norm}</div><div class="cap">vector norm</div></div>
    <div class="stat"><div class="num">${data.n_history_used}</div><div class="cap">history pairs used</div></div>
    <div class="stat"><div class="num">L${data.layer} / α${data.alpha}</div><div class="cap">layer / strength</div></div>
  `;

  const infHTML = data.top_influences
    .map(
      (inf, i) => `
      <div class="influence-item">
        <div class="sim">#${i + 1} · cosine similarity to style vector: ${inf.similarity}</div>
        <div class="snippet">"${escapeHTML(inf.y_i).slice(0, 160)}${inf.y_i.length > 160 ? "…" : ""}"</div>
      </div>`
    )
    .join("");
  el("influences").innerHTML = `
    <div class="prompt-hint" style="margin-bottom:0.8rem;">Past writings that most shaped this person's style vector:</div>
    ${infHTML}
  `;
}

// --------------------------- Mode toggle -----------------------------

function initModeToggle() {
  const buttons = el("mode-toggle").querySelectorAll("button");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      applyModeToPrompt();
      el("comparison").style.display = "none";
      el("gold-strip-wrap").innerHTML = "";
    });
  });
}

// --------------------------- Generate -----------------------------

async function onGenerate() {
  const btn = el("generate-btn");
  btn.disabled = true;
  setStatus("Generating (this can take 10–40 seconds)…");
  el("comparison").style.display = "none";
  el("gold-strip-wrap").innerHTML = "";

  try {
    const payload = {
      task: state.task,
      user_id: state.userId,
      mode: state.mode,
      prompt: state.mode === "free" ? el("prompt-input").value : "",
    };
    const data = await fetchJSON("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderGeneration(data);
    setStatus("");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

function renderGeneration(data) {
  el("comparison").style.display = "grid";
  el("base-output").textContent = data.baseline_output || "(empty output)";
  el("steer-output").textContent = data.steered_output || "(empty output)";

  const baseMetaParts = [`${data.baseline_time_seconds}s`];
  const steerMetaParts = [`${data.steered_time_seconds}s`, `layer ${data.layer}, α=${data.alpha}`];

  if (data.scores) {
    const rougeBase = data.scores.baseline.rougeL;
    const rougeSteer = data.scores.steered.rougeL;
    const meteorBase = data.scores.baseline.meteor;
    const meteorSteer = data.scores.steered.meteor;

    const rougePct = rougeBase !== 0 ? ((rougeSteer - rougeBase) / rougeBase) * 100 : null;
    const meteorPct = meteorBase !== 0 ? ((meteorSteer - meteorBase) / meteorBase) * 100 : null;

    baseMetaParts.push(`ROUGE-L ${rougeBase}`, `METEOR ${meteorBase}`);

    const rougeLabel = rougePct !== null
      ? `<span class="${rougePct >= 0 ? "score-up" : "score-down"}">(${rougePct >= 0 ? "+" : ""}${rougePct.toFixed(1)}%)</span>`
      : "";
    const meteorLabel = meteorPct !== null
      ? `<span class="${meteorPct >= 0 ? "score-up" : "score-down"}">(${meteorPct >= 0 ? "+" : ""}${meteorPct.toFixed(1)}%)</span>`
      : "";

    steerMetaParts.push(
      `ROUGE-L ${rougeSteer} ${rougeLabel}`,
      `METEOR ${meteorSteer} ${meteorLabel}`
    );

    el("gold-strip-wrap").innerHTML = `
      <div class="gold-strip"><strong>Reference (ground truth):</strong> ${escapeHTML(data.gold)}</div>
    `;
  }

  // Style alignment -- shown inline, both modes, since it needs no gold
  // reference (unlike ROUGE-L/METEOR above, which only appear when a real
  // reference answer exists).
  if (data.style_alignment) {
    const { baseline, steered } = data.style_alignment;
    const delta = steered - baseline;
    const cls = delta >= 0 ? "score-up" : "score-down";
    const sign = delta >= 0 ? "+" : "";

    baseMetaParts.push(`Style Alignment ${baseline.toFixed(3)}`);
    steerMetaParts.push(
      `Style Alignment ${steered.toFixed(3)} <span class="${cls}">(${sign}${delta.toFixed(3)})</span>`
      + ` <span title="A directional signal, most reliable averaged across many examples -- can look noisy on a single generation." style="cursor:help; color:var(--text-dim);">ⓘ</span>`
    );
  }

  el("base-meta").innerHTML = baseMetaParts.map((p) => `<span>${p}</span>`).join("");
  el("steer-meta").innerHTML = steerMetaParts.map((p) => `<span>${p}</span>`).join("");
}

// --------------------------- Benchmark summary -----------------------------

async function loadSummary() {
  try {
    const data = await fetchJSON("/api/summary");
    if (!data) {
      el("summary-wrap").innerHTML = `
        <div class="summary-placeholder">
          No benchmark summary generated yet. Run <code>precompute_summary.py</code>
          after your full evaluation notebook completes to populate this panel.
        </div>`;
      return;
    }
    renderSummary(data);
  } catch (e) {
    el("summary-wrap").innerHTML = `<div class="summary-placeholder">Could not load summary: ${e.message}</div>`;
  }
}

function renderSummary(rows) {
  const body = rows
    .map((r) => {
      const cls = r.ours_improv_pct >= 0 ? "improve-pos" : "improve-neg";
      const sign = r.ours_improv_pct >= 0 ? "+" : "";
      return `<tr>
        <td>${r.task}</td><td>${r.metric}</td>
        <td>${r.ours_base}</td><td>${r.ours_stylevector}</td>
        <td class="${cls}">${sign}${r.ours_improv_pct}%</td>
        <td>${r.n}</td>
      </tr>`;
    })
    .join("");
  el("summary-wrap").innerHTML = `
    <table class="summary-table">
      <thead><tr><th>Task</th><th>Metric</th><th>Base</th><th>Steered</th><th>Improv.</th><th>n</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --------------------------- Boot -----------------------------

(async function init() {
  initModeToggle();
  el("generate-btn").addEventListener("click", onGenerate);
  await initTasks();
  await loadSummary();
})();
