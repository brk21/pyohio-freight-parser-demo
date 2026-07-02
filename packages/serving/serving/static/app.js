/*
 * freight-parser playground — vanilla JS client for the serving API.
 *
 * Endpoints (see the build contract):
 *   GET  /ready   -> {"status": "ready"}
 *   GET  /models  -> {"models": [ {name, base, label, generation, note,
 *                                   has_adapter, is_default} ... ],
 *                     "default": "newer"}
 *   POST /parse   body {text, guidance, model}
 *                 -> {items, model, requested_model, fell_back,
 *                     reference_guarded, duration}
 *
 * No frameworks, no CDN — everything here is plain DOM so the page works
 * offline on a demo laptop.
 */
"use strict";

// ── Element handles (IDs must match index.html) ──────────────────────────
const els = {
  confirmation: document.getElementById("confirmation"),
  guidance: document.getElementById("guidance"),
  model: document.getElementById("model"),
  parseBtn: document.getElementById("parse-btn"),

  statusLed: document.getElementById("status-led"),
  statusText: document.getElementById("status-text"),

  runInfo: document.getElementById("run-info"),
  fallbackNote: document.getElementById("fallback-note"),
  guardNote: document.getElementById("guard-note"),
  error: document.getElementById("error"),

  placeholder: document.getElementById("placeholder"),
  loading: document.getElementById("loading"),
  output: document.getElementById("output"),
  outputCode: document.getElementById("output-code"),
};

// ── Small helpers ─────────────────────────────────────────────────────────

/** Show/hide an element via the `hidden` attribute. */
function setHidden(el, hidden) {
  el.hidden = !!hidden;
}

/** Format a duration in seconds for the run-info line. */
function fmtDuration(seconds) {
  if (typeof seconds !== "number" || !isFinite(seconds)) return "?";
  return seconds.toFixed(2) + "s";
}

/**
 * Turn a JS value into pretty, HTML-escaped, lightly syntax-highlighted JSON.
 * HTML is escaped BEFORE any span markup is injected, so model output can
 * never inject live markup into the page.
 */
function highlightJSON(value) {
  let json = JSON.stringify(value, null, 2);
  json = json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return json.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    (match) => {
      let cls = "tok-num";
      if (/^"/.test(match)) {
        cls = /:$/.test(match.trim()) ? "tok-key" : "tok-str";
      } else if (/^(true|false)$/.test(match)) {
        cls = "tok-bool";
      } else if (match === "null") {
        cls = "tok-null";
      }
      return '<span class="' + cls + '">' + match + "</span>";
    }
  );
}

// ── Status LED (GET /ready) ────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const res = await fetch("/ready", { method: "GET" });
    if (!res.ok) throw new Error("not ready");
    const data = await res.json();
    const ready = data && data.status === "ready";
    els.statusLed.dataset.state = ready ? "ready" : "waiting";
    els.statusText.textContent = ready ? "ready" : "warming up…";
  } catch (err) {
    els.statusLed.dataset.state = "down";
    els.statusText.textContent = "offline";
  }
}

// ── Model dropdown (GET /models) ────────────────────────────────────────────

async function loadModels() {
  try {
    const res = await fetch("/models", { method: "GET" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const models = (data && data.models) || [];
    const defaultName = data && data.default;

    els.model.innerHTML = "";
    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.disabled = true;
      opt.selected = true;
      opt.textContent = "no models available";
      els.model.appendChild(opt);
      return;
    }

    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      let text = m.label || m.name;
      if (m.is_default) text += " (default)";
      // Annotate — but do NOT disable — untrained models, so the demo can
      // pick one and show the server's graceful fallback in action.
      if (!m.has_adapter) text += " — untrained";
      opt.textContent = text;
      if (m.note) opt.title = m.note;
      if (m.name === defaultName || (defaultName == null && m.is_default)) {
        opt.selected = true;
      }
      els.model.appendChild(opt);
    }
  } catch (err) {
    els.model.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.disabled = true;
    opt.selected = true;
    opt.textContent = "could not load models";
    els.model.appendChild(opt);
    showError("Could not load model list: " + err.message);
  }
}

// ── Output rendering ────────────────────────────────────────────────────────

function showError(message) {
  els.error.textContent = message;
  setHidden(els.error, false);
}

function clearNotes() {
  setHidden(els.error, true);
  setHidden(els.fallbackNote, true);
  setHidden(els.guardNote, true);
  setHidden(els.runInfo, true);
}

function setLoading(isLoading) {
  els.parseBtn.disabled = isLoading;
  els.parseBtn.classList.toggle("is-loading", isLoading);
  els.model.disabled = isLoading;
  if (isLoading) {
    setHidden(els.placeholder, true);
    setHidden(els.output, true);
    setHidden(els.loading, false);
  } else {
    setHidden(els.loading, true);
  }
}

/** Render a successful /parse response. */
function renderResult(data) {
  const items = Array.isArray(data.items) ? data.items : [];

  // Run info: which model actually ran + how long + item count.
  els.runInfo.textContent =
    "ran " + data.model + " · " + fmtDuration(data.duration) +
    " · " + items.length + (items.length === 1 ? " item" : " items");
  setHidden(els.runInfo, false);

  // Fallback note: requested model was untrained/unknown and got swapped.
  if (data.fell_back) {
    els.fallbackNote.textContent =
      'requested "' + data.requested_model +
      '" is untrained — fell back to "' + data.model + '"';
    setHidden(els.fallbackNote, false);
  }

  // Reference-guard note: the guard nulled a fabricated PO/BOL value.
  if (data.reference_guarded) {
    setHidden(els.guardNote, false);
  }

  els.outputCode.innerHTML = highlightJSON(items);
  setHidden(els.placeholder, true);
  setHidden(els.output, false);
}

// ── Parse action (POST /parse) ──────────────────────────────────────────────

async function onParse() {
  clearNotes();
  setLoading(true);

  const text = els.confirmation.value;
  const guidanceRaw = els.guidance.value.trim();
  const modelVal = els.model.value;

  const body = {
    text: text,
    guidance: guidanceRaw === "" ? null : guidanceRaw,
    model: modelVal === "" ? null : modelVal,
  };

  try {
    const res = await fetch("/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      // Try to surface a structured error detail if the API sent one.
      let detail = "HTTP " + res.status;
      try {
        const errBody = await res.json();
        if (errBody && errBody.detail) detail = errBody.detail;
      } catch (_) {
        /* non-JSON error body; keep the status text */
      }
      throw new Error(detail);
    }

    const data = await res.json();
    setLoading(false);
    renderResult(data);
    // The very first parse warms the model, so recheck readiness afterward.
    refreshStatus();
  } catch (err) {
    setLoading(false);
    setHidden(els.placeholder, false);
    showError("Parse failed: " + err.message);
  }
}

// ── Wire-up ──────────────────────────────────────────────────────────────────

els.parseBtn.addEventListener("click", onParse);

// Ctrl/Cmd+Enter from the textarea triggers a parse — handy during a live demo.
els.confirmation.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    onParse();
  }
});

loadModels();
refreshStatus();
