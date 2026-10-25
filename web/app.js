/* ==========================================================================
   Herogod — Qwen Image Edit Studio frontend
   No framework, no build step. Talks to the FastAPI backend over relative URLs.
   ========================================================================== */
(() => {
  "use strict";

  const STORAGE_KEY = "herogod.settings.v2";
  const TOKEN_KEY = "herogod.token.v2";
  const BASE_KEY = "herogod.apiBase.v2";
  const HISTORY_KEY = "herogod.prompts.v2";

  // ---------------------------------------------------------------- helpers
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "html") node.innerHTML = value;
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (key === "dataset") Object.assign(node.dataset, value);
      else node.setAttribute(key, value);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
  }

  // ------------------------------------------------------------------ state
  const state = {
    config: null,
    engine: null,
    apiBase: localStorage.getItem(BASE_KEY) || "",
    token: localStorage.getItem(TOKEN_KEY) || "",
    connected: false,
    base: null, // { file, url, width, height, name }
    refs: [], // { id, file, url }
    result: null, // { url, job }
    currentJob: null,
    polling: null,
    view: "result",
    compare: 50,
    busy: false,
    lastSeed: null,
    history: [],
    hasStored: false,
    settings: {
      steps: 20,
      trueCfg: 4,
      loraScale: 1,
      adapter: "",
      aspect: "match",
      styles: [],
      negativeOn: false,
      negative: "",
      autoChain: false,
      autoScroll: true,
    },
  };

  // -------------------------------------------------------------- persistence
  function loadSettings() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      state.hasStored = Boolean(stored);
      const raw = JSON.parse(stored || "{}");
      state.settings = { ...state.settings, ...raw };
    } catch (_) {
      /* ignore corrupt storage */
    }
    try {
      const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      if (Array.isArray(raw)) state.history = raw.slice(0, 8);
    } catch (_) {
      /* ignore */
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.settings));
    } catch (_) {
      /* storage may be full or blocked */
    }
  }

  function rememberPrompt(prompt) {
    state.history = [prompt, ...state.history.filter((item) => item !== prompt)].slice(0, 8);
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
    } catch (_) {
      /* ignore */
    }
    renderHistory();
  }

  // ----------------------------------------------------------------- network
  function apiUrl(path) {
    const base = state.apiBase.replace(/\/$/, "");
    return `${base}${path}`;
  }

  function authHeaders(extra = {}) {
    const headers = { ...extra };
    if (state.token) headers["X-Herogod-Token"] = state.token;
    return headers;
  }

  async function api(path, options = {}) {
    const { method = "GET", body, json, raw = false, timeout = 120000 } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const init = { method, headers: authHeaders(options.headers || {}), signal: controller.signal };
    if (json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(json);
    } else if (body) {
      init.body = body;
    }
    try {
      const response = await fetch(apiUrl(path), init);
      if (raw && response.ok) return response;
      let data = null;
      const type = response.headers.get("content-type") || "";
      if (type.includes("application/json")) {
        data = await response.json().catch(() => null);
      } else {
        data = await response.text().catch(() => null);
      }
      if (!response.ok) {
        const detail = (data && data.detail) || (typeof data === "string" && data) || `HTTP ${response.status}`;
        const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        error.status = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The backend did not respond in time.");
      if (error instanceof TypeError) {
        throw new Error("Cannot reach the backend. Check the Backend URL in Engine settings.");
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function imageUrl(path) {
    if (!path) return "";
    const separator = path.includes("?") ? "&" : "?";
    const withToken = state.token ? `${path}${separator}token=${encodeURIComponent(state.token)}` : path;
    return apiUrl(withToken);
  }

  // ------------------------------------------------------------------ toasts
  function toast(message, kind = "info", ttl = 5200) {
    const box = $("#toasts");
    const node = el("div", { class: `toast ${kind}`, role: "status" }, [
      el("span", { class: "t-bar" }),
      el("span", { class: "t-text", text: message }),
      el("button", { class: "t-close", type: "button", "aria-label": "Dismiss", text: "×" }),
    ]);
    const remove = () => node.remove();
    node.querySelector(".t-close").addEventListener("click", remove);
    box.append(node);
    if (ttl > 0) setTimeout(remove, ttl);
    while (box.children.length > 4) box.firstElementChild.remove();
  }

  function handleAuthError(error) {
    if (error && error.status === 401) {
      toast("Access token required. Open Engine → Connection and paste the token from the Colab cell.", "warn", 9000);
      openSettings();
      const input = $("#api-token");
      input.value = state.token;
      input.focus();
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------ canvas
  function setBaseImage(file, name) {
    if (state.base && state.base.url) URL.revokeObjectURL(state.base.url);
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      state.base = { file, url, width: image.naturalWidth, height: image.naturalHeight, name: name || file.name || "image.png" };
      $("#dropzone").hidden = true;
      $("#layers").hidden = false;
      $("#img-input").src = url;
      if (!state.result) $("#img-result").src = url;
      $("#btn-sample").blur();
      $("#btn-replace").hidden = false;
      updateStageMeta();
      updateSizeLabel();
      updateSubmitState();
    };
    image.onerror = () => toast("That image could not be read. Try a PNG, JPG or WebP.", "error");
    image.src = url;
  }

  function showResult(job) {
    state.result = { url: job.url, thumb: job.thumb_url, job };
    $("#layers").hidden = false;
    $("#dropzone").hidden = true;
    $("#img-result").src = imageUrl(job.url);
    $("#btn-save").disabled = false;
    $("#btn-reuse").disabled = false;
    $("#btn-copy-prompt").disabled = false;
    $("#btn-fullscreen").disabled = false;
    const size = job.settings && job.settings.size ? `${job.settings.size[0]}×${job.settings.size[1]}` : "";
    $("#seed-readout").textContent = [
      job.seed !== null && job.seed !== undefined ? `seed ${job.seed}` : "",
      size,
      job.meta && job.meta.seconds ? formatDuration(job.meta.seconds) : "",
      job.adapter || "",
    ]
      .filter(Boolean)
      .join("  ·  ");
    state.lastSeed = job.seed;
    updateStageMeta();
    if (state.settings.autoScroll) {
      $("#stage").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function clearResult() {
    state.result = null;
    $("#img-result").removeAttribute("src");
    $("#btn-save").disabled = true;
    $("#btn-reuse").disabled = true;
    $("#btn-copy-prompt").disabled = true;
    $("#btn-fullscreen").disabled = true;
    $("#seed-readout").textContent = "";
  }

  function updateStageMeta() {
    const parts = [];
    if (state.base) parts.push(`${state.base.width}×${state.base.height} in`);
    if (state.result && state.result.job) {
      const job = state.result.job;
      const size = job.settings && job.settings.size ? job.settings.size : null;
      if (size) parts.push(`${size[0]}×${size[1]} out`);
      if (job.meta && job.meta.steps) parts.push(`${job.meta.steps} steps`);
      if (job.meta && job.meta.device) parts.push(job.meta.device);
    }
    $("#stage-meta").textContent = parts.join("  ·  ");
  }

  function setView(view) {
    state.view = view;
    $("#canvas").dataset.view = view;
    $$(".seg").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
    const compare = $("#compare");
    const canCompare = Boolean(state.result && state.base);
    compare.hidden = !canCompare;
    if (view === "compare" && canCompare) {
      $("#img-before").src = state.base.url;
      setCompare(state.compare);
    }
    if (view === "compare" && !canCompare) {
      toast("Compare needs an input image and a finished result.", "info");
      setView("result");
    }
  }

  function setCompare(percent) {
    state.compare = clamp(percent, 0, 100);
    const handle = $("#compare-handle");
    handle.style.left = `${state.compare}%`;
    handle.setAttribute("aria-valuenow", String(Math.round(state.compare)));
    $("#compare-clip").style.clipPath = `inset(0 ${100 - state.compare}% 0 0)`;
  }

  function setVeil(active, text) {
    const veil = $("#result-veil");
    veil.classList.toggle("is-on", active);
    if (text) $("#veil-text").textContent = text;
    if (!active) $("#veil-bar-fill").style.width = "0%";
  }

  // ------------------------------------------------------------------ refs
  function addRefs(files) {
    const max = state.config ? state.config.limits.max_refs : 3;
    const images = Array.from(files).filter((file) => /^image\/(png|jpeg|webp)$/.test(file.type));
    if (!images.length) {
      toast("Reference images must be PNG, JPG or WebP.", "error");
      return;
    }
    for (const file of images.slice(0, max - state.refs.length)) {
      state.refs.push({ id: `ref-${Date.now()}-${Math.random().toString(16).slice(2, 7)}`, file, url: URL.createObjectURL(file) });
    }
    if (state.refs.length >= max) toast(`Up to ${max} reference images.`, "info");
    renderRefs();
  }

  function renderRefs() {
    const container = $("#refs");
    $$(".ref", container).forEach((node) => node.remove());
    for (const ref of state.refs) {
      const item = el("div", { class: "ref", title: ref.file.name }, [
        el("img", { src: ref.url, alt: ref.file.name }),
        el("button", { class: "ref-remove", type: "button", "aria-label": `Remove ${ref.file.name}`, text: "×" }),
      ]);
      item.querySelector(".ref-remove").addEventListener("click", () => {
        URL.revokeObjectURL(ref.url);
        state.refs = state.refs.filter((entry) => entry.id !== ref.id);
        renderRefs();
      });
      container.insertBefore(item, $("#ref-add"));
    }
    const max = state.config ? state.config.limits.max_refs : 3;
    $("#ref-add").hidden = state.refs.length >= max;
    $("#refs-note").textContent = `${state.refs.length} / ${max} · optional`;
  }

  // ------------------------------------------------------------ static lists
  function renderStyles() {
    const container = $("#style-chips");
    container.textContent = "";
    for (const style of (state.config && state.config.styles) || []) {
      const chip = el("button", {
        class: "style-chip",
        type: "button",
        "aria-pressed": "false",
        title: style.prompt ? `${style.prompt}${style.negative ? `\nAvoid: ${style.negative}` : ""}` : "No extra prompt fragment",
        text: style.label,
      });
      chip.dataset.style = style.id;
      chip.addEventListener("click", () => toggleStyle(style.id));
      container.append(chip);
    }
    syncStyleChips();
  }

  function toggleStyle(id) {
    const selected = state.settings.styles;
    if (selected.includes(id)) {
      state.settings.styles = selected.filter((item) => item !== id);
    } else if (id === "none") {
      state.settings.styles = [];
    } else {
      if (selected.length >= 4) {
        toast("Four styles are the maximum. Deselect one first.", "warn");
        return;
      }
      state.settings.styles = [...selected.filter((item) => item !== "none"), id];
    }
    syncStyleChips();
    saveSettings();
  }

  function syncStyleChips() {
    $$(".style-chip").forEach((chip) => {
      chip.setAttribute("aria-pressed", String(state.settings.styles.includes(chip.dataset.style)));
    });
  }

  function renderAspects() {
    const container = $("#aspect-grid");
    container.textContent = "";
    const aspects = (state.config && state.config.aspects) || [];
    for (const aspect of aspects) {
      const ratio = aspect.ratio || 1;
      const scale = 16 / Math.max(1, ratio);
      const box = el("i");
      box.style.width = `${Math.round(ratio > 1 ? 18 : 18 * ratio)}px`;
      box.style.height = `${Math.round(ratio > 1 ? 18 / ratio : 18)}px`;
      if (!aspect.ratio) {
        box.style.width = "16px";
        box.style.height = "16px";
        box.style.borderStyle = "dashed";
      }
      void scale;
      const button = el("button", { class: "aspect-btn", type: "button", "aria-pressed": "false" }, [box, aspect.label]);
      button.dataset.aspect = aspect.id;
      button.addEventListener("click", () => {
        state.settings.aspect = aspect.id;
        syncAspects();
        updateSizeLabel();
        saveSettings();
      });
      container.append(button);
    }
    syncAspects();
  }

  function syncAspects() {
    $$(".aspect-btn").forEach((button) => {
      const active = button.dataset.aspect === state.settings.aspect;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function renderPerf() {
    const presets = (state.config && state.config.performance) || [];
    for (const button of $$(".perf-btn")) {
      const preset = presets.find((item) => item.id === button.dataset.perf);
      if (!preset) continue;
      button.querySelector("small").textContent = `${preset.steps} steps`;
      if (button.dataset.wired === "1") continue;
      button.dataset.wired = "1";
      button.addEventListener("click", () => {
        state.settings.steps = preset.steps;
        state.settings.trueCfg = preset.true_cfg;
        syncControls();
        saveSettings();
      });
    }
    syncPerfActive();
  }

  function syncPerfActive() {
    const presets = (state.config && state.config.performance) || [];
    for (const button of $$(".perf-btn")) {
      const preset = presets.find((item) => item.id === button.dataset.perf);
      button.classList.toggle(
        "is-active",
        Boolean(preset && preset.steps === Number(state.settings.steps) && preset.true_cfg === Number(state.settings.trueCfg)),
      );
    }
  }

  function renderAdapters() {
    const select = $("#adapter");
    const adapters = (state.engine && state.engine.adapters) || [];
    select.textContent = "";
    if (!adapters.length) {
      select.append(el("option", { value: "", text: "Default adapter" }));
      return;
    }
    for (const adapter of adapters) {
      select.append(el("option", { value: adapter.name, text: `${adapter.label}${adapter.loaded ? " · loaded" : ""}` }));
    }
    const wanted = state.settings.adapter || adapters[0].name;
    select.value = adapters.some((item) => item.name === wanted) ? wanted : adapters[0].name;
    state.settings.adapter = select.value;
    $("#adapter-hint").textContent = `${adapters.find((item) => item.name === select.value).repo}`;
    $("#lora-hint").textContent = `Adapter: ${adapters.find((item) => item.name === select.value).label}`;
  }

  function renderHistory() {
    const wrap = $("#prompt-history");
    const container = $("#history-chips");
    container.textContent = "";
    if (!state.history.length) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    for (const prompt of state.history.slice(0, 5)) {
      const chip = el("button", { class: "history-chip", type: "button", title: prompt, text: prompt });
      chip.addEventListener("click", () => {
        $("#prompt").value = prompt;
        $("#prompt").dispatchEvent(new Event("input"));
        $("#prompt").focus();
      });
      container.append(chip);
    }
  }

  // ----------------------------------------------------------------- controls
  function setRangeFill(input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const value = Number(input.value);
    const percent = ((value - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${clamp(percent, 0, 100)}%`);
  }

  function syncControls() {
    const settings = state.settings;
    $("#steps").value = String(settings.steps);
    $("#true-cfg").value = String(settings.trueCfg);
    $("#lora-scale").value = String(settings.loraScale);
    $("#steps-out").textContent = String(settings.steps);
    $("#true-cfg-out").textContent = Number(settings.trueCfg).toFixed(1);
    $("#lora-scale-out").textContent = Number(settings.loraScale).toFixed(2);
    for (const id of ["#steps", "#true-cfg", "#lora-scale"]) setRangeFill($(id));
    $("#negative").value = settings.negative || "";
    $("#negative-wrap").hidden = !settings.negativeOn;
    $("#btn-negative-toggle").setAttribute("aria-pressed", String(settings.negativeOn));
    syncStyleChips();
    syncAspects();
    syncPerfActive();
    updateSizeLabel();
    updateSubmitState();
  }

  function aspectRatio() {
    if (state.settings.aspect === "match" || !state.settings.aspect) {
      if (state.base) return state.base.width / state.base.height;
      return 1;
    }
    const aspects = (state.config && state.config.aspects) || [];
    const preset = aspects.find((item) => item.id === state.settings.aspect);
    return preset && preset.ratio ? preset.ratio : 1;
  }

  function computeSize() {
    const budget = (state.config && state.config.limits.max_pixels) || 1024 * 1024;
    const ratio = aspectRatio();
    const multiple = 8;
    let width = Math.sqrt(budget * ratio);
    let height = width / ratio;
    width = Math.max(multiple * 4, Math.round(width / multiple) * multiple);
    height = Math.max(multiple * 4, Math.round(height / multiple) * multiple);
    const scale = Math.min(1, Math.sqrt(budget / (width * height)));
    // floor so the rounded pair never exceeds the server's pixel budget
    width = Math.max(multiple * 4, Math.floor((width * scale) / multiple) * multiple);
    height = Math.max(multiple * 4, Math.floor((height * scale) / multiple) * multiple);
    return [width, height];
  }

  function updateSizeLabel() {
    const [width, height] = computeSize();
    $("#size-out").textContent = `${width} × ${height}`;
  }

  function updateSubmitState() {
    const button = $("#btn-generate");
    const ready = Boolean(state.base) && $("#prompt").value.trim().length > 0 && !state.busy;
    button.disabled = !ready && !state.busy;
    const hint = $("#submit-hint");
    if (state.busy) hint.textContent = "Working… press Stop to cancel.";
    else if (!state.base) hint.textContent = "Add an image and describe the edit.";
    else if (!$("#prompt").value.trim()) hint.textContent = "Describe the edit you want.";
    else hint.textContent = "⌘/Ctrl + Enter also generates.";
  }

  function setBusy(active) {
    state.busy = active;
    const button = $("#btn-generate");
    button.classList.toggle("is-busy", active);
    $("#btn-cancel").hidden = !active;
    $("#gen-progress").textContent = active ? "Starting…" : "";
    if (!active) setVeil(false);
    updateSubmitState();
  }

  // ------------------------------------------------------------------- engine
  function renderEngine() {
    const engine = state.engine;
    if (!engine) return;
    const dot = $("#engine-dot");
    const label = $("#engine-label");
    const device = $("#engine-device");
    const stateName = engine.state || "unknown";
    dot.dataset.state = stateName === "ready" ? "ready" : stateName === "loading" ? "loading" : stateName === "error" ? "error" : "idle";
    const names = {
      ready: "Model ready",
      loading: "Loading model…",
      not_loaded: "Model idle",
      error: "Model error",
      unknown: "Unknown",
    };
    label.textContent = names[stateName] || stateName;
    device.textContent = engine.gpu || (engine.device && engine.device !== "none" ? engine.device : "");
    $("#engine-chip").title = engine.detail || "";

    const kv = $("#engine-kv");
    kv.textContent = "";
    const rows = [
      ["State", engine.state],
      ["Base model", engine.base_model],
      ["Adapter", engine.active_adapter || "—"],
      ["LoRA scale", engine.lora_scale !== null && engine.lora_scale !== undefined ? Number(engine.lora_scale).toFixed(2) : "—"],
      ["GPU", engine.gpu || "CPU only"],
      ["Precision", engine.dtype],
      ["Quantisation", engine.quant],
      ["Offload", engine.offload],
      ["Load time", engine.load_seconds ? `${engine.load_seconds}s` : "—"],
    ];
    for (const [key, value] of rows) {
      kv.append(el("dt", { text: key }), el("dd", { text: String(value ?? "—") }));
    }
    renderAdapters();
  }

  async function refreshEngine() {
    try {
      state.engine = await api("/api/engine");
      state.connected = true;
      renderEngine();
    } catch (error) {
      if (handleAuthError(error)) return;
      $("#engine-dot").dataset.state = "offline";
      $("#engine-label").textContent = "Backend offline";
      $("#engine-device").textContent = "";
      $("#engine-chip").title = error.message;
    }
  }

  // ------------------------------------------------------------------ gallery
  async function refreshGallery() {
    let data;
    try {
      data = await api("/api/jobs");
    } catch (error) {
      handleAuthError(error);
      return;
    }
    const grid = $("#gallery-grid");
    grid.textContent = "";
    const jobs = (data && data.jobs) || [];
    $("#gallery-count").textContent = String(jobs.filter((job) => job.status === "done").length);
    $("#gallery-empty").hidden = jobs.length > 0;
    for (const job of jobs) {
      const item = el("button", { class: "gallery-item", type: "button", title: job.prompt });
      if (job.status === "done" && job.thumb_url) {
        item.append(el("img", { src: imageUrl(job.thumb_url), alt: job.prompt.slice(0, 90), loading: "lazy" }));
      } else {
        item.append(el("div", { class: "gi-meta", text: job.status }));
      }
      item.append(el("span", { class: `gi-state ${job.status}`, text: job.status.toUpperCase() }));
      item.append(el("div", { class: "gi-meta", text: `${job.status === "done" ? "seed " + job.seed : job.prompt.slice(0, 60)}` }));
      item.addEventListener("click", async () => {
        if (job.status !== "done") {
          toast("That job has no finished image yet.", "info");
          return;
        }
        showResult(job);
        if (job.settings && job.settings.seed !== undefined) {
          $("#seed").value = String(job.settings.seed);
          $("#seed-out").textContent = String(job.settings.seed);
        }
        closeDrawers();
        setView("result");
      });
      grid.append(item);
    }
  }

  // ------------------------------------------------------------------- jobs
  async function waitForJob(jobId) {
    let delay = 600;
    for (;;) {
      await sleep(delay);
      delay = Math.min(2500, Math.round(delay * 1.15));
      let job;
      try {
        job = await api(`/api/jobs/${jobId}`);
      } catch (error) {
        if (handleAuthError(error)) continue;
        throw error;
      }
      state.currentJob = job;
      if (job.status === "running" || job.status === "queued") {
        const percent = Math.round((job.progress || 0) * 100);
        $("#gen-progress").textContent = job.steps ? `Step ${job.step}/${job.steps}` : "Loading…";
        setVeil(true, job.steps ? `Step ${job.step} / ${job.steps}` : "Loading model…");
        $("#veil-bar-fill").style.width = `${percent}%`;
        continue;
      }
      return job;
    }
  }

  async function submit(event) {
    if (event) event.preventDefault();
    if (state.busy) return;
    const prompt = $("#prompt").value.trim();
    if (!state.base) {
      toast("Add a base image first.", "warn");
      return;
    }
    if (!prompt) {
      toast("Describe the edit you want.", "warn");
      $("#prompt").focus();
      return;
    }

    const form = new FormData();
    form.append("image", state.base.file, state.base.name);
    for (const ref of state.refs) form.append("refs", ref.file, ref.file.name);
    form.append("prompt", prompt);
    form.append("negative", state.settings.negativeOn ? state.settings.negative || "" : "");
    form.append("styles", JSON.stringify(state.settings.styles));
    form.append("steps", String(state.settings.steps));
    form.append("true_cfg", String(state.settings.trueCfg));
    form.append("lora_scale", String(state.settings.loraScale));
    form.append("adapter", state.settings.adapter || "");
    form.append("seed", String($("#seed").value || -1));
    form.append("aspect", state.settings.aspect || "match");

    setBusy(true);
    setVeil(true, "Uploading…");
    $("#gen-progress").textContent = "Queued…";
    try {
      const job = await api("/api/jobs", { method: "POST", body: form, timeout: 180000 });
      rememberPrompt(prompt);
      setVeil(true, "Loading model…");
      const finished = await waitForJob(job.id);
      if (finished.status === "done") {
        showResult(finished);
        toast(`Done in ${formatDuration(finished.elapsed)} · seed ${finished.seed}`, "ok");
        $("#seed").value = String(finished.seed);
        $("#seed-out").textContent = String(finished.seed);
        state.lastSeed = finished.seed;
        if (state.settings.autoChain) await useResultAsInput(true);
        refreshGallery();
        refreshEngine();
      } else if (finished.status === "cancelled") {
        toast("Generation cancelled.", "warn");
        setVeil(false);
      } else {
        toast(finished.error || "Generation failed.", "error", 12000);
        setVeil(false);
      }
    } catch (error) {
      if (!handleAuthError(error)) toast(error.message, "error", 12000);
      setVeil(false);
    } finally {
      setBusy(false);
      refreshGallery();
    }
  }

  async function cancelJob() {
    if (!state.currentJob) return;
    try {
      await api(`/api/jobs/${state.currentJob.id}/cancel`, { method: "POST" });
      toast("Cancel requested. The current step will finish first.", "info");
    } catch (error) {
      if (!handleAuthError(error)) toast(error.message, "error");
    }
  }

  async function useResultAsInput(silent = false) {
    if (!state.result) return;
    try {
      const response = await api(state.result.url, { raw: true, timeout: 60000 });
      const blob = await response.blob();
      const file = new File([blob], `herogod-${state.result.job.seed || "result"}.png`, { type: "image/png" });
      setBaseImage(file, file.name);
      if (!silent) toast("Result is now the input image.", "ok");
    } catch (error) {
      if (!handleAuthError(error)) toast(`Could not reuse the result: ${error.message}`, "error");
    }
  }

  async function saveResult() {
    if (!state.result) return;
    try {
      const response = await api(state.result.url, { raw: true, timeout: 60000 });
      const blob = await response.blob();
      const link = el("a", { href: URL.createObjectURL(blob), download: `herogod-${state.result.job.seed || Date.now()}.png` });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 4000);
    } catch (error) {
      if (!handleAuthError(error)) toast(`Download failed: ${error.message}`, "error");
    }
  }

  function openLightbox() {
    if (!state.result) return;
    $("#lightbox-img").src = imageUrl(state.result.url);
    $("#lightbox").showModal();
  }

  async function copyPrompt() {
    if (!state.result) return;
    try {
      await navigator.clipboard.writeText(state.result.job.prompt);
      toast("Prompt copied.", "ok", 2200);
    } catch (_) {
      toast("Clipboard is blocked by the browser.", "warn");
    }
  }

  // ------------------------------------------------------------------ drawers
  function openDrawer(id) {
    closeDrawers();
    const drawer = $(id);
    drawer.hidden = false;
    if (id === "#gallery") {
      $("#btn-gallery").setAttribute("aria-expanded", "true");
      refreshGallery();
    }
    if (id === "#settings") {
      $("#api-token").value = state.token;
      $("#api-base").value = state.apiBase;
      renderEngine();
    }
  }

  function closeDrawers() {
    for (const id of ["#gallery", "#settings"]) {
      $(id).hidden = true;
    }
    $("#btn-gallery").setAttribute("aria-expanded", "false");
  }

  function openSettings() {
    openDrawer("#settings");
  }

  // -------------------------------------------------------------- connection
  async function connect() {
    const token = $("#api-token").value.trim();
    let base = $("#api-base").value.trim();
    base = base.replace(/\/$/, "");
    if (base && !/^https?:\/\//.test(base)) {
      toast("The backend URL must start with http:// or https://.", "error");
      return;
    }
    if (base && base === window.location.origin) base = "";
    state.token = token;
    state.apiBase = base;
    localStorage.setItem(TOKEN_KEY, token);
    if (base) localStorage.setItem(BASE_KEY, base);
    else localStorage.removeItem(BASE_KEY);
    const button = $("#btn-connect");
    button.disabled = true;
    button.textContent = "Connecting…";
    try {
      await bootstrap(true);
      toast("Connected.", "ok", 2600);
      closeDrawers();
    } catch (error) {
      toast(error.message, "error", 9000);
    } finally {
      button.disabled = false;
      button.textContent = "Reconnect";
    }
  }

  async function bootstrap(reconnecting = false) {
    try {
      const health = await api("/api/health", { timeout: 12000 });
      $("#mock-badge").hidden = !health.mock && !(state.config && state.config.mock);
      state.config = await api("/api/config");
      renderStyles();
      renderAspects();
      renderPerf();
      if (state.config.defaults && !state.hasStored) {
        state.settings.steps = state.config.defaults.steps;
        state.settings.trueCfg = state.config.defaults.true_cfg;
        state.settings.loraScale = state.config.defaults.lora_scale;
        state.settings.aspect = state.config.defaults.size_mode || "match";
        state.hasStored = true;
        saveSettings();
      }
      void reconnecting;
      await refreshEngine();
      if (state.config.notice) $("#safety-copy").textContent = state.config.notice.summary;
      state.connected = true;
      syncControls();
      refreshGallery();
      $("#prompt-hint").textContent = state.config.mock ? "mock engine" : "";
    } catch (error) {
      if (error.status === 401) {
        handleAuthError(error);
        throw new Error("The access token was rejected.");
      }
      throw new Error(error.message);
    }
  }

  // ---------------------------------------------------------------- listeners
  function wireCanvasEvents() {
    const canvas = $("#canvas");
    const dropzone = $("#dropzone");

    const block = (event) => {
      event.preventDefault();
      event.stopPropagation();
    };
    for (const target of [canvas, dropzone, document.body]) {
      target.addEventListener("dragover", (event) => {
        block(event);
        canvas.classList.add("is-dragover");
      });
      target.addEventListener("dragleave", (event) => {
        block(event);
        if (event.target === canvas || event.target === dropzone) canvas.classList.remove("is-dragover");
      });
      target.addEventListener("drop", (event) => {
        block(event);
        canvas.classList.remove("is-dragover");
        const files = event.dataTransfer && event.dataTransfer.files;
        if (!files || !files.length) return;
        handleDroppedFiles(files);
      });
    }

    document.addEventListener("paste", (event) => {
      const items = (event.clipboardData && event.clipboardData.files) || [];
      if (!items.length) return;
      handleDroppedFiles(items);
    });
  }

  function handleDroppedFiles(files) {
    const images = Array.from(files).filter((file) => /^image\//.test(file.type));
    if (!images.length) {
      toast("That file is not an image. Use PNG, JPG or WebP.", "warn");
      return;
    }
    if (!state.base) {
      setBaseImage(images[0], images[0].name);
      if (images.length > 1) addRefs(images.slice(1));
      $("#dropzone").hidden = true;
    } else {
      addRefs(images);
    }
  }

  function wireCompare() {
    const handle = $("#compare-handle");
    const clip = $("#compare-clip");
    let dragging = false;

    const positionFromEvent = (event) => {
      const rect = clip.getBoundingClientRect();
      const x = (event.touches ? event.touches[0].clientX : event.clientX) - rect.left;
      return (x / rect.width) * 100;
    };

    const start = (event) => {
      dragging = true;
      setCompare(positionFromEvent(event));
      event.preventDefault();
    };
    const move = (event) => {
      if (!dragging) return;
      setCompare(positionFromEvent(event));
      event.preventDefault();
    };
    const end = () => {
      dragging = false;
    };

    handle.addEventListener("pointerdown", start);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
    clip.addEventListener("pointerdown", start);
    handle.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") setCompare(state.compare - 2);
      if (event.key === "ArrowRight") setCompare(state.compare + 2);
      if (event.key === "Home") setCompare(0);
      if (event.key === "End") setCompare(100);
    });
  }

  function makeSampleImage() {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");
    const background = ctx.createLinearGradient(0, 0, 1024, 1024);
    background.addColorStop(0, "#1b1f3b");
    background.addColorStop(0.55, "#3b1f42");
    background.addColorStop(1, "#12141c");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, 1024, 1024);

    const glow = ctx.createRadialGradient(520, 380, 40, 520, 420, 460);
    glow.addColorStop(0, "rgba(255,140,105,0.85)");
    glow.addColorStop(0.4, "rgba(255,106,77,0.28)");
    glow.addColorStop(1, "rgba(255,106,77,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, 1024, 1024);

    ctx.fillStyle = "rgba(12,14,20,0.72)";
    ctx.beginPath();
    ctx.arc(512, 470, 190, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(512, 300);
    ctx.lineTo(700, 640);
    ctx.lineTo(324, 640);
    ctx.closePath();
    ctx.fillStyle = "rgba(143,123,255,0.35)";
    ctx.fill();

    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.font = "600 34px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Herogod sample", 512, 830);
    ctx.font = "400 22px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.fillText("replace this placeholder with your own photo", 512, 870);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        const file = new File([blob], "herogod-sample.png", { type: "image/png" });
        resolve(file);
      }, "image/png");
    });
  }

  function wireControls() {
    const prompt = $("#prompt");
    prompt.addEventListener("input", () => {
      $("#prompt-count").textContent = `${prompt.value.length} / 2000`;
      updateSubmitState();
    });

    $("#btn-clear-prompt").addEventListener("click", () => {
      prompt.value = "";
      prompt.dispatchEvent(new Event("input"));
      prompt.focus();
    });

    $("#btn-negative-toggle").addEventListener("click", () => {
      state.settings.negativeOn = !state.settings.negativeOn;
      saveSettings();
      syncControls();
      if (state.settings.negativeOn) $("#negative").focus();
    });

    $("#negative").addEventListener("input", (event) => {
      state.settings.negative = event.target.value;
      saveSettings();
    });

    $("#steps").addEventListener("input", (event) => {
      state.settings.steps = Number(event.target.value);
      $("#steps-out").textContent = String(state.settings.steps);
      setRangeFill(event.target);
      syncPerfActive();
      saveSettings();
    });

    $("#true-cfg").addEventListener("input", (event) => {
      state.settings.trueCfg = Number(event.target.value);
      $("#true-cfg-out").textContent = state.settings.trueCfg.toFixed(1);
      setRangeFill(event.target);
      syncPerfActive();
      saveSettings();
    });

    $("#lora-scale").addEventListener("input", (event) => {
      state.settings.loraScale = Number(event.target.value);
      $("#lora-scale-out").textContent = state.settings.loraScale.toFixed(2);
      setRangeFill(event.target);
      saveSettings();
    });

    $("#seed").addEventListener("input", (event) => {
      $("#seed-out").textContent = event.target.value === "-1" || event.target.value === "" ? "random" : event.target.value;
    });

    $("#btn-seed-random").addEventListener("click", () => {
      $("#seed").value = "-1";
      $("#seed-out").textContent = "random";
    });

    $("#btn-seed-last").addEventListener("click", () => {
      const seed = state.lastSeed ?? (state.result && state.result.job.seed);
      if (seed === null || seed === undefined) {
        toast("No seed has been used yet.", "info", 2600);
        return;
      }
      $("#seed").value = String(seed);
      $("#seed-out").textContent = String(seed);
    });

    $("#adapter").addEventListener("change", (event) => {
      state.settings.adapter = event.target.value;
      const adapters = (state.engine && state.engine.adapters) || [];
      const adapter = adapters.find((item) => item.name === event.target.value);
      if (adapter) {
        $("#adapter-hint").textContent = adapter.repo;
        $("#lora-hint").textContent = `Adapter: ${adapter.label}`;
      }
      saveSettings();
    });

    $("#btn-reset-settings").addEventListener("click", () => {
      state.settings.steps = 20;
      state.settings.trueCfg = 4;
      state.settings.loraScale = 1;
      state.settings.aspect = "match";
      state.settings.styles = [];
      state.settings.negativeOn = false;
      state.settings.negative = "";
      saveSettings();
      syncControls();
      toast("Settings reset to defaults.", "ok", 2400);
    });

    $("#edit-form").addEventListener("submit", submit);
    $("#btn-cancel").addEventListener("click", cancelJob);
    $("#btn-pick").addEventListener("click", () => $("#file-input").click());
    $("#btn-replace").addEventListener("click", () => $("#file-input").click());
    $("#ref-add").addEventListener("click", () => $("#ref-input").click());

    $("#file-input").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) setBaseImage(file, file.name);
      event.target.value = "";
    });

    $("#ref-input").addEventListener("change", (event) => {
      if (event.target.files) addRefs(event.target.files);
      event.target.value = "";
    });

    $("#btn-sample").addEventListener("click", async () => {
      const file = await makeSampleImage();
      setBaseImage(file, file.name);
      if (!$("#prompt").value.trim()) {
        $("#prompt").value = "make it a cinematic night portrait, warm rim light, shallow depth of field";
        $("#prompt").dispatchEvent(new Event("input"));
      }
    });

    $$(".seg").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    $("#btn-save").addEventListener("click", saveResult);
    $("#btn-reuse").addEventListener("click", () => useResultAsInput(false));
    $("#btn-copy-prompt").addEventListener("click", copyPrompt);
    $("#btn-fullscreen").addEventListener("click", openLightbox);

    $("#btn-gallery").addEventListener("click", () => {
      if ($("#gallery").hidden) openDrawer("#gallery");
      else closeDrawers();
    });
    $("#btn-gallery-close").addEventListener("click", closeDrawers);
    $("#btn-gallery-clear").addEventListener("click", async () => {
      try {
        const result = await api("/api/jobs", { method: "DELETE" });
        toast(`Removed ${result.removed} finished job${result.removed === 1 ? "" : "s"}.`, "ok", 2600);
        refreshGallery();
      } catch (error) {
        if (!handleAuthError(error)) toast(error.message, "error");
      }
    });

    $("#btn-settings").addEventListener("click", () => {
      if ($("#settings").hidden) openSettings();
      else closeDrawers();
    });
    $("#btn-settings-close").addEventListener("click", closeDrawers);
    $("#btn-connect").addEventListener("click", connect);

    $("#btn-engine-load").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      const original = button.textContent;
      button.disabled = true;
      button.textContent = "Loading…";
      toast("Loading the model. This can take several minutes the first time.", "info", 8000);
      try {
        state.engine = await api("/api/engine/load", { method: "POST", timeout: 3600000 });
        renderEngine();
        toast("Model loaded.", "ok");
      } catch (error) {
        if (!handleAuthError(error)) toast(error.message, "error", 12000);
      } finally {
        button.disabled = false;
        button.textContent = original;
        refreshEngine();
      }
    });

    $("#btn-engine-unload").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const result = await api("/api/engine/unload", { method: "POST", timeout: 120000 });
        state.engine = result.status || null;
        renderEngine();
        toast(result.unloaded ? "Model unloaded and VRAM released." : "Nothing was loaded.", "ok");
      } catch (error) {
        if (!handleAuthError(error)) toast(error.message, "error");
      } finally {
        button.disabled = false;
        refreshEngine();
      }
    });

    $("#btn-help").addEventListener("click", () => $("#help-dialog").showModal());

    $("#lightbox").addEventListener("click", (event) => {
      if (event.target === $("#lightbox")) $("#lightbox").close();
    });

    $("#opt-auto-reuse").addEventListener("change", (event) => {
      state.settings.autoChain = event.target.checked;
      saveSettings();
    });
    $("#opt-auto-scroll").addEventListener("change", (event) => {
      state.settings.autoScroll = event.target.checked;
      saveSettings();
    });
    $("#btn-forget").addEventListener("click", () => {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(HISTORY_KEY);
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(BASE_KEY);
      state.token = "";
      state.apiBase = "";
      toast("Local settings cleared. Reloading…", "ok", 1600);
      setTimeout(() => window.location.reload(), 900);
    });

    document.addEventListener("keydown", (event) => {
      const meta = event.metaKey || event.ctrlKey;
      if (meta && event.key === "Enter") {
        event.preventDefault();
        if (!state.busy) submit();
      }
      if (event.key === "Escape") closeDrawers();
      if (event.key.toLowerCase() === "g" && !meta && !/input|textarea|select/i.test(document.activeElement.tagName)) {
        $("#btn-gallery").click();
      }
    });
  }

  // -------------------------------------------------------------------- boot
  async function boot() {
    loadSettings();
    const params = new URLSearchParams(window.location.search);
    if (params.get("token")) {
      // one-click links from the Colab launcher carry the token; keep it out of the address bar afterwards
      state.token = params.get("token").trim();
      localStorage.setItem(TOKEN_KEY, state.token);
    }
    if (params.get("api")) state.apiBase = (params.get("api") || "").replace(/\/$/, "");
    if (params.has("token") || params.has("api")) {
      params.delete("token");
      params.delete("api");
      const query = params.toString();
      window.history.replaceState({}, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
    }
    $("#opt-auto-reuse").checked = Boolean(state.settings.autoChain);
    $("#opt-auto-scroll").checked = state.settings.autoScroll !== false;

    renderHistory();
    wireCanvasEvents();
    wireCompare();
    wireControls();
    syncControls();

    try {
      await bootstrap();
      if (state.config && state.config.mock) {
        $("#mock-badge").hidden = false;
        toast("Mock engine active: no model is loaded, output is a placeholder.", "warn", 8000);
      }
      const jobs = await api("/api/jobs").catch(() => null);
      const done = jobs && jobs.jobs ? jobs.jobs.find((job) => job.status === "done") : null;
      if (done) showResult(done);
    } catch (error) {
      if (!handleAuthError(error)) {
        toast(error.message, "error", 12000);
        openSettings();
      }
    }

    setInterval(() => {
      if (!document.hidden) refreshEngine();
    }, 20000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
