(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const icon = (name, className = "") => `<svg${className ? ` class="${className}"` : ""} aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  const MAX_FILE_SIZE = 15 * 1024 * 1024;
  const MAX_SESSIONS = 20;
  const SETTINGS_KEY = "herogod.settings.v1";
  const TOKEN_KEY = "herogod.access-token.v1";

  const PROVIDERS = {
    local: { name: "Local / Colab", short: "Local", description: "Open model on your runtime" },
    huggingface: { name: "Hugging Face", short: "Hugging Face", description: "Your inference credits" },
    gemini: { name: "Gemini", short: "Gemini", description: "Your Google AI Studio key" }
  };

  const MODEL_NAMES = {
    "timbrooks/instruct-pix2pix": "InstructPix2Pix",
    "stabilityai/stable-diffusion-xl-base-1.0": "SDXL img2img",
    "gemini-2.5-flash-image": "Gemini Flash Image"
  };

  const defaultBackend = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
  const defaultSettings = {
    backendUrl: defaultBackend,
    provider: "local",
    model: "timbrooks/instruct-pix2pix",
    modelPinned: false,
    advanced: { steps: 20, guidance: 7.5, imageGuidance: 1.5, strength: 0.65, seed: -1 }
  };

  const state = {
    settings: loadSettings(),
    accessToken: loadAccessToken(),
    sessions: [],
    activeSession: null,
    mode: "edit",
    draftImage: null,
    referenceImage: null,
    processing: false,
    abortController: null,
    backend: { status: "checking", data: null },
    db: null,
    storageWarningShown: false,
    saveTimer: null,
    dragDepth: 0
  };

  function loadSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
      if (!saved || typeof saved !== "object") return structuredClone(defaultSettings);
      return {
        ...structuredClone(defaultSettings),
        ...saved,
        advanced: { ...defaultSettings.advanced, ...(saved.advanced || {}) }
      };
    } catch {
      return structuredClone(defaultSettings);
    }
  }

  function loadAccessToken() {
    try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
  }

  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); } catch { /* Settings are non-critical. */ }
  }

  function saveAccessToken() {
    try {
      if (state.accessToken) sessionStorage.setItem(TOKEN_KEY, state.accessToken);
      else sessionStorage.removeItem(TOKEN_KEY);
    } catch { /* The token can still be used for this page lifetime. */ }
  }

  function uid(prefix = "id") {
    if (window.crypto?.randomUUID) return `${prefix}-${crypto.randomUUID()}`;
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, character => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    })[character]);
  }

  function normalizeBackendUrl(value) {
    let url = String(value || "").trim();
    if (!url) throw new Error("Enter a backend URL.");
    if (!/^https?:\/\//i.test(url)) url = `http://${url}`;
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("The backend must use HTTP or HTTPS.");
    parsed.hash = "";
    parsed.search = "";
    return parsed.href.replace(/\/$/, "");
  }

  function authHeaders(token = state.accessToken) {
    return token ? { "Authorization": `Bearer ${token}`, "X-Herogod-Token": token } : {};
  }

  function createSession() {
    const now = Date.now();
    return { id: uid("creation"), title: "New creation", createdAt: now, updatedAt: now, messages: [], draftImage: null, referenceImage: null, mode: "edit" };
  }

  function openDatabase() {
    return new Promise((resolve, reject) => {
      if (!window.indexedDB) return reject(new Error("IndexedDB is unavailable."));
      const request = indexedDB.open("herogod-studio", 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains("sessions")) database.createObjectStore("sessions", { keyPath: "id" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Could not open local history."));
      request.onblocked = () => reject(new Error("Local history is blocked by another tab."));
    });
  }

  function databaseRequest(mode, operation) {
    return new Promise((resolve, reject) => {
      if (!state.db) return resolve(null);
      try {
        const transaction = state.db.transaction("sessions", mode);
        const request = operation(transaction.objectStore("sessions"));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        transaction.onerror = () => reject(transaction.error);
      } catch (error) { reject(error); }
    });
  }

  async function loadSessions() {
    if (!state.db) return [];
    const sessions = await databaseRequest("readonly", store => store.getAll());
    return (sessions || []).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  function snapshotActiveSession() {
    if (!state.activeSession) return null;
    state.activeSession.draftImage = state.draftImage;
    state.activeSession.referenceImage = state.referenceImage;
    state.activeSession.mode = state.mode;
    state.activeSession.updatedAt = Date.now();
    return state.activeSession;
  }

  function persistActiveSession(immediate = false) {
    clearTimeout(state.saveTimer);
    const save = async () => {
      const session = snapshotActiveSession();
      if (!session || !state.db) return;
      try { await databaseRequest("readwrite", store => store.put(session)); }
      catch (error) {
        console.warn("Could not save Herogod history:", error);
        if (!state.storageWarningShown) {
          state.storageWarningShown = true;
          toast("Browser storage is full or unavailable. Download images you want to keep.", "error", 5000);
        }
      }
    };
    if (immediate) save();
    else state.saveTimer = setTimeout(save, 250);
  }

  async function pruneSessions() {
    if (state.sessions.length <= MAX_SESSIONS) return;
    const removable = state.sessions.slice(MAX_SESSIONS);
    state.sessions = state.sessions.slice(0, MAX_SESSIONS);
    for (const session of removable) {
      try { await databaseRequest("readwrite", store => store.delete(session.id)); } catch { /* Non-critical cleanup. */ }
    }
  }

  function renderHistory() {
    const list = $("#historyList");
    const sessionsWithContent = state.sessions.filter(session => session.messages?.length || session.id === state.activeSession?.id);
    if (!sessionsWithContent.length) {
      list.innerHTML = `<div class="history-empty">Your creations will appear here.<br>Start with an image and a little imagination.</div>`;
      return;
    }
    list.innerHTML = sessionsWithContent.map(session => `
      <div class="history-item${session.id === state.activeSession?.id ? " active" : ""}" data-session-id="${escapeHtml(session.id)}">
        <button class="history-open" data-action="open-session" title="${escapeHtml(session.title)}">
          ${icon(session.mode === "face" ? "face" : "image")}<span>${escapeHtml(session.title || "New creation")}</span>
        </button>
        <button class="history-delete" data-action="delete-session" aria-label="Delete ${escapeHtml(session.title || "creation")}" title="Delete creation">${icon("trash")}</button>
      </div>`).join("");
  }

  function activateSession(session) {
    if (!session || state.processing) return;
    persistActiveSession(true);
    state.activeSession = session;
    state.draftImage = session.draftImage || null;
    state.referenceImage = session.referenceImage || null;
    state.mode = session.mode || "edit";
    $("#promptInput").value = "";
    autoResizeTextarea();
    renderAll();
    persistActiveSession();
    closeSidebar();
  }

  function newSession() {
    if (state.processing) {
      toast("Stop the current generation before starting a new creation.");
      return;
    }
    persistActiveSession(true);
    const currentEmpty = state.activeSession && !state.activeSession.messages.length && !state.draftImage;
    if (!currentEmpty) {
      const session = createSession();
      state.sessions.unshift(session);
      state.activeSession = session;
    } else {
      state.activeSession.title = "New creation";
      state.activeSession.messages = [];
      state.activeSession.updatedAt = Date.now();
    }
    state.mode = "edit";
    state.draftImage = null;
    state.referenceImage = null;
    $("#promptInput").value = "";
    $("#consentInput").checked = false;
    autoResizeTextarea();
    renderAll();
    persistActiveSession();
    pruneSessions();
    closeSidebar();
    $("#promptInput").focus();
  }

  async function deleteSession(sessionId) {
    if (state.processing && sessionId === state.activeSession?.id) {
      toast("Stop the current generation before deleting this creation.", "error");
      return;
    }
    const session = state.sessions.find(item => item.id === sessionId);
    if (!session) return;
    if (session.messages?.length && !window.confirm("Delete this creation and its locally saved images?")) return;
    state.sessions = state.sessions.filter(item => item.id !== sessionId);
    try { await databaseRequest("readwrite", store => store.delete(sessionId)); } catch { /* It is already removed from this view. */ }
    if (state.activeSession?.id === sessionId) {
      const next = state.sessions[0] || createSession();
      if (!state.sessions.length) state.sessions = [next];
      state.activeSession = next;
      state.draftImage = next.draftImage || null;
      state.referenceImage = next.referenceImage || null;
      state.mode = next.mode || "edit";
    }
    renderAll();
    toast("Creation deleted.");
  }

  function renderAll() {
    renderHistory();
    renderMessages();
    renderAttachments();
    updateModeUI();
    updateSendButton();
    const title = state.activeSession?.title || "New creation";
    $("#sessionTitle").textContent = title;
    document.title = `${title} — Herogod`;
  }

  function renderMessages() {
    const container = $("#messages");
    const messages = state.activeSession?.messages || [];
    $("#welcome").hidden = messages.length > 0;
    container.innerHTML = messages.map(message => message.role === "user" ? renderUserMessage(message) : renderAssistantMessage(message)).join("");
  }

  function renderUserMessage(message) {
    const images = (message.images || []).map((image, index) => `
      <div class="user-image">
        <img src="${escapeHtml(image.dataUrl)}" alt="${escapeHtml(image.label || "Uploaded image")}" data-action="expand-user-image" data-message-id="${message.id}" data-image-index="${index}">
        <span>${escapeHtml(image.label || "Source")}</span>
      </div>`).join("");
    return `<article class="message user-message" data-message-id="${message.id}">
      <div class="message-avatar">Y</div>
      <div class="message-body">
        <div class="message-author">You</div>
        ${images ? `<div class="user-images">${images}</div>` : ""}
        <div class="user-bubble">${escapeHtml(message.text)}</div>
      </div>
    </article>`;
  }

  function renderAssistantMessage(message) {
    let content = "";
    if (message.status === "loading") {
      content = `<div class="loading-card">
        <div class="loading-preview"><div class="loading-orbit">${icon("spark")}</div></div>
        <div class="loading-line"></div>
        <div class="loading-copy"><strong>${message.mode === "face" ? "Mapping your face swap…" : "Bringing your idea to life…"}</strong><span>${escapeHtml(message.providerName || "Your AI runtime")} is working. The first local run may need to download a model.</span></div>
      </div>`;
    } else if (message.status === "error") {
      content = `<div class="error-card">
        <strong>${icon("info")}Couldn’t complete this edit</strong>
        <p>${escapeHtml(message.error || "The backend returned an unknown error.")}</p>
        <button data-action="open-settings" data-message-id="${message.id}">Check connection settings</button>
      </div>`;
    } else if (message.image) {
      const elapsed = Number(message.elapsedMs) > 0 ? `${(Number(message.elapsedMs) / 1000).toFixed(1)}s` : "";
      const seed = message.seed !== undefined && message.seed !== null && Number(message.seed) >= 0 ? `Seed ${message.seed}` : "";
      content = `<div class="result-card">
        <div class="result-image-wrap">
          <img class="result-image" src="${escapeHtml(message.image)}" alt="AI edited result" data-action="expand-result" data-message-id="${message.id}">
          <div class="image-hover-actions">
            <button data-action="expand-result" data-message-id="${message.id}" aria-label="Expand image" title="Expand">${icon("expand")}</button>
            <button data-action="download-result" data-message-id="${message.id}" aria-label="Download image" title="Download">${icon("download")}</button>
          </div>
        </div>
        <div class="result-info">
          <div class="result-caption"><strong>${message.mode === "face" ? "Face swap complete" : "Edit complete"}</strong> · ${escapeHtml(message.providerName || "Herogod")}</div>
          <div class="result-actions">
            <button class="result-action" data-action="use-as-input" data-message-id="${message.id}" title="Continue editing this result">${icon("refresh")}<span>Use as input</span></button>
            <button class="result-action" data-action="download-result" data-message-id="${message.id}">${icon("download")}<span>Download</span></button>
          </div>
        </div>
        <div class="result-meta"><span>${escapeHtml(MODEL_NAMES[message.model] || message.model || "Open model")}</span>${seed ? `<span>${escapeHtml(seed)}</span>` : ""}${elapsed ? `<span>${escapeHtml(elapsed)}</span>` : ""}</div>
      </div>`;
    }
    return `<article class="message assistant-message" data-message-id="${message.id}">
      <div class="message-avatar">${icon("spark")}</div>
      <div class="message-body"><div class="message-author">Herogod</div>${content}</div>
    </article>`;
  }

  function scrollToBottom(behavior = "smooth") {
    requestAnimationFrame(() => $("#chatScroll").scrollTo({ top: $("#chatScroll").scrollHeight, behavior }));
  }

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not read that image."));
      reader.readAsDataURL(file);
    });
  }

  function dataUrlToBlob(dataUrl) {
    const match = /^data:([^;]+);base64,(.+)$/s.exec(dataUrl);
    if (!match) throw new Error("The attached image data is invalid.");
    const binary = atob(match[2]);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return new Blob([bytes], { type: match[1] });
  }

  async function attachFile(file, kind = "image") {
    if (!file) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      toast("Choose a PNG, JPG, or WebP image.", "error");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      toast("That image is over 15 MB. Choose a smaller file.", "error");
      return;
    }
    try {
      const dataUrl = await fileToDataUrl(file);
      const image = { dataUrl, name: file.name || `${kind}.png`, type: file.type, size: file.size };
      if (kind === "reference") state.referenceImage = image;
      else state.draftImage = image;
      renderAttachments();
      updateSendButton();
      persistActiveSession();
      const label = kind === "reference" ? "Face reference added." : "Image ready. Describe your edit.";
      toast(label);
      if (state.mode === "edit") $("#promptInput").focus();
    } catch (error) { toast(error.message, "error"); }
  }

  function renderAttachments() {
    const container = $("#attachmentPreviews");
    const items = [];
    if (state.draftImage) {
      items.push(`<div class="attachment-preview">
        <img src="${escapeHtml(state.draftImage.dataUrl)}" alt="Source preview">
        <span><strong>${escapeHtml(state.draftImage.name || "Source image")}</strong><small>${state.mode === "face" ? "Target image" : "Editing source"}</small></span>
        <button type="button" class="remove-attachment" data-remove-image="image" aria-label="Remove source image">${icon("close")}</button>
      </div>`);
    }
    if (state.mode === "face" && state.referenceImage) {
      items.push(`<div class="attachment-preview">
        <img src="${escapeHtml(state.referenceImage.dataUrl)}" alt="Face reference preview">
        <span><strong>${escapeHtml(state.referenceImage.name || "Face reference")}</strong><small>Face to place in target</small></span>
        <button type="button" class="remove-attachment" data-remove-image="reference" aria-label="Remove face reference">${icon("close")}</button>
      </div>`);
    }
    if (state.mode === "face" && !state.referenceImage) {
      items.push(`<button type="button" class="attachment-add" id="addReferenceButton">${icon("face")}Add face reference</button>`);
    }
    container.innerHTML = items.join("");
    container.hidden = items.length === 0;
  }

  function updateModeUI() {
    $$('[data-mode-button]').forEach(button => {
      const active = button.dataset.modeButton === state.mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const faceMode = state.mode === "face";
    $("#consentRow").hidden = !faceMode;
    $("#promptInput").placeholder = faceMode ? "Add an optional note about this face swap…" : "Describe what you want to change…";
    $("#modelSelect").disabled = faceMode;
    $("#advancedToggle").disabled = faceMode || state.settings.provider === "gemini";
    if (faceMode) {
      $("#advancedPanel").hidden = true;
      $("#advancedToggle").classList.remove("active");
      $("#composerNote").innerHTML = `Face swap uses your local/Colab InsightFace runtime. Consent is required. <button id="setupLink">Setup guide</button>`;
    } else {
      updateAdvancedVisibility();
      updateComposerNote();
    }
    updateSendButton();
  }

  function updateAdvancedVisibility() {
    const provider = state.settings.provider;
    const model = state.settings.model;
    $("#advancedToggle").disabled = provider === "gemini";
    if (provider === "gemini") {
      $("#advancedPanel").hidden = true;
      $("#advancedToggle").classList.remove("active");
      return;
    }
    const isSDXL = model === "stabilityai/stable-diffusion-xl-base-1.0";
    const isLocal = provider === "local";
    $("#strengthControl").hidden = !isSDXL;
    $("#imageGuidanceControl").hidden = !isLocal || isSDXL;
    $("#stepsControl").hidden = false;
    $("#guidanceControl").hidden = false;
  }

  function updateComposerNote() {
    const provider = state.settings.provider;
    const providerStatus = state.backend.data?.providers?.find(item => item.id === provider);
    let text = "Herogod can make mistakes. Review images before sharing.";
    if (state.backend.status !== "online") text = "Connect a local or Colab backend to create.";
    else if (providerStatus && !providerStatus.configured) {
      text = provider === "local"
        ? "Local Diffusers dependencies are not installed on this backend."
        : `Add ${provider === "gemini" ? "GEMINI_API_KEY" : "HF_TOKEN"} to your backend before creating.`;
    }
    else if (provider === "local") text = "Self-hosted open model · no AI API key or per-image fee.";
    else text = `${PROVIDERS[provider].name} uses your own free-tier quota. Provider rules apply.`;
    $("#composerNote").innerHTML = `${escapeHtml(text)} <button id="setupLink">Setup guide</button>`;
  }

  function updateSendButton() {
    const button = $("#sendButton");
    if (state.processing) {
      button.disabled = false;
      button.classList.add("processing");
      button.innerHTML = icon("stop");
      button.setAttribute("aria-label", "Stop generation");
      button.title = "Stop generation";
      return;
    }
    button.classList.remove("processing");
    button.innerHTML = icon("send");
    button.setAttribute("aria-label", "Send prompt");
    button.title = "Send prompt";
    const hasPrompt = $("#promptInput").value.trim().length > 0;
    const valid = state.mode === "face"
      ? Boolean(state.draftImage && state.referenceImage && $("#consentInput").checked)
      : Boolean(state.draftImage && hasPrompt);
    button.disabled = !valid;
  }

  function setMode(mode) {
    if (!["edit", "face"].includes(mode) || state.processing) return;
    state.mode = mode;
    renderAttachments();
    updateModeUI();
    persistActiveSession();
  }

  function autoResizeTextarea() {
    const textarea = $("#promptInput");
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
  }

  function readAdvancedSettings() {
    return {
      steps: Number($("#stepsInput").value),
      guidance: Number($("#guidanceInput").value),
      imageGuidance: Number($("#imageGuidanceInput").value),
      strength: Number($("#strengthInput").value),
      seed: Number($("#seedInput").value)
    };
  }

  function applyAdvancedSettings() {
    const advanced = state.settings.advanced;
    $("#stepsInput").value = advanced.steps;
    $("#guidanceInput").value = advanced.guidance;
    $("#imageGuidanceInput").value = advanced.imageGuidance;
    $("#strengthInput").value = advanced.strength;
    $("#seedInput").value = advanced.seed;
    updateRangeOutputs();
  }

  function updateRangeOutputs() {
    $("#stepsValue").value = $("#stepsInput").value;
    $("#guidanceValue").value = $("#guidanceInput").value;
    $("#imageGuidanceValue").value = $("#imageGuidanceInput").value;
    $("#strengthValue").value = $("#strengthInput").value;
  }

  async function submitPrompt() {
    if (state.processing) {
      state.abortController?.abort("user");
      return;
    }
    const prompt = $("#promptInput").value.trim();
    if (!state.draftImage) {
      toast("Attach an image first.", "error");
      $("#attachButton").focus();
      return;
    }
    if (state.mode === "edit" && !prompt) return;
    if (state.mode === "face" && (!state.referenceImage || !$("#consentInput").checked)) {
      toast("Add a face reference and confirm consent.", "error");
      return;
    }

    const request = {
      mode: state.mode,
      prompt: state.mode === "face" ? (prompt || "Swap the target face using my consensual reference image.") : prompt,
      image: state.draftImage,
      reference: state.referenceImage,
      provider: state.settings.provider,
      model: state.settings.model,
      advanced: readAdvancedSettings()
    };
    const userMessage = {
      id: uid("message"), role: "user", text: request.prompt,
      images: [{ dataUrl: request.image.dataUrl, label: request.mode === "face" ? "Target" : "Source" }],
      createdAt: Date.now()
    };
    if (request.mode === "face") userMessage.images.push({ dataUrl: request.reference.dataUrl, label: "Face reference" });
    const assistantMessage = {
      id: uid("message"), role: "assistant", status: "loading", mode: request.mode,
      providerName: request.mode === "face" ? "Local InsightFace" : PROVIDERS[request.provider].name,
      createdAt: Date.now()
    };

    if (!state.activeSession.messages.length || state.activeSession.title === "New creation") {
      state.activeSession.title = request.prompt.replace(/\s+/g, " ").slice(0, 42) || "Face swap creation";
    }
    state.activeSession.messages.push(userMessage, assistantMessage);
    state.processing = true;
    state.abortController = new AbortController();
    $("#promptInput").value = "";
    autoResizeTextarea();
    renderAll();
    scrollToBottom();
    persistActiveSession();

    const timeout = setTimeout(() => state.abortController?.abort("timeout"), 5 * 60 * 1000);
    try {
      const body = new FormData();
      body.append("image", dataUrlToBlob(request.image.dataUrl), request.image.name || "source.png");
      let endpoint = "/api/face-swap";
      if (request.mode === "face") {
        body.append("reference_image", dataUrlToBlob(request.reference.dataUrl), request.reference.name || "reference.png");
        body.append("consent", "true");
      } else {
        endpoint = "/api/edit";
        body.append("prompt", request.prompt);
        body.append("provider", request.provider);
        body.append("model", request.model);
        body.append("steps", String(request.advanced.steps));
        body.append("guidance", String(request.advanced.guidance));
        body.append("image_guidance", String(request.advanced.imageGuidance));
        body.append("strength", String(request.advanced.strength));
        body.append("seed", String(request.advanced.seed));
      }
      const response = await fetch(`${normalizeBackendUrl(state.settings.backendUrl)}${endpoint}`, {
        method: "POST", headers: authHeaders(), body, signal: state.abortController.signal
      });
      const result = await parseApiResponse(response);
      if (!result.image || !/^data:image\/(png|jpeg|webp);base64,/i.test(result.image)) throw new Error("The backend returned an invalid image response.");
      assistantMessage.status = "done";
      assistantMessage.image = result.image;
      assistantMessage.providerName = PROVIDERS[result.provider]?.name || (request.mode === "face" ? "Local InsightFace" : request.providerName);
      assistantMessage.model = result.model || request.model;
      assistantMessage.seed = result.seed;
      assistantMessage.elapsedMs = result.elapsed_ms;
      state.draftImage = {
        dataUrl: result.image,
        name: `herogod-${request.mode === "face" ? "face-swap" : "edit"}-${Date.now()}.png`,
        type: "image/png"
      };
      toast("Your image is ready.");
    } catch (error) {
      const reason = state.abortController?.signal.reason;
      assistantMessage.status = "error";
      if (reason === "user") assistantMessage.error = "Generation stopped. Your source image is still attached, so you can try again.";
      else if (reason === "timeout") assistantMessage.error = "The request took longer than 5 minutes. A first model download can be slow—check the backend logs, then try again.";
      else assistantMessage.error = friendlyError(error);
      if (!$("#promptInput").value && request.mode === "edit") {
        $("#promptInput").value = request.prompt;
        autoResizeTextarea();
      }
    } finally {
      clearTimeout(timeout);
      state.processing = false;
      state.abortController = null;
      renderAll();
      scrollToBottom();
      persistActiveSession(true);
    }
  }

  async function parseApiResponse(response) {
    let payload;
    const contentType = response.headers.get("content-type") || "";
    try { payload = contentType.includes("application/json") ? await response.json() : { detail: await response.text() }; }
    catch { payload = { detail: `Backend returned HTTP ${response.status}.` }; }
    if (!response.ok) {
      const detail = payload?.detail;
      let message = typeof detail === "string" ? detail : payload?.error || `Request failed with HTTP ${response.status}.`;
      if (Array.isArray(detail)) message = detail.map(item => item.msg || JSON.stringify(item)).join("; ");
      if (response.status === 401 || response.status === 403) message = "The backend rejected the access token. Check Connection settings.";
      throw new Error(message);
    }
    return payload;
  }

  function friendlyError(error) {
    if (error instanceof TypeError && /fetch/i.test(error.message)) {
      return "Herogod could not reach the backend. Check the URL, make sure the server is running, and confirm that your tunnel has not expired.";
    }
    return error.message || "An unexpected error interrupted the request.";
  }

  async function checkBackend(url = state.settings.backendUrl, token = state.accessToken, updateGlobal = true) {
    if (updateGlobal) setConnectionStatus("checking");
    try {
      const normalized = normalizeBackendUrl(url);
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12000);
      const response = await fetch(`${normalized}/api/health`, { headers: authHeaders(token), signal: controller.signal, cache: "no-store" });
      clearTimeout(timer);
      if (!response.ok) throw new Error(response.status === 401 || response.status === 403 ? "Access token rejected" : `HTTP ${response.status}`);
      const data = await response.json();
      if (data.app !== "Herogod") throw new Error("This URL is not a Herogod backend");
      if (updateGlobal) {
        state.backend = { status: "online", data };
        applyBackendDefaults(data);
        setConnectionStatus("online");
        updateComposerNote();
      }
      return { ok: true, data };
    } catch (error) {
      if (updateGlobal) {
        state.backend = { status: "offline", data: null };
        setConnectionStatus("offline");
        updateComposerNote();
      }
      return { ok: false, error: error.name === "AbortError" ? "Connection timed out" : friendlyError(error) };
    }
  }

  function applyBackendDefaults(data) {
    if (state.settings.modelPinned || state.settings.provider !== "local") return;
    const localProvider = data.providers?.find(provider => provider.id === "local");
    const defaultModel = localProvider?.default_model;
    if (!defaultModel || defaultModel === state.settings.model) return;
    const value = `local:${defaultModel}`;
    if ($$("#modelSelect option").some(option => option.value === value)) {
      state.settings.model = defaultModel;
      $("#modelSelect").value = value;
      saveSettings();
      updateAdvancedVisibility();
    }
  }

  function setConnectionStatus(status) {
    const dot = $("#connectionPill .status-dot");
    dot.className = `status-dot ${status}`;
    const labels = { checking: "Checking backend", online: "Backend connected", offline: "Backend offline" };
    $("#connectionLabel").textContent = labels[status] || "Backend";
  }

  function openSettings() {
    closeSidebar();
    $("#backendUrlInput").value = state.settings.backendUrl;
    $("#accessTokenInput").value = state.accessToken;
    $("#connectionTestResult").textContent = "";
    $("#connectionTestResult").className = "";
    openDialog($("#settingsDialog"));
  }

  function openGuide(tab = "free") {
    closeSidebar();
    switchGuideTab(tab);
    openDialog($("#guideDialog"));
  }

  function openDialog(dialog) {
    if (!dialog.open) dialog.showModal();
  }

  function switchGuideTab(tab) {
    $$('[data-guide-tab]').forEach(button => {
      const active = button.dataset.guideTab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    $$('[data-guide-panel]').forEach(panel => {
      const active = panel.dataset.guidePanel === tab;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
  }

  function findMessage(messageId) {
    return state.activeSession?.messages.find(message => message.id === messageId);
  }

  function expandImage(dataUrl, label = "Generated image") {
    $("#expandedImage").src = dataUrl;
    $("#expandedImageLabel").textContent = label;
    $("#expandedDownload").href = dataUrl;
    $("#expandedDownload").download = `herogod-${Date.now()}.png`;
    openDialog($("#imageDialog"));
  }

  function downloadImage(message) {
    if (!message?.image) return;
    const anchor = document.createElement("a");
    anchor.href = message.image;
    anchor.download = `herogod-${message.mode === "face" ? "face-swap" : "edit"}-${Date.now()}.png`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    toast("Download started.");
  }

  function useAsInput(message) {
    if (!message?.image || state.processing) return;
    state.draftImage = { dataUrl: message.image, name: `herogod-input-${Date.now()}.png`, type: "image/png" };
    state.mode = "edit";
    renderAttachments();
    updateModeUI();
    updateSendButton();
    persistActiveSession();
    $("#promptInput").focus();
    toast("Result is ready for your next edit.");
  }

  function toast(message, type = "success", duration = 3200) {
    const element = document.createElement("div");
    element.className = `toast ${type}`;
    element.innerHTML = `${icon(type === "error" ? "info" : "check")}<span>${escapeHtml(message)}</span>`;
    $("#toastRegion").appendChild(element);
    setTimeout(() => {
      element.classList.add("leaving");
      setTimeout(() => element.remove(), 220);
    }, duration);
  }

  function openSidebar() { document.body.classList.add("sidebar-open"); }
  function closeSidebar() { document.body.classList.remove("sidebar-open"); }

  function bindEvents() {
    $("#newChatButton").addEventListener("click", newSession);
    $("#mobileMenu").addEventListener("click", openSidebar);
    $("#sidebarClose").addEventListener("click", closeSidebar);
    $("#sidebarScrim").addEventListener("click", closeSidebar);
    $("#settingsButton").addEventListener("click", openSettings);
    $("#connectionPill").addEventListener("click", openSettings);
    $("#guideButton").addEventListener("click", () => openGuide("free"));
    $("#topGuideButton").addEventListener("click", () => openGuide("free"));
    $("#privacyButton").addEventListener("click", () => openGuide("responsible"));
    $("#heroUploadButton").addEventListener("click", () => $("#imageInput").click());
    $("#attachButton").addEventListener("click", () => $("#imageInput").click());
    $("#imageInput").addEventListener("change", event => { attachFile(event.target.files[0], "image"); event.target.value = ""; });
    $("#referenceInput").addEventListener("change", event => { attachFile(event.target.files[0], "reference"); event.target.value = ""; });
    $("#composer").addEventListener("submit", event => { event.preventDefault(); submitPrompt(); });
    $("#promptInput").addEventListener("input", () => { autoResizeTextarea(); updateSendButton(); });
    $("#promptInput").addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!$("#sendButton").disabled || state.processing) submitPrompt();
      }
    });
    $("#consentInput").addEventListener("change", updateSendButton);
    $$('[data-mode-button]').forEach(button => button.addEventListener("click", () => setMode(button.dataset.modeButton)));
    $$('.suggestion-card').forEach(card => card.addEventListener("click", () => {
      setMode(card.dataset.mode || "edit");
      if (card.dataset.prompt) $("#promptInput").value = card.dataset.prompt;
      autoResizeTextarea();
      updateSendButton();
      if (!state.draftImage) {
        toast(card.dataset.mode === "face" ? "Add a target image, then a consensual face reference." : "Add an image to use this idea.");
        $("#imageInput").click();
      } else $("#promptInput").focus();
    }));

    $("#attachmentPreviews").addEventListener("click", event => {
      const remove = event.target.closest("[data-remove-image]");
      const add = event.target.closest("#addReferenceButton");
      if (remove) {
        if (remove.dataset.removeImage === "reference") state.referenceImage = null;
        else state.draftImage = null;
        renderAttachments();
        updateSendButton();
        persistActiveSession();
      }
      if (add) $("#referenceInput").click();
    });

    $("#modelSelect").addEventListener("change", event => {
      const value = event.target.value;
      const separator = value.indexOf(":");
      state.settings.provider = value.slice(0, separator);
      state.settings.model = value.slice(separator + 1);
      state.settings.modelPinned = true;
      saveSettings();
      updateAdvancedVisibility();
      updateComposerNote();
      updateSendButton();
    });

    $("#advancedToggle").addEventListener("click", () => {
      const panel = $("#advancedPanel");
      panel.hidden = !panel.hidden;
      $("#advancedToggle").classList.toggle("active", !panel.hidden);
      $("#advancedToggle").setAttribute("aria-expanded", String(!panel.hidden));
    });
    ["#stepsInput", "#guidanceInput", "#imageGuidanceInput", "#strengthInput", "#seedInput"].forEach(selector => {
      $(selector).addEventListener("input", () => {
        updateRangeOutputs();
        state.settings.advanced = readAdvancedSettings();
        saveSettings();
      });
    });

    $("#historyList").addEventListener("click", event => {
      const action = event.target.closest("[data-action]");
      const item = event.target.closest("[data-session-id]");
      if (!action || !item) return;
      if (action.dataset.action === "open-session") activateSession(state.sessions.find(session => session.id === item.dataset.sessionId));
      if (action.dataset.action === "delete-session") deleteSession(item.dataset.sessionId);
    });

    $("#messages").addEventListener("click", event => {
      const target = event.target.closest("[data-action]");
      if (!target) return;
      const message = findMessage(target.dataset.messageId);
      switch (target.dataset.action) {
        case "expand-result": if (message?.image) expandImage(message.image, MODEL_NAMES[message.model] || "Generated image"); break;
        case "download-result": downloadImage(message); break;
        case "use-as-input": useAsInput(message); break;
        case "open-settings": openSettings(); break;
        case "expand-user-image": {
          const image = message?.images?.[Number(target.dataset.imageIndex)];
          if (image) expandImage(image.dataUrl, image.label || "Uploaded image");
          break;
        }
      }
    });

    $$('[data-close-dialog]').forEach(button => button.addEventListener("click", () => $(`#${button.dataset.closeDialog}`).close()));
    $$('dialog').forEach(dialog => dialog.addEventListener("click", event => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside || dialog.id === "imageDialog") dialog.close();
    }));
    $$('[data-guide-tab]').forEach(button => button.addEventListener("click", () => switchGuideTab(button.dataset.guideTab)));
    $("#settingsGuideLink").addEventListener("click", () => { $("#settingsDialog").close(); openGuide("api"); });
    $("#colabSettingsButton").addEventListener("click", () => { $("#guideDialog").close(); openSettings(); });

    $("#saveSettingsButton").addEventListener("click", async () => {
      try {
        const url = normalizeBackendUrl($("#backendUrlInput").value);
        state.settings.backendUrl = url;
        state.accessToken = $("#accessTokenInput").value.trim();
        saveSettings();
        saveAccessToken();
        $("#settingsDialog").close();
        await checkBackend();
        if (state.backend.status === "online") toast("Backend connected.");
      } catch (error) { toast(error.message, "error"); }
    });

    $("#testConnectionButton").addEventListener("click", async () => {
      const button = $("#testConnectionButton");
      const result = $("#connectionTestResult");
      button.disabled = true;
      result.className = "";
      result.textContent = "Testing…";
      const test = await checkBackend($("#backendUrlInput").value, $("#accessTokenInput").value.trim(), false);
      button.disabled = false;
      result.className = test.ok ? "success" : "error";
      result.textContent = test.ok ? `Connected · ${test.data.providers?.filter(provider => provider.configured).length || 0} provider(s) ready` : test.error;
    });

    document.addEventListener("click", event => {
      if (event.target.closest("#setupLink")) openGuide("colab");
    });

    const composer = $("#composer");
    ["dragenter", "dragover"].forEach(type => composer.addEventListener(type, event => {
      event.preventDefault();
      if (type === "dragenter") state.dragDepth += 1;
      composer.classList.add("drag-over");
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    }));
    composer.addEventListener("dragleave", event => {
      event.preventDefault();
      state.dragDepth = Math.max(0, state.dragDepth - 1);
      if (!state.dragDepth) composer.classList.remove("drag-over");
    });
    composer.addEventListener("drop", event => {
      event.preventDefault();
      state.dragDepth = 0;
      composer.classList.remove("drag-over");
      const file = event.dataTransfer?.files?.[0];
      if (file) attachFile(file, state.mode === "face" && state.draftImage && !state.referenceImage ? "reference" : "image");
    });
    window.addEventListener("dragover", event => event.preventDefault());
    window.addEventListener("drop", event => event.preventDefault());
    window.addEventListener("beforeunload", () => persistActiveSession(true));
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") closeSidebar();
    });
  }

  async function init() {
    bindEvents();
    applyAdvancedSettings();
    const modelValue = `${state.settings.provider}:${state.settings.model}`;
    if ($$("#modelSelect option").some(option => option.value === modelValue)) $("#modelSelect").value = modelValue;
    else {
      state.settings.provider = defaultSettings.provider;
      state.settings.model = defaultSettings.model;
      state.settings.modelPinned = false;
      $("#modelSelect").value = `${defaultSettings.provider}:${defaultSettings.model}`;
    }
    try {
      state.db = await openDatabase();
      state.sessions = await loadSessions();
    } catch (error) {
      console.warn("Herogod history is unavailable:", error);
      state.db = null;
    }
    if (!state.sessions.length) state.sessions = [createSession()];
    state.activeSession = state.sessions[0];
    state.draftImage = state.activeSession.draftImage || null;
    state.referenceImage = state.activeSession.referenceImage || null;
    state.mode = state.activeSession.mode || "edit";
    renderAll();
    checkBackend();
  }

  init();
})();
