/* Run with jsdom 26+: NODE_PATH=/path/to/node_modules node tests/frontend_smoke.cjs */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const javascript = fs.readFileSync(path.join(root, "app.js"), "utf8");
const outputImage = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
const modelIds = [
  "timbrooks/instruct-pix2pix",
  "stabilityai/stable-diffusion-xl-base-1.0",
  "black-forest-labs/flux-3-image-klein",
  "briaai/Fibo-2.0-edit-turbo",
  "Qwen/Qwen-Image-Edit-2511"
];

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
async function waitFor(assertion, label) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try { return assertion(); } catch (error) {
      if (attempt === 99) throw new Error(`${label}: ${error.message}`);
      await delay(20);
    }
  }
}

function attach(window, selector, name) {
  const input = window.document.querySelector(selector);
  const file = new window.File([Buffer.from("source-image")], name, { type: "image/png" });
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  input.dispatchEvent(new window.Event("change", { bubbles: true }));
}

function selectModel(window, value) {
  const select = window.document.querySelector("#modelSelect");
  select.value = value;
  select.dispatchEvent(new window.Event("change", { bubbles: true }));
}

async function main() {
  const dom = new JSDOM(html, {
    url: "https://studio.example/",
    runScripts: "outside-only",
    pretendToBeVisual: true
  });
  const { window } = dom;
  window.structuredClone = global.structuredClone;
  window.console.warn = () => {};
  window.requestAnimationFrame = callback => window.setTimeout(callback, 0);
  window.document.querySelector("#chatScroll").scrollTo = () => {};
  for (const dialog of window.document.querySelectorAll("dialog")) {
    dialog.showModal = () => { dialog.open = true; };
    dialog.close = () => { dialog.open = false; };
  }

  const calls = [];
  window.fetch = async (url, options = {}) => {
    const call = { url: String(url), options };
    calls.push(call);
    const hfToken = options.headers?.["X-HF-Token"] || "";
    let payload;
    if (call.url.endsWith("/api/health")) {
      payload = {
        app: "Herogod",
        providers: [
          { id: "local", configured: true, default_model: modelIds[0] },
          { id: "huggingface", configured: Boolean(hfToken) },
          { id: "gemini", configured: false }
        ],
        models: modelIds.map(id => ({
          id,
          requires_hf_token: [modelIds[2], modelIds[3]].includes(id),
          token_available: Boolean(hfToken),
          prompt_upsampling: id === modelIds[2],
          prompt_note: `Prompt guidance for ${id}.`,
          hardware: "Test hardware guidance."
        }))
      };
    } else if (call.url.endsWith("/api/edit")) {
      payload = { image: outputImage, provider: "local", model: modelIds[0], seed: 7, elapsed_ms: 10 };
    } else if (call.url.endsWith("/api/face-swap")) {
      payload = { image: outputImage, provider: "local", model: "inswapper", seed: -1, elapsed_ms: 10 };
    } else {
      throw new Error(`Unexpected request: ${call.url}`);
    }
    return {
      ok: true,
      status: 200,
      headers: { get: () => "application/json" },
      json: async () => payload,
      text: async () => JSON.stringify(payload)
    };
  };

  window.eval(javascript);
  await waitFor(() => assert.equal(calls.filter(call => call.url.endsWith("/api/health")).length, 1), "initial health check");
  await waitFor(() => assert.equal(window.document.querySelector("#connectionLabel").textContent, "Backend connected"), "online status");

  selectModel(window, `local:${modelIds[2]}`);
  assert.equal(window.document.querySelector("#stepsInput").value, "16");
  assert.equal(window.document.querySelector("#guidanceControl").hidden, true);
  assert.match(window.document.querySelector("#promptInput").placeholder, /complete resulting scene/);
  assert.match(window.document.querySelector("#composerNote").textContent, /Gated model/);

  window.document.querySelector("#settingsButton").click();
  window.document.querySelector("#accessTokenInput").value = "herogod-test-token";
  window.document.querySelector("#hfTokenInput").value = "hf_test_token";
  window.document.querySelector("#saveSettingsButton").click();
  await waitFor(() => assert.equal(calls.filter(call => call.url.endsWith("/api/health")).length, 2), "settings health check");
  const authenticatedHealth = calls.filter(call => call.url.endsWith("/api/health"))[1];
  assert.equal(authenticatedHealth.options.headers.Authorization, "Bearer herogod-test-token");
  assert.equal(authenticatedHealth.options.headers["X-HF-Token"], "hf_test_token");
  assert.equal(window.sessionStorage.getItem("herogod.hf-token.v1"), "hf_test_token");
  assert.doesNotMatch(window.localStorage.getItem("herogod.settings.v1"), /hf_test_token|herogod-test-token/);
  await waitFor(() => assert.match(window.document.querySelector("#composerNote").textContent, /upsampler enabled/), "upsampler status");

  selectModel(window, `local:${modelIds[3]}`);
  assert.equal(window.document.querySelector("#stepsInput").value, "4");
  assert.equal(window.document.querySelector("#guidanceControl").hidden, true);
  selectModel(window, `local:${modelIds[4]}`);
  assert.equal(window.document.querySelector("#stepsInput").value, "40");
  assert.equal(window.document.querySelector("#guidanceInput").value, "4");
  assert.equal(window.document.querySelector("#guidanceLabel").textContent, "True CFG");
  assert.equal(window.document.querySelector("#guidanceControl").hidden, false);
  selectModel(window, `local:${modelIds[0]}`);
  assert.equal(window.document.querySelector("#stepsInput").value, "20");
  assert.equal(window.document.querySelector("#imageGuidanceControl").hidden, false);

  attach(window, "#imageInput", "source.png");
  await waitFor(() => assert.match(window.document.querySelector("#attachmentPreviews").textContent, /source.png/), "source attachment");
  const prompt = window.document.querySelector("#promptInput");
  prompt.value = "make the background blue";
  prompt.dispatchEvent(new window.Event("input", { bubbles: true }));
  assert.equal(window.document.querySelector("#sendButton").disabled, false);
  window.document.querySelector("#sendButton").click();
  await waitFor(() => assert.equal(window.document.querySelectorAll(".result-image").length, 1), "edit result");
  const edit = calls.find(call => call.url.endsWith("/api/edit"));
  assert.equal(edit.options.headers["X-HF-Token"], "hf_test_token");
  assert.equal(edit.options.body.get("model"), modelIds[0]);
  assert.equal(edit.options.body.get("prompt"), "make the background blue");
  assert.equal(edit.options.body.get("steps"), "20");

  window.document.querySelector('[data-action="use-as-input"]').click();
  assert.match(window.document.querySelector("#attachmentPreviews").textContent, /herogod-input/);
  window.document.querySelector('[data-mode-button="face"]').click();
  attach(window, "#referenceInput", "reference.png");
  await waitFor(() => assert.match(window.document.querySelector("#attachmentPreviews").textContent, /reference.png/), "face reference");
  const consent = window.document.querySelector("#consentInput");
  consent.checked = true;
  consent.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.equal(window.document.querySelector("#sendButton").disabled, false);
  window.document.querySelector("#sendButton").click();
  await waitFor(() => assert.equal(calls.filter(call => call.url.endsWith("/api/face-swap")).length, 1), "face-swap request");
  await waitFor(() => assert.equal(window.document.querySelectorAll(".result-image").length, 2), "face-swap result");
  const faceSwap = calls.find(call => call.url.endsWith("/api/face-swap"));
  assert.equal(faceSwap.options.headers["X-HF-Token"], undefined);
  assert.equal(faceSwap.options.headers.Authorization, "Bearer herogod-test-token");

  assert.doesNotMatch(window.document.documentElement.innerHTML, /hf_test_token|herogod-test-token/);
  window.close();
  console.log("Frontend smoke test passed: settings, token secrecy, model defaults, edit, reuse, and face swap.");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
