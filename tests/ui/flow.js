#!/usr/bin/env node
/**
 * End-to-end flow test in jsdom: upload → generate → result → gallery → compare →
 * reuse as input. Runs against the mock engine, so it needs no GPU.
 *
 *   BASE=http://127.0.0.1:8123/ node flow.js
 */
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = (process.env.BASE || "http://127.0.0.1:8123/").replace(/\/?$/, "/");
const PNG = fs.readFileSync(path.join(__dirname, "sample.png"));
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (error) => errors.push(String(error.stack || error.message)));
virtualConsole.on("error", (...args) => errors.push("console.error: " + args.join(" ")));

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

(async () => {
  const dom = await JSDOM.fromURL(BASE, {
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    virtualConsole,
    beforeParse(window) {
      window.fetch = (input, init) => fetch(new URL(input, BASE).toString(), init);
      window.FormData = FormData;
      window.File = File;
      window.URL.createObjectURL = () => "blob:ui-test-" + Math.random().toString(16).slice(2);
      window.URL.revokeObjectURL = () => {};
      window.HTMLDialogElement.prototype.showModal = function showModal() {
        this.open = true;
      };
      window.HTMLDialogElement.prototype.close = function close() {
        this.open = false;
      };
      window.Element.prototype.scrollIntoView = () => {};
      // jsdom does not decode images: fake a loaded 800×600 bitmap
      window.Image = function FakeImage() {
        const img = window.document.createElement("img");
        setTimeout(() => {
          Object.defineProperty(img, "naturalWidth", { value: 800, configurable: true });
          Object.defineProperty(img, "naturalHeight", { value: 600, configurable: true });
          if (typeof img.onload === "function") img.onload();
        }, 5);
        return img;
      };
    },
  });

  const { window } = dom;
  const doc = window.document;
  const $ = (selector) => doc.querySelector(selector);
  let failures = 0;
  const check = (name, ok, extra = "") => {
    if (!ok) failures += 1;
    console.log(`${ok ? "ok  " : "FAIL"}  ${name}${extra ? " — " + extra : ""}`);
  };

  await fetch(`${BASE}api/jobs`, { method: "DELETE" }).catch(() => {});
  await wait(1800);

  const input = $("#file-input");
  const file = new File([PNG], "photo.png", { type: "image/png" });
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  input.dispatchEvent(new window.Event("change", { bubbles: true }));
  await wait(300);

  check("base image accepted", $("#dropzone").hidden === true && $("#layers").hidden === false);
  check("stage meta shows the input size", $("#stage-meta").textContent.includes("800×600"), $("#stage-meta").textContent);
  check("output size follows the input aspect", $("#size-out").textContent === "1176 × 880", $("#size-out").textContent);

  $("#prompt").value = "turn the background into a rainy neon street";
  $("#prompt").dispatchEvent(new window.Event("input", { bubbles: true }));
  check("generate enabled with image and prompt", $("#btn-generate").disabled === false);

  $("#edit-form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await wait(400);
  check("generation starts", $("#btn-generate").classList.contains("is-busy"));
  check("stop button appears", $("#btn-cancel").hidden === false);

  for (let attempt = 0; attempt < 100 && $("#btn-generate").classList.contains("is-busy"); attempt += 1) {
    await wait(200);
  }
  check("generation finishes", !$("#btn-generate").classList.contains("is-busy"));
  check("result image is set", ($("#img-result").getAttribute("src") || "").includes("/api/jobs/"), $("#img-result").getAttribute("src"));
  check("save enabled", $("#btn-save").disabled === false);
  check("seed readout filled", /seed \d+/.test($("#seed-readout").textContent), $("#seed-readout").textContent);
  check("seed input updated", Number($("#seed").value) > 0, $("#seed").value);

  $("#btn-gallery").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await wait(600);
  check("gallery lists the finished job", doc.querySelectorAll(".gallery-item").length === 1, String(doc.querySelectorAll(".gallery-item").length));
  check("gallery badge counts results", $("#gallery-count").textContent === "1", $("#gallery-count").textContent);

  doc.querySelector('.seg[data-view="compare"]').dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await wait(100);
  check("compare overlay visible", $("#compare").hidden === false);
  check("compare handle centred", $("#compare-handle").style.left === "50%", $("#compare-handle").style.left);

  $("#btn-reuse").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await wait(800);
  check("result reused as input", $("#layers").hidden === false && $("#btn-save").disabled === false);

  check("no script errors during the flow", errors.length === 0, errors.join(" | ").slice(0, 500));
  console.log(failures ? `\n${failures} check(s) failed` : "\nflow suite: all checks passed");
  process.exit(failures ? 1 : 0);
})().catch((error) => {
  console.error("harness failure", error);
  process.exit(2);
});
