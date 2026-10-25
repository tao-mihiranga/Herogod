#!/usr/bin/env node
/**
 * Boot smoke test: load the studio in jsdom against a running (mock) backend and
 * verify the first render plus the main interactions. No GPU, no weights, no
 * real browser needed.
 *
 *   BASE=http://127.0.0.1:8123/ node boot.js
 */
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = (process.env.BASE || "http://127.0.0.1:8123/").replace(/\/?$/, "/");
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
      window.URL.createObjectURL = () => "blob:ui-test";
      window.URL.revokeObjectURL = () => {};
      window.HTMLDialogElement.prototype.showModal = function showModal() {
        this.open = true;
      };
      window.HTMLDialogElement.prototype.close = function close() {
        this.open = false;
      };
      window.Element.prototype.scrollIntoView = () => {};
      window.scrollTo = () => {};
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
  await wait(2000);

  check("boot produced no script errors", errors.length === 0, errors.join(" | ").slice(0, 400));
  check("style chips rendered", doc.querySelectorAll(".style-chip").length >= 10, String(doc.querySelectorAll(".style-chip").length));
  check("aspect buttons rendered", doc.querySelectorAll(".aspect-btn").length >= 5);
  check("performance presets wired", Array.from(doc.querySelectorAll(".perf-btn")).every((button) => button.dataset.wired === "1"));
  check("engine label updated", /model|ready|idle|loading|error/i.test($("#engine-label").textContent), $("#engine-label").textContent);
  check("engine facts populated", $("#engine-kv").children.length >= 8, String($("#engine-kv").children.length));
  check("adapter selected", $("#adapter").options.length >= 1, $("#adapter").value);
  check("generate disabled without an image", $("#btn-generate").disabled === true);
  check("output size computed", /\d+ × \d+/.test($("#size-out").textContent), $("#size-out").textContent);

  const chip = doc.querySelector('.style-chip[data-style="cinematic"]');
  chip.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  check("style chip toggles", chip.getAttribute("aria-pressed") === "true");

  doc.querySelector('.perf-btn[data-perf="turbo"]').dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  check("performance preset sets steps", $("#steps").value === "8", $("#steps").value);

  $("#prompt").value = "test prompt";
  $("#prompt").dispatchEvent(new window.Event("input", { bubbles: true }));
  check("prompt counter updates", $("#prompt-count").textContent === "11 / 2000", $("#prompt-count").textContent);

  doc.querySelector('.seg[data-view="input"]').dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  check("view switch applies", $("#canvas").dataset.view === "input");

  $("#btn-gallery").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await wait(400);
  check("gallery drawer opens", $("#gallery").hidden === false);
  check("gallery empty state shown", $("#gallery-empty").hidden === false);

  $("#btn-settings").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await wait(300);
  check("engine drawer opens", $("#settings").hidden === false);
  check("engine drawer lists runtime facts", $("#engine-kv").textContent.includes("mock"), $("#engine-kv").textContent.slice(0, 80));

  check("no script errors after interactions", errors.length === 0, errors.join(" | ").slice(0, 400));
  console.log(failures ? `\n${failures} check(s) failed` : "\nboot suite: all checks passed");
  process.exit(failures ? 1 : 0);
})().catch((error) => {
  console.error("harness failure", error);
  process.exit(2);
});
