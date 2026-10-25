#!/usr/bin/env node
/**
 * Dependency-free frontend smoke test.
 *
 *  - every `$("#id")` used by app.js must exist in index.html
 *  - the HTML must be CSP-safe (no inline handlers, no inline <script> body)
 *  - every referenced asset must exist on disk
 *  - styles.css must define the class hooks the markup relies on
 *
 * Run: node tests/frontend_smoke.cjs
 */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const web = path.join(root, "web");
const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
const js = fs.readFileSync(path.join(web, "app.js"), "utf8");
const css = fs.readFileSync(path.join(web, "styles.css"), "utf8");

let failures = 0;
const fail = (message) => {
  failures += 1;
  console.error(`  FAIL  ${message}`);
};
const ok = (message) => console.log(`  ok    ${message}`);

// 1. ids referenced from JS exist in the markup
const htmlIds = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
const jsIds = new Set([...js.matchAll(/\$\("#([A-Za-z][\w-]*)"/g)].map((match) => match[1]));
const missingIds = [...jsIds].filter((id) => !htmlIds.has(id));
if (missingIds.length) fail(`app.js references missing ids: ${missingIds.join(", ")}`);
else ok(`${jsIds.size} element ids referenced by app.js all exist`);

// 2. no CSP-hostile inline script or inline event handlers
if (/<script(?![^>]*\bsrc=)[^>]*>[\s\S]*?\S[\s\S]*?<\/script>/.test(html)) fail("index.html contains an inline <script>");
else ok("no inline scripts");
const inlineHandlers = [...html.matchAll(/\son[a-z]+\s*=\s*"/gi)].map((match) => match[0]);
if (inlineHandlers.length) fail(`inline event handlers found: ${inlineHandlers.join(", ")}`);
else ok("no inline event handlers");

// 3. referenced local assets exist
const assets = [...html.matchAll(/(?:href|src)="(\/[^"]+)"/g)].map((match) => match[1]);
for (const asset of new Set(assets)) {
  const candidate = asset === "/favicon.svg" ? path.join(root, "favicon.svg") : path.join(web, asset.slice(1));
  if (!fs.existsSync(candidate)) fail(`asset missing on disk: ${asset}`);
}
ok(`${new Set(assets).size} local assets resolved`);

// 4. class hooks present in CSS
const requiredClasses = [
  "topbar", "layout", "stage", "canvas", "dropzone", "layers", "compare", "panel",
  "style-chip", "aspect-btn", "perf-btn", "gallery-grid", "toast", "drawer", "modal",
];
const missingClasses = requiredClasses.filter((name) => !new RegExp(`\\.${name}\\b`).test(css));
if (missingClasses.length) fail(`missing CSS classes: ${missingClasses.join(", ")}`);
else ok(`${requiredClasses.length} layout classes are styled`);

// 5. basic sanity: form posts through fetch, no hardcoded localhost
if (/http:\/\/localhost|http:\/\/127\.0\.0\.1/.test(js)) fail("app.js hardcodes localhost (use relative URLs)");
else ok("frontend uses relative API URLs");

// 6. accessibility basics
if (!html.includes('lang="en"')) fail("html element is missing lang");
if (!/<label[^>]*for="/.test(html)) fail("no label/for pairs found");
if (!html.includes('aria-live="polite"')) fail("toasts are not announced");
ok("accessibility hooks present");

console.log(failures ? `\n${failures} check(s) failed` : "\nfrontend smoke: all checks passed");
process.exit(failures ? 1 : 0);
