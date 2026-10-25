#!/usr/bin/env node
/**
 * Runs the browser-level smoke tests against a real (mock-engine) server.
 *   cd tests/ui && npm install && npm test
 */
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || "8123";
const BASE = `http://127.0.0.1:${PORT}/`;
const root = path.resolve(__dirname, "..", "..");
const python = process.env.PYTHON || (fs.existsSync(path.join(root, ".venv", "bin", "python")) ? path.join(root, ".venv", "bin", "python") : "python3");

if (!fs.existsSync(path.join(__dirname, "node_modules", "jsdom"))) {
  console.log("jsdom is not installed. Run: cd tests/ui && npm install");
  process.exit(2);
}

const server = spawn(python, ["server.py", "--mock", "--port", PORT], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
let serverLog = "";
server.stdout.on("data", (chunk) => (serverLog += chunk));
server.stderr.on("data", (chunk) => (serverLog += chunk));

const waitForServer = async () => {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(`${BASE}api/health`);
      if (response.ok) return true;
    } catch (_) {
      /* not up yet */
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
};

(async () => {
  const ready = await waitForServer();
  if (!ready) {
    console.error("Server did not start:\n" + serverLog.slice(-3000));
    server.kill();
    process.exit(1);
  }

  let failures = 0;
  for (const script of ["boot.js", "flow.js"]) {
    console.log(`\n── ${script} ─────────────────────────────────────────────`);
    const result = spawnSync(process.execPath, [path.join(__dirname, script)], {
      cwd: __dirname,
      stdio: "inherit",
      env: { ...process.env, BASE },
    });
    if (result.status !== 0) failures += 1;
  }

  server.kill();
  console.log(failures ? `\n${failures} UI suite(s) failed` : "\nUI suites passed");
  process.exit(failures ? 1 : 0);
})();
