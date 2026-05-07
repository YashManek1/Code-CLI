/**
 * debug-proxy.mjs
 * Run this INSTEAD of your real proxy to see exactly what the CLI sends after trust.
 * Usage: node debug-proxy.mjs
 * Then in another terminal: run your CLI
 */

import http from "http";

const PORT = 8082;
const LOG_FILE = "debug-proxy-log.txt";
import { appendFileSync, writeFileSync } from "fs";

// Clear log on start
writeFileSync(
  LOG_FILE,
  `=== Debug Proxy started at ${new Date().toISOString()} ===\n`,
);

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  appendFileSync(LOG_FILE, line + "\n");
}

const server = http.createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
  });
  req.on("end", () => {
    log(`\n--- INCOMING REQUEST ---`);
    log(`METHOD: ${req.method}`);
    log(`PATH:   ${req.url}`);
    log(`HEADERS: ${JSON.stringify(req.headers, null, 2)}`);
    if (body) {
      try {
        log(`BODY: ${JSON.stringify(JSON.parse(body), null, 2)}`);
      } catch {
        log(`BODY (raw): ${body.slice(0, 500)}`);
      }
    }

    // Route-specific mock responses so CLI doesn't hang
    const path = req.url;

    // HEAD / GET health check
    if (req.method === "HEAD" || (req.method === "GET" && path === "/")) {
      log(`RESPONSE: 200 health check`);
      res.writeHead(200);
      res.end();
      return;
    }

    // Bootstrap — CLI calls this right after trust to get account info
    if (path.includes("/bootstrap") || path.includes("/api/bootstrap")) {
      log(`RESPONSE: 200 bootstrap mock`);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          client_data: null,
          additional_model_options: [],
        }),
      );
      return;
    }

    // Models list
    if (path.includes("/models") || path.includes("/v1/models")) {
      log(`RESPONSE: 200 models mock`);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          data: [
            { id: "claude-sonnet-4-20250514", object: "model" },
            { id: "claude-opus-4-20250514", object: "model" },
          ],
        }),
      );
      return;
    }

    // OAuth / profile endpoints
    if (path.includes("/oauth") || path.includes("/profile")) {
      log(`RESPONSE: 200 oauth/profile mock`);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ id: "mock-user", email: "mock@mock.com" }));
      return;
    }

    // Policy / limits
    if (
      path.includes("/policy") ||
      path.includes("/limits") ||
      path.includes("/quota")
    ) {
      log(`RESPONSE: 200 policy mock`);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ allowed: true, limits: {} }));
      return;
    }

    // Messages — actual API call
    if (path.includes("/messages")) {
      log(`RESPONSE: 200 messages mock (returning simple text)`);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          id: "msg_mock",
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "Mock response from debug proxy." }],
          model: "claude-sonnet-4-20250514",
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
        }),
      );
      return;
    }

    // Anything else — return 200 empty so CLI doesn't hang
    log(`RESPONSE: 200 (catch-all for unknown path)`);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`\n✅ Debug proxy listening on http://127.0.0.1:${PORT}`);
  console.log(`📄 Logging all requests to: ${LOG_FILE}`);
  console.log(`\nNow run your CLI in another terminal:`);
  console.log(`  $env:ANTHROPIC_API_KEY="ccnim"`);
  console.log(`  $env:ANTHROPIC_BASE_URL="http://127.0.0.1:8082"`);
  console.log(`  $env:CLAUDE_DISABLE_AUTOUPDATE="1"`);
  console.log(`  # DO NOT set NODE_ENV or ANTHROPIC_AUTH_TOKEN`);
  console.log(`  bun run src/entrypoints/cli.tsx\n`);
});
