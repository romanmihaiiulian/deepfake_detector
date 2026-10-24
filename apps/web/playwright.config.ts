import { defineConfig, devices } from "@playwright/test";

const PORT = 8765;

// End-to-end tests run against the real API in fake-analyzer mode (no model weights needed),
// serving the built SPA from dist/. Run `npm run build` first (`make e2e` does).
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `uv run deepfake-api --port ${PORT} --log-level warning`,
    url: `http://127.0.0.1:${PORT}/api/v1/health`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: {
      DFD_ANALYZER: "fake",
      DFD_FAKE_DELAY_S: "1.0",
      DFD_VAR_DIR: "test-results/var",
      DFD_WEB_DIST: "dist",
    },
  },
});
