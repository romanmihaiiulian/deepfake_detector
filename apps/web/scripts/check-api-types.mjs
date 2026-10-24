// Fails when src/api/schema.d.ts is not what openapi-typescript generates from openapi.json.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "api-types-"));
try {
  const fresh = join(dir, "schema.d.ts");
  execFileSync("node_modules/.bin/openapi-typescript", ["openapi.json", "-o", fresh], {
    stdio: "ignore",
  });
  if (readFileSync(fresh, "utf8") !== readFileSync("src/api/schema.d.ts", "utf8")) {
    console.error("src/api/schema.d.ts is out of date: run `make gen-api`");
    process.exit(1);
  }
  console.log("src/api/schema.d.ts is up to date");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
