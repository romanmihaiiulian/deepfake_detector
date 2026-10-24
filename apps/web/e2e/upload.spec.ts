import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";

const sample = fileURLToPath(new URL("./fixtures/sample.mp4", import.meta.url));
const notes = fileURLToPath(new URL("./fixtures/notes.txt", import.meta.url));

test("upload a video, read the result, delete it", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Check a video for face manipulation" })).toBeVisible();
  await expect(page.getByText(/Demo mode: no detection model runs/)).toBeVisible();

  await page.getByTestId("file-input").setInputFiles(sample);
  await expect(page).toHaveURL(/\/analyses\/[0-9a-f]{32}$/);
  await expect(page.getByRole("progressbar")).toBeVisible();

  const verdict = page.getByRole("heading", {
    name: /^(Likely manipulated|Likely authentic|Inconclusive)$/,
  });
  await expect(verdict).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("status")).toHaveText(/Demo mode: this result is synthetic/);
  await expect(page.getByRole("meter", { name: "Manipulation score" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Manipulation score over time/ })).toBeVisible();

  const evidence = page.getByRole("img", { name: /^Face 1 at/ }).first();
  await expect(evidence).toBeVisible();
  expect(await evidence.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);

  const resultUrl = page.url();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete my data" }).click();
  await expect(page).toHaveURL(/\/$/);

  await page.goto(resultUrl);
  await expect(page.getByRole("heading", { name: "This analysis no longer exists" })).toBeVisible();
});

test("unsupported files are rejected before upload", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(notes);
  await expect(page.getByRole("alert")).toHaveText("Please choose an MP4, MOV or WEBM video.");
  await expect(page).toHaveURL(/\/$/);
});

test("the API rejects a renamed non-video", async ({ request }) => {
  const response = await request.post("/api/v1/analyses", {
    multipart: { file: { name: "fake.mp4", mimeType: "video/mp4", buffer: Buffer.from("nope".repeat(100)) } },
  });
  expect(response.status()).toBe(422);
  expect((await response.json()).error.code).toBe("unreadable");
});
