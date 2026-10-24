import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { UploadPage } from "./UploadPage";

describe("UploadPage", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({ status: "ok", models_loaded: true, analyzer: "fake", device: "none", queue_depth: 0, version: "0.1.0" }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("rejects unsupported files before uploading", async () => {
    render(<UploadPage />);
    const input = screen.getByTestId("file-input");
    await userEvent.upload(input, new File(["x"], "clip.avi", { type: "video/x-msvideo" }), {
      applyAccept: false,
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("Please choose an MP4, MOV or WEBM video.");
  });

  it("announces demo mode", async () => {
    render(<UploadPage />);
    expect(await screen.findByText(/Demo mode: no detection model runs on this server/)).toBeInTheDocument();
  });

  it("explains what happens to the video", () => {
    render(<UploadPage />);
    expect(screen.getByText(/deleted as soon as the analysis finishes/)).toBeInTheDocument();
  });
});
