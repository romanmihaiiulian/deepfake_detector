import { describe, expect, it } from "vitest";

import { parseRoute } from "../router";
import { formatSeconds } from "./text";
import { validateFile } from "./validation";

describe("validateFile", () => {
  it.each(["clip.mp4", "CLIP.MOV", "recording.webm"])("accepts %s", (name) => {
    expect(validateFile({ name, size: 1024 })).toBeNull();
  });

  it.each(["clip.avi", "clip", "notes.mp4.txt"])("rejects %s", (name) => {
    expect(validateFile({ name, size: 1024 })).toMatch(/MP4, MOV or WEBM/);
  });

  it("rejects empty and oversized files", () => {
    expect(validateFile({ name: "a.mp4", size: 0 })).toMatch(/empty/);
    expect(validateFile({ name: "a.mp4", size: 3 * 1024 * 1024 }, 2)).toMatch(/at most 2 MB/);
  });
});

describe("parseRoute", () => {
  it("recognises the two pages", () => {
    expect(parseRoute("/")).toEqual({ name: "upload" });
    expect(parseRoute("/analyses/0123456789abcdef0123456789abcdef")).toEqual({
      name: "analysis",
      id: "0123456789abcdef0123456789abcdef",
    });
    expect(parseRoute("/analyses/not-an-id")).toEqual({ name: "notFound" });
  });
});

describe("formatSeconds", () => {
  it("formats short and long times", () => {
    expect(formatSeconds(2.25)).toBe("2.3 s");
    expect(formatSeconds(75)).toBe("1:15.0");
  });
});
