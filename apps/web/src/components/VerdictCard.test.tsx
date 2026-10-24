import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Verdict } from "../api/types";
import { makeResult } from "../test/fixtures";
import { VerdictCard } from "./VerdictCard";

describe("VerdictCard", () => {
  it.each<[Verdict, string]>([
    ["manipulated_likely", "Likely manipulated"],
    ["authentic_likely", "Likely authentic"],
    ["inconclusive", "Inconclusive"],
  ])("shows a heading for %s", (verdict, title) => {
    render(<VerdictCard result={makeResult({ verdict })} />);
    expect(screen.getByRole("heading", { name: title })).toBeInTheDocument();
  });

  it("explains the score against the thresholds that were applied", () => {
    render(<VerdictCard result={makeResult()} />);
    const meter = screen.getByRole("meter", { name: "Manipulation score" });
    expect(meter).toHaveAttribute("aria-valuetext", "0.86");
    expect(screen.getByText(/At or above 0.80 the video is flagged/)).toBeInTheDocument();
    expect(screen.getByText("Medium confidence")).toBeInTheDocument();
    expect(screen.getByText("Video quality: Fair")).toBeInTheDocument();
  });

  it("translates reason codes into sentences", () => {
    render(<VerdictCard result={makeResult({ verdict: "inconclusive", verdict_reasons: ["no_face"] })} />);
    expect(screen.getByText("No face was visible long enough to analyze.")).toBeInTheDocument();
  });

  it("handles a missing score", () => {
    render(<VerdictCard result={makeResult({ manipulation_probability: null, verdict: "inconclusive" })} />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuetext", "not available");
  });
});
