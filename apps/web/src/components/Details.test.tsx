import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeResult } from "../test/fixtures";
import { DetectorScores } from "./Details";

describe("DetectorScores", () => {
  it("lists raw detector scores and the uncalibrated note", () => {
    render(<DetectorScores result={makeResult()} />);
    expect(screen.getByRole("rowheader", { name: "GenD (frame detector)" })).toBeInTheDocument();
    expect(screen.queryByRole("rowheader", { name: /Calibrated/ })).not.toBeInTheDocument();
    expect(screen.getByText("Scores are raw detector probabilities.")).toBeInTheDocument();
  });

  it("shows the calibrated probability next to the raw scores", () => {
    const result = makeResult({
      manipulation_probability: 0.31,
      detector_scores: { gend: 0.86, temporal: 0.74 },
      calibration: {
        method: "platt",
        note: "manipulation_probability is calibrated on consented lab data.",
        inputs: ["gend", "temporal"],
        prior: 0.5,
        fitted_on: "run r1",
        raw_probability: 0.86,
      },
    });
    render(<DetectorScores result={result} />);
    expect(screen.getByRole("rowheader", { name: "Temporal model" })).toBeInTheDocument();
    const calibrated = screen.getByRole("rowheader", { name: "Calibrated (platt)" });
    expect(calibrated.nextElementSibling).toHaveTextContent("0.31");
    expect(screen.getByText(/calibrated on consented lab data/)).toBeInTheDocument();
  });
});
