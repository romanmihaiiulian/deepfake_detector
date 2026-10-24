import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { makeResult } from "../test/fixtures";
import { Timeline } from "./Timeline";

describe("Timeline", () => {
  it("draws one line per track and shades suspicious intervals", () => {
    render(<Timeline result={makeResult()} />);
    expect(screen.getAllByTestId("track-line")).toHaveLength(2);
    expect(screen.getAllByTestId("interval")).toHaveLength(1);
    expect(screen.getByRole("img")).toHaveAccessibleName(
      "Manipulation score over time for 2 face tracks, with 1 suspicious interval.",
    );
  });

  it("breaks the line where a frame was not scored", () => {
    render(<Timeline result={makeResult()} />);
    const path = screen.getAllByTestId("track-line")[0]!.getAttribute("d") ?? "";
    expect(path.match(/M/g)).toHaveLength(2);
  });

  it("offers an accessible table view", async () => {
    render(<Timeline result={makeResult()} />);
    await userEvent.click(screen.getByRole("button", { name: "Show as table" }));
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(1 + 16);
    expect(screen.getByText("not scored")).toBeInTheDocument();
  });

  it("explains when no face was tracked", () => {
    render(<Timeline result={makeResult({ tracks: [], suspicious_intervals: [] })} />);
    expect(screen.getByText(/No face was tracked/)).toBeInTheDocument();
  });
});
