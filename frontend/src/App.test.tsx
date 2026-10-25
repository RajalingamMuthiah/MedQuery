import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AskResponse } from "./api";
import { Answer, STATUS_STYLE } from "./components/Answer";
import { PassagePanel } from "./components/PassagePanel";

const passage = {
  id: "htn-jnc7-express:0026",
  doc_id: "htn-jnc7-express",
  title: "JNC 7 Express",
  publisher: "NHLBI",
  as_of: "2003",
  superseded: true,
  superseded_note: "Superseded by later hypertension guidance.",
  page_start: 3,
  page_end: 3,
  section: "Classification",
  kind: "text",
  text: "Prehypertension is 120-139 mmHg systolic. Other text.",
};

const response: AskResponse = {
  question: "What is prehypertension?",
  answer: "Prehypertension is 120-139 mmHg systolic [htn-jnc7-express:0026]. This is superseded.",
  abstained: false,
  sentences: [
    {
      text: "Prehypertension is 120-139 mmHg systolic [htn-jnc7-express:0026].",
      citations: ["htn-jnc7-express:0026"],
      highlights: { "htn-jnc7-express:0026": [[0, 41]] },
    },
    { text: "This is superseded.", citations: [], highlights: {} },
  ],
  passages: [passage],
  latency: { retrieval_ms: 15, generation_s: 0.8 },
  model: "openai/gpt-oss-20b",
  notice: "Not medical advice.",
};

describe("Answer", () => {
  it("shows citation chips (with a superseded flag) and opens the source on click", async () => {
    const onCite = vi.fn();
    render(<Answer response={response} onCite={onCite} />);
    expect(screen.getByText("Prehypertension is 120-139 mmHg systolic.")).toBeInTheDocument();
    const chip = screen.getByRole("button", { name: /JNC 7 Express, page 3, superseded/ });
    await userEvent.click(chip);
    expect(onCite).toHaveBeenCalledWith(passage, 0);
  });

  it("renders an abstention as 'the documents don't cover this', not an error", () => {
    render(<Answer response={{ ...response, abstained: true, sentences: [] }} onCite={() => {}} />);
    expect(screen.getByRole("heading", { name: "The documents don't cover this" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("colours and labels sentences by verification status", () => {
    const { container } = render(
      <Answer response={response} statuses={["unsupported", "note"]} onCite={() => {}} />,
    );
    const [first, second] = container.querySelectorAll("p[data-status]");
    expect(first).toHaveAttribute("data-status", "unsupported");
    expect(first.className).toContain("line-through");
    expect(second).toHaveTextContent("Note");
    expect(STATUS_STYLE.supported.className).toContain("green");
    expect(STATUS_STYLE.unverified.className).toContain("amber");
  });
});

describe("PassagePanel", () => {
  it("shows document, page, section, the superseded warning and highlights", () => {
    render(<PassagePanel passage={passage} highlights={[[0, 41]]} onClose={() => {}} />);
    expect(screen.getByRole("dialog", { name: "JNC 7 Express" })).toBeInTheDocument();
    expect(screen.getByText(/NHLBI · as of 2003 · page 3/)).toBeInTheDocument();
    expect(screen.getByText(/Superseded guidance \(2003\)/)).toBeInTheDocument();
    expect(screen.getByText("Prehypertension is 120-139 mmHg systolic.").tagName).toBe("MARK");
  });
});
