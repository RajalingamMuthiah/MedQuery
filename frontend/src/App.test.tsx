import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, type AskResponse, type Passage } from "./api";
import { AbstentionCard, AnswerCard, ErrorCard, SourcesList } from "./components/AnswerCard";
import { PassagePanel } from "./components/PassagePanel";
import { VerifyBar } from "./components/VerifyBar";
import { numberCitations, stripCitations } from "./lib/citations";
import { friendlyError } from "./lib/errors";
import { parseRoute } from "./lib/route";
import { verifySummary } from "./lib/status";

const jnc: Passage = {
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
const dash: Passage = {
  ...jnc,
  id: "htn-dash:0004",
  doc_id: "htn-dash",
  title: "Your Guide to Lowering Your Blood Pressure with DASH",
  publisher: "NHLBI",
  as_of: "2006",
  superseded: false,
  superseded_note: "",
  page_start: 8,
  page_end: 9,
  text: "Eat less sodium.",
};

const response: AskResponse = {
  question: "What is prehypertension?",
  answer: "…",
  abstained: false,
  sentences: [
    {
      text: "Prehypertension is 120-139 mmHg systolic [htn-dash:0004][htn-jnc7-express:0026].",
      citations: ["htn-dash:0004", "htn-jnc7-express:0026"],
      highlights: { "htn-jnc7-express:0026": [[0, 41]] },
    },
    {
      text: "This guidance is superseded [htn-jnc7-express:0026].",
      citations: ["htn-jnc7-express:0026"],
      highlights: {},
    },
  ],
  passages: [jnc, dash],
  latency: { retrieval_ms: 18.4, generation_s: 0.86 },
  model: "openai/gpt-oss-20b",
  notice: "Not medical advice.",
};

describe("numbered citations", () => {
  it("numbers cited passages by first appearance and strips the raw markers", () => {
    expect(numberCitations(response.sentences, response.passages).map((s) => [s.number, s.passage.id])).toEqual([
      [1, "htn-dash:0004"],
      [2, "htn-jnc7-express:0026"],
    ]);
    expect(stripCitations(response.sentences[0].text)).toBe("Prehypertension is 120-139 mmHg systolic.");
  });

  it("renders superscript [1] [2] and opens the right passage on click", async () => {
    const onCite = vi.fn();
    render(<AnswerCard response={response} onCite={onCite} />);
    const two = screen.getAllByRole("button", { name: /^Source 2: JNC 7 Express, p\. 3$/ });
    expect(two).toHaveLength(2); // cited by both sentences, same number
    expect(screen.getByRole("button", { name: /^Source 1:/ })).toHaveTextContent("[1]");
    await userEvent.click(two[0]);
    expect(onCite).toHaveBeenCalledWith(jnc, 0, expect.any(HTMLElement));
    expect(screen.getByText(/Retrieved in 18 ms · generated in 0\.9 s/)).toBeInTheDocument();
  });
});

describe("sources list", () => {
  it("lists each numbered source with title, publisher, year, page and a Superseded badge", async () => {
    const onOpen = vi.fn();
    render(<SourcesList response={response} onOpen={onOpen} />);
    const items = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("1Your Guide to Lowering Your Blood Pressure with DASHNHLBI · 2006 · pp. 8–9");
    expect(items[0]).not.toHaveTextContent("Superseded");
    expect(items[1]).toHaveTextContent("JNC 7 Express");
    expect(items[1]).toHaveTextContent("NHLBI · 2003 · p. 3");
    expect(items[1]).toHaveTextContent("Superseded");
    await userEvent.click(within(items[1]).getByRole("button"));
    expect(onOpen).toHaveBeenCalledWith(jnc, expect.any(HTMLElement));
  });
});

describe("abstention", () => {
  it("is a gentle card with a link to Sources, never an error", () => {
    render(<AbstentionCard />);
    expect(screen.getByRole("heading", { name: "Not covered by the documents" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /See what the sources cover/ })).toHaveAttribute("href", "#/sources");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("verification", () => {
  it("summarises statuses, counting notes separately", () => {
    expect(verifySummary(["supported", "note"])).toBe("1 of 1 sentence supported · 1 note");
    expect(verifySummary(["supported", "unsupported", "unverified", "supported"])).toBe("2 of 4 sentences supported");
  });

  it("shows the note, the summary and a legend, and marks each sentence with a dot and a label", () => {
    const result = {
      sentences: [
        { text: "a", status: "supported" as const },
        { text: "b", status: "unsupported" as const },
      ],
      verified: true,
      verifier: "verify_v2:x",
      latency_s: 1,
    };
    render(<VerifyBar result={result} checking={false} disabled={false} onCheck={() => {}} />);
    expect(screen.getByText("can take up to a minute")).toBeInTheDocument();
    expect(screen.getByText("1 of 2 sentences supported")).toBeInTheDocument();
    const legend = screen.getByRole("list", { name: "Legend" });
    expect(legend).toHaveTextContent("Supported");
    expect(legend).toHaveTextContent("Unsupported");
    expect(legend).toHaveTextContent("Unverified");
    expect(legend).toHaveTextContent("Note");

    const { container } = render(
      <AnswerCard response={response} statuses={["supported", "unsupported"]} onCite={() => {}} />,
    );
    const [first, second] = container.querySelectorAll("p[data-status]");
    expect(first).toHaveAttribute("data-status", "supported");
    expect(within(first as HTMLElement).getByLabelText("Supported")).toBeInTheDocument();
    expect(second).toHaveAttribute("data-status", "unsupported");
    expect(second.innerHTML).toContain("line-through");
  });

  it("says when the check could not be completed, and shows progress while checking", () => {
    const failed = { sentences: [{ text: "a", status: "unverified" as const }], verified: false, verifier: "v", latency_s: 1 };
    const { rerender } = render(<VerifyBar result={failed} checking={false} disabled={false} onCheck={() => {}} />);
    expect(screen.getByText("Unverified: the check could not be completed")).toBeInTheDocument();
    rerender(<VerifyBar result={null} checking={true} disabled={false} onCheck={() => {}} />);
    expect(screen.getByRole("button", { name: /Checking/ })).toHaveAttribute("aria-busy", "true");
  });
});

describe("errors and routing", () => {
  it("turns API and network errors into friendly messages with a retry", async () => {
    expect(friendlyError(new ApiError(429, "the daily request limit has been reached"))).toBe(
      "Daily demo limit reached — try again tomorrow.",
    );
    expect(friendlyError(new ApiError(429, "too many requests"))).toMatch(/wait a minute/);
    expect(friendlyError(new ApiError(422, "the question is empty"))).toBe("the question is empty");
    expect(friendlyError(new TypeError("Failed to fetch"))).toMatch(/Can't reach the MedQuery server/);
    const retry = vi.fn();
    render(<ErrorCard message="Daily demo limit reached — try again tomorrow." onRetry={retry} />);
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Daily demo limit reached");
  });

  it("reads the page from the URL hash", () => {
    expect([parseRoute(""), parseRoute("#/sources"), parseRoute("#/about"), parseRoute("#/x")]).toEqual([
      "ask",
      "sources",
      "about",
      "ask",
    ]);
  });
});

describe("PassagePanel", () => {
  it("shows document, page, section, the superseded warning and highlights", () => {
    render(<PassagePanel passage={jnc} number={2} highlights={[[0, 41]]} onClose={() => {}} />);
    expect(screen.getByRole("dialog", { name: "JNC 7 Express" })).toBeInTheDocument();
    expect(screen.getByText("Source 2")).toBeInTheDocument();
    expect(screen.getByText(/NHLBI · as of 2003 · page 3 · Classification/)).toBeInTheDocument();
    expect(screen.getByText(/Superseded guidance \(2003\)/)).toBeInTheDocument();
    expect(screen.getByText("Prehypertension is 120-139 mmHg systolic.").tagName).toBe("MARK");
  });
});
