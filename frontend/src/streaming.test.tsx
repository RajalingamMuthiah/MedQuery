import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { askStream, type AskResponse } from "./api";
import { AnswerCard, DraftAnswer } from "./components/AnswerCard";
import { VerifyBar } from "./components/VerifyBar";
import { createSSEParser, mayBeAbstention } from "./lib/sse";

const passage = {
  id: "tb-ltbi-2020:0020",
  doc_id: "tb-ltbi-2020",
  title: "Latent TB Infection Treatment",
  publisher: "CDC",
  as_of: "2020",
  superseded: false,
  superseded_note: "",
  page_start: 8,
  page_end: 8,
  section: "Regimens",
  kind: "text",
  text: "Rifampin daily for 4 months.",
};

const final: AskResponse = {
  question: "How long is 4R?",
  answer: "Rifampin is given daily for 4 months [tb-ltbi-2020:0020].",
  abstained: false,
  sentences: [
    {
      text: "Rifampin is given daily for 4 months [tb-ltbi-2020:0020].",
      citations: ["tb-ltbi-2020:0020"],
      highlights: {},
    },
  ],
  passages: [passage],
  latency: { retrieval_ms: 15, generation_s: 0.9, ttft_s: 0.3 },
  model: "openai/gpt-oss-20b",
  cached: false,
  notice: "Not medical advice.",
};

const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/** A streamed Response we can push events into, one at a time. */
function controlledStream() {
  let push!: (text: string) => void;
  let close!: () => void;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const enc = new TextEncoder();
      push = (text) => controller.enqueue(enc.encode(text));
      close = () => controller.close();
    },
  });
  return { response: new Response(body, { status: 200 }), push, close };
}

function json(data: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(data), { status }));
}

afterEach(() => vi.unstubAllGlobals());

describe("SSE parsing and the abstention prefix", () => {
  it("emits events only when complete, across chunk boundaries", () => {
    const seen: string[] = [];
    const feed = createSSEParser((e) => seen.push(`${e.event}:${e.data}`));
    feed('event: delta\ndata: {"text":"Ri');
    expect(seen).toEqual([]);
    feed('f"}\n\nevent: final\r\ndata: {}\r\n\r\n');
    expect(seen).toEqual(['delta:{"text":"Rif"}', "final:{}"]);
  });

  it("hides a draft while it could still be the abstention sentence", () => {
    expect(mayBeAbstention("I can")).toBe(true);
    expect(mayBeAbstention("I cannot answer this from the")).toBe(true);
    expect(mayBeAbstention("Rifampin is")).toBe(false);
  });
});

describe("askStream", () => {
  it("passes deltas through and resolves with the final response", async () => {
    const body = sse("retrieval", { passages: [] }) + sse("delta", { text: "Rifampin " }) +
      sse("delta", { text: "daily" }) + sse("final", final);
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(body))));
    const deltas: string[] = [];
    const result = await askStream("q?", { onDelta: (t) => deltas.push(t) });
    expect(deltas).toEqual(["Rifampin ", "daily"]);
    expect(result.answer).toBe(final.answer);
  });

  it("turns an error event into an ApiError", async () => {
    const body = sse("error", { error: "the language model could not be reached", status: 503 });
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(body))));
    await expect(askStream("q?", { onDelta: () => {} })).rejects.toMatchObject({ status: 503 });
  });
});

describe("states", () => {
  it("shows the streaming draft without citation markers, even half-typed ones", () => {
    render(<DraftAnswer text="Rifampin daily [tb-ltbi-2020:0020]. Then [tb-l" />);
    expect(screen.getByRole("article", { name: /still arriving/ })).toHaveTextContent("Rifampin daily. Then");
    expect(screen.getByRole("article")).not.toHaveTextContent("[");
  });

  it("labels a cached answer subtly", () => {
    render(<AnswerCard response={{ ...final, cached: true }} onCite={() => {}} />);
    expect(screen.getByTestId("cached-label")).toHaveTextContent("Cached answer");
  });

  it("explains the rate-limit wait while checking", () => {
    render(<VerifyBar result={null} checking={true} disabled={false} onCheck={() => {}} waitS={12.2} />);
    expect(screen.getByText("Waiting about 13 s for the rate limit, then checking…")).toBeInTheDocument();
  });
});

function routes(handlers: Record<string, () => Promise<Response>>) {
  return vi.fn((url: string) => {
    const path = new URL(url).pathname;
    const handler = handlers[path];
    if (!handler) throw new Error(`unexpected fetch ${path}`);
    return handler();
  });
}

describe("App", () => {
  it("shows 'starting up' and disables asking until the server is ready", async () => {
    vi.stubGlobal("fetch", routes({
      "/health": () => json({ status: "starting", ready: false, error: null, daily_remaining: 60 }),
      "/samples": () => json({ samples: [] }),
    }));
    render(<App />);
    expect(await screen.findByText(/Starting up… loading the documents/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("streams a draft, then replaces it with the numbered answer card", async () => {
    const stream = controlledStream();
    vi.stubGlobal("fetch", routes({
      "/health": () => json({ status: "ok", ready: true, error: null, daily_remaining: 60 }),
      "/samples": () => json({ samples: [] }),
      "/ask/stream": () => Promise.resolve(stream.response),
    }));
    render(<App />);
    await waitFor(() => expect(screen.queryByText(/Starting up/)).toBeNull());
    await userEvent.type(screen.getByLabelText(/Ask about/), "How long is 4R?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    stream.push(sse("delta", { text: "Rifampin is given " }));
    expect(await screen.findByRole("article", { name: /still arriving/ })).toHaveTextContent("Rifampin is given");
    stream.push(sse("final", final));
    stream.close();
    expect(await screen.findByRole("button", { name: /^Source 1: Latent TB/ })).toHaveTextContent("[1]");
    expect(screen.queryByRole("article", { name: /still arriving/ })).toBeNull();
  });

  it("never shows an abstention as a draft; it ends as the 'not covered' card", async () => {
    const stream = controlledStream();
    vi.stubGlobal("fetch", routes({
      "/health": () => json({ status: "ok", ready: true, error: null, daily_remaining: 60 }),
      "/samples": () => json({ samples: [] }),
      "/ask/stream": () => Promise.resolve(stream.response),
    }));
    render(<App />);
    await waitFor(() => expect(screen.queryByText(/Starting up/)).toBeNull());
    await userEvent.type(screen.getByLabelText(/Ask about/), "Mars?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    stream.push(sse("delta", { text: "I cannot answer " }));
    expect(await screen.findByRole("status", { name: "Finding an answer" })).toBeInTheDocument();
    expect(screen.queryByText(/I cannot answer/)).toBeNull();
    stream.push(sse("final", { ...final, abstained: true, sentences: [], answer: "I cannot answer this from the provided documents." }));
    stream.close();
    expect(await screen.findByRole("heading", { name: "Not covered by the documents" })).toBeInTheDocument();
  });
});
