import { useEffect, useState } from "react";
import {
  ask,
  samples as loadSamples,
  verify,
  type AskResponse,
  type Passage,
  type Sample,
  type VerifyResponse,
} from "./api";
import { Answer, STATUS_STYLE } from "./components/Answer";
import { PassagePanel } from "./components/PassagePanel";
import { Sources } from "./components/Sources";

const NOTICE =
  "Not medical advice. MedQuery answers only from its source documents and does not diagnose or give dosing instructions. Consult a qualified health professional.";

export default function App() {
  const [page, setPage] = useState<"ask" | "sources">("ask");
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [check, setCheck] = useState<VerifyResponse | null>(null);
  const [open, setOpen] = useState<{ passage: Passage; sentence: number } | null>(null);
  const [busy, setBusy] = useState<"" | "ask" | "verify">("");
  const [error, setError] = useState("");
  const [sampleList, setSampleList] = useState<Sample[]>([]);

  useEffect(() => {
    loadSamples()
      .then((r) => setSampleList(r.samples))
      .catch(() => setSampleList([]));
  }, []);

  async function submit(q: string) {
    if (!q.trim()) return;
    setBusy("ask");
    setError("");
    setCheck(null);
    setOpen(null);
    try {
      setResponse(await ask(q));
    } catch (e) {
      setResponse(null);
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function runCheck() {
    if (!response) return;
    setBusy("verify");
    setError("");
    try {
      setCheck(
        await verify(response.question, response.answer, response.passages.map((p) => p.id)),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 pb-16">
      <div role="note" className="sticky top-0 z-10 -mx-4 bg-amber-100 px-4 py-2 text-sm text-amber-950">
        <strong>Not medical advice.</strong> {NOTICE.replace("Not medical advice. ", "")}
      </div>
      <header className="flex flex-wrap items-center justify-between gap-2 py-4">
        <h1 className="text-xl font-bold">MedQuery</h1>
        <nav aria-label="Main" className="flex gap-2">
          {(["ask", "sources"] as const).map((p) => (
            <button
              key={p}
              type="button"
              aria-current={page === p ? "page" : undefined}
              onClick={() => setPage(p)}
              className={`rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-600 ${page === p ? "bg-slate-900 text-white" : "border border-slate-400"}`}
            >
              {p === "ask" ? "Ask" : "Sources"}
            </button>
          ))}
        </nav>
      </header>

      {page === "sources" ? (
        <Sources />
      ) : (
        <main className="space-y-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submit(question);
            }}
            className="space-y-2"
          >
            <label htmlFor="question" className="block font-medium">
              Ask about diabetes, hypertension or tuberculosis
            </label>
            <textarea
              id="question"
              value={question}
              maxLength={500}
              rows={3}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full rounded border border-slate-400 p-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
            />
            <div className="flex items-center justify-between text-sm text-slate-600">
              <span>{question.length}/500</span>
              <button
                type="submit"
                disabled={busy !== "" || !question.trim()}
                className="rounded bg-blue-800 px-4 py-2 text-white disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                {busy === "ask" ? "Searching…" : "Ask"}
              </button>
            </div>
          </form>

          {sampleList.length > 0 && !response && (
            <section aria-labelledby="samples-title">
              <h2 id="samples-title" className="text-sm font-medium text-slate-700">
                Sample questions
              </h2>
              <ul className="mt-1 flex flex-wrap gap-2">
                {sampleList.map((s) => (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => {
                        setQuestion(s.question);
                        void submit(s.question);
                      }}
                      className="rounded border border-slate-400 bg-white px-2 py-1 text-left text-sm hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                    >
                      {s.question}
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {error && (
            <p role="alert" className="rounded border border-red-700 bg-red-50 p-2 text-red-900">
              {error}
            </p>
          )}

          <div aria-live="polite">
            {response && (
              <div className="space-y-2">
                <Answer
                  response={response}
                  statuses={check?.sentences.map((s) => s.status)}
                  onCite={(passage, sentence) => setOpen({ passage, sentence })}
                />
                {!response.abstained && (
                  <div className="flex flex-wrap items-center gap-3 text-sm">
                    <button
                      type="button"
                      onClick={() => void runCheck()}
                      disabled={busy !== ""}
                      className="rounded border border-slate-500 px-3 py-1 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-600"
                    >
                      {busy === "verify" ? "Checking…" : "Check this answer"}
                    </button>
                    {check && !check.verified && (
                      <span className="rounded bg-amber-200 px-2 py-0.5 text-amber-950">
                        Unverified: the check could not be completed
                      </span>
                    )}
                    {check && (
                      <span className="text-slate-600">
                        {Object.values(STATUS_STYLE)
                          .map((s) => s.label)
                          .join(" · ")}
                      </span>
                    )}
                    <span className="text-slate-600">
                      retrieval {response.latency.retrieval_ms} ms · generation{" "}
                      {response.latency.generation_s.toFixed(1)} s
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      )}

      {open && (
        <PassagePanel
          passage={open.passage}
          highlights={response?.sentences[open.sentence]?.highlights[open.passage.id] ?? []}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
