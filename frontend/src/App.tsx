import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  ask,
  askStream,
  health as loadHealth,
  samples as loadSamples,
  verify,
  verifyEstimate,
  type AskResponse,
  type Passage,
  type Sample,
  type VerifyResponse,
} from "./api";
import { About } from "./components/About";
import {
  AbstentionCard,
  AnswerCard,
  AnswerSkeleton,
  DraftAnswer,
  ErrorCard,
  SourcesList,
  StartingUp,
} from "./components/AnswerCard";
import { AskBox, EmptyState } from "./components/Ask";
import { Banner, Footer, Header } from "./components/Layout";
import { PassagePanel } from "./components/PassagePanel";
import { Sources } from "./components/Sources";
import { VerifyBar } from "./components/VerifyBar";
import { numberCitations } from "./lib/citations";
import { friendlyError } from "./lib/errors";
import { useRoute } from "./lib/route";
import { mayBeAbstention } from "./lib/sse";

export type Startup = "starting" | "ready" | "error" | "unreachable";

/** Poll /health until the server reports ready (warm start), every `everyMs`. */
export function useStartup(everyMs = 2000): { state: Startup; error: string } {
  const [state, setState] = useState<Startup>("starting");
  const [error, setError] = useState("");
  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const h = await loadHealth();
        if (stop) return;
        if (h.ready) return setState("ready");
        if (h.status === "error") {
          setError(h.error ?? "");
          return setState("error");
        }
        setState("starting");
      } catch {
        if (!stop) setState("unreachable");
      }
      if (!stop) timer = setTimeout(poll, everyMs);
    };
    void poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [everyMs]);
  return { state, error };
}

const STARTUP_MESSAGE: Record<Exclude<Startup, "ready">, string> = {
  starting: "Starting up… loading the documents. This takes a few seconds.",
  error: "MedQuery could not start. Check the server log.",
  unreachable: "Can't reach the MedQuery server yet — retrying…",
};

interface OpenPassage {
  passage: Passage;
  sentence: number | null;
}

export default function App() {
  const route = useRoute();
  const startup = useStartup();
  const [draft, setDraft] = useState("");
  const [waitS, setWaitS] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [check, setCheck] = useState<VerifyResponse | null>(null);
  const [open, setOpen] = useState<OpenPassage | null>(null);
  const [busy, setBusy] = useState<"" | "ask" | "verify">("");
  const [error, setError] = useState<{ message: string; retry: () => void } | null>(null);
  const [sampleList, setSampleList] = useState<Sample[]>([]);
  const trigger = useRef<HTMLElement | null>(null);

  useEffect(() => {
    loadSamples()
      .then((r) => setSampleList(r.samples))
      .catch(() => setSampleList([]));
  }, []);

  const submit = useCallback(async (q: string) => {
    if (!q.trim()) return;
    setAsked(q.trim());
    setBusy("ask");
    setError(null);
    setCheck(null);
    setOpen(null);
    setResponse(null);
    setDraft("");
    try {
      setResponse(await askStream(q, { onDelta: (t) => setDraft((d) => d + t) }));
    } catch (e) {
      if (e instanceof ApiError) {
        setError({ message: friendlyError(e), retry: () => void submit(q) });
      } else {
        try {
          setResponse(await ask(q)); // streaming failed: fall back to the plain endpoint
        } catch (fallback) {
          setError({ message: friendlyError(fallback), retry: () => void submit(q) });
        }
      }
    } finally {
      setDraft("");
      setBusy("");
    }
  }, []);

  async function runCheck() {
    if (!response) return;
    setBusy("verify");
    setError(null);
    const ids = response.passages.map((p) => p.id);
    try {
      const estimate = await verifyEstimate(response.question, response.answer, ids).catch(() => null);
      setWaitS(estimate && estimate.wait_s > 2 ? estimate.wait_s : null);
      setCheck(await verify(response.question, response.answer, ids));
    } catch (e) {
      setError({ message: friendlyError(e), retry: () => void runCheck() });
    } finally {
      setWaitS(null);
      setBusy("");
    }
  }

  const openPassage = (passage: Passage, sentence: number | null, from: HTMLElement) => {
    trigger.current = from;
    setOpen({ passage, sentence });
  };
  const closePassage = useCallback(() => {
    setOpen(null);
    trigger.current?.focus(); // back to the citation that opened it
  }, []);

  const sources = response ? numberCitations(response.sentences, response.passages) : [];
  const highlightsFor = (o: OpenPassage): [number, number][] => {
    if (!response) return [];
    if (o.sentence !== null) return response.sentences[o.sentence]?.highlights[o.passage.id] ?? [];
    return response.sentences.flatMap((s) => s.highlights[o.passage.id] ?? []);
  };
  const started = busy === "ask" || response !== null || error !== null;

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-1">
        Skip to content
      </a>
      <Banner />
      <Header route={route} />
      <main id="main" className="mx-auto w-full max-w-4xl flex-1 px-4">
        {route === "sources" && <Sources />}
        {route === "about" && <About />}
        {route === "ask" && startup.state !== "ready" && (
          <StartingUp
            message={
              startup.state === "error" && startup.error
                ? `${STARTUP_MESSAGE.error} (${startup.error})`
                : STARTUP_MESSAGE[startup.state]
            }
          />
        )}
        {route === "ask" && !started && (
          <EmptyState
            question={question}
            setQuestion={setQuestion}
            onAsk={(q) => void submit(q)}
            busy={busy !== "" || startup.state !== "ready"}
            samples={sampleList}
          />
        )}
        {route === "ask" && started && (
          <div className="mx-auto max-w-2xl space-y-5 pb-32 pt-8">
            {asked && <h1 className="text-2xl font-semibold tracking-tight">{asked}</h1>}
            <div aria-live="polite" className="space-y-5">
              {busy === "ask" && (draft && !mayBeAbstention(draft) ? <DraftAnswer text={draft} /> : <AnswerSkeleton />)}
              {error && <ErrorCard message={error.message} onRetry={error.retry} />}
              {response?.abstained && <AbstentionCard />}
              {response && !response.abstained && (
                <>
                  <AnswerCard
                    response={response}
                    statuses={check?.sentences.map((s) => s.status)}
                    checking={busy === "verify"}
                    onCite={(p, i, el) => openPassage(p, i, el)}
                  />
                  <VerifyBar
                    result={check}
                    checking={busy === "verify"}
                    disabled={busy !== ""}
                    onCheck={() => void runCheck()}
                    waitS={waitS}
                  />
                  <SourcesList response={response} onOpen={(p, el) => openPassage(p, null, el)} />
                </>
              )}
            </div>
          </div>
        )}
      </main>
      {route === "ask" && started && (
        <div className="sticky bottom-0 z-10 border-t border-slate-200 bg-slate-50/90 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
          <div className="mx-auto max-w-2xl px-4">
            <AskBox
              value={question}
              onChange={setQuestion}
              onSubmit={(q) => void submit(q)}
              busy={busy !== "" || startup.state !== "ready"}
            />
          </div>
        </div>
      )}
      <Footer />
      {open && (
        <PassagePanel
          passage={open.passage}
          number={sources.find((s) => s.passage.id === open.passage.id)?.number}
          highlights={highlightsFor(open)}
          onClose={closePassage}
        />
      )}
    </div>
  );
}
