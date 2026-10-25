import { useCallback, useEffect, useRef, useState } from "react";
import {
  ask,
  samples as loadSamples,
  verify,
  type AskResponse,
  type Passage,
  type Sample,
  type VerifyResponse,
} from "./api";
import { About } from "./components/About";
import { AbstentionCard, AnswerCard, AnswerSkeleton, ErrorCard, SourcesList } from "./components/AnswerCard";
import { AskBox, EmptyState } from "./components/Ask";
import { Banner, Footer, Header } from "./components/Layout";
import { PassagePanel } from "./components/PassagePanel";
import { Sources } from "./components/Sources";
import { VerifyBar } from "./components/VerifyBar";
import { numberCitations } from "./lib/citations";
import { friendlyError } from "./lib/errors";
import { useRoute } from "./lib/route";

interface OpenPassage {
  passage: Passage;
  sentence: number | null;
}

export default function App() {
  const route = useRoute();
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
    try {
      setResponse(await ask(q));
    } catch (e) {
      setError({ message: friendlyError(e), retry: () => void submit(q) });
    } finally {
      setBusy("");
    }
  }, []);

  async function runCheck() {
    if (!response) return;
    setBusy("verify");
    setError(null);
    try {
      setCheck(await verify(response.question, response.answer, response.passages.map((p) => p.id)));
    } catch (e) {
      setError({ message: friendlyError(e), retry: () => void runCheck() });
    } finally {
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
        {route === "ask" && !started && (
          <EmptyState question={question} setQuestion={setQuestion} onAsk={(q) => void submit(q)} busy={busy !== ""} samples={sampleList} />
        )}
        {route === "ask" && started && (
          <div className="mx-auto max-w-2xl space-y-5 pb-32 pt-8">
            {asked && <h1 className="text-2xl font-semibold tracking-tight">{asked}</h1>}
            <div aria-live="polite" className="space-y-5">
              {busy === "ask" && <AnswerSkeleton />}
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
                  <VerifyBar result={check} checking={busy === "verify"} disabled={busy !== ""} onCheck={() => void runCheck()} />
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
            <AskBox value={question} onChange={setQuestion} onSubmit={(q) => void submit(q)} busy={busy !== ""} />
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
