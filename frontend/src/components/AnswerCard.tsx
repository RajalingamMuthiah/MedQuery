import type { AskResponse, Passage, VerifyStatus } from "../api";
import { numberCitations, pageLabel, stripCitations } from "../lib/citations";
import { STATUS } from "../lib/status";
import { AlertIcon, BookIcon } from "./icons";
import { focusRing } from "./Layout";

interface AnswerCardProps {
  response: AskResponse;
  statuses?: VerifyStatus[];
  checking?: boolean;
  onCite: (passage: Passage, sentenceIndex: number, trigger: HTMLElement) => void;
}

export function AnswerCard({ response, statuses, checking = false, onCite }: AnswerCardProps) {
  const sources = numberCitations(response.sentences, response.passages);
  const numberOf = new Map(sources.map((s) => [s.passage.id, s.number]));
  return (
    <article
      aria-label="Answer"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 sm:p-6"
    >
      <div className="space-y-3 text-[1.0625rem] leading-relaxed">
        {response.sentences.map((sentence, i) => {
          const status = statuses?.[i];
          return (
            <p
              key={i}
              data-status={status ?? "none"}
              className={`group flex gap-2 ${checking ? "animate-pulse motion-reduce:animate-none" : ""}`}
            >
              {status && (
                <span className="relative mt-2.5 flex shrink-0" tabIndex={0} aria-label={STATUS[status].label}>
                  <span className={`h-2.5 w-2.5 rounded-full ${STATUS[status].dot}`} aria-hidden />
                  <span className="pointer-events-none absolute left-4 top-1/2 z-10 -translate-y-1/2 whitespace-nowrap rounded bg-slate-900 px-2 py-0.5 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 dark:bg-slate-100 dark:text-slate-900">
                    {STATUS[status].label}
                  </span>
                </span>
              )}
              <span className={status === "unsupported" ? "text-slate-500 line-through decoration-red-600/70 dark:text-slate-400" : ""}>
                {status && <span className="sr-only">{STATUS[status].label}: </span>}
                {stripCitations(sentence.text)}
                {sentence.citations
                  .filter((id) => numberOf.has(id))
                  .map((id) => {
                    const source = sources.find((s) => s.passage.id === id)!;
                    return (
                      <sup key={id} className="ml-0.5">
                        <button
                          type="button"
                          onClick={(e) => onCite(source.passage, i, e.currentTarget)}
                          aria-label={`Source ${source.number}: ${source.passage.title}, ${pageLabel(source.passage)}`}
                          className={`rounded px-0.5 text-xs font-semibold text-teal-800 hover:bg-teal-50 dark:text-teal-300 dark:hover:bg-teal-950 ${focusRing}`}
                        >
                          [{source.number}]
                        </button>
                      </sup>
                    );
                  })}
              </span>
            </p>
          );
        })}
      </div>
      <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
        Retrieved in {Math.round(response.latency.retrieval_ms)} ms · generated in{" "}
        {response.latency.generation_s.toFixed(1)} s
      </p>
    </article>
  );
}

interface SourcesListProps {
  response: AskResponse;
  onOpen: (passage: Passage, trigger: HTMLElement) => void;
}

export function SourcesList({ response, onOpen }: SourcesListProps) {
  const sources = numberCitations(response.sentences, response.passages);
  if (!sources.length) return null;
  return (
    <section aria-labelledby="sources-heading">
      <h2 id="sources-heading" className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-300">
        Sources
      </h2>
      <ol className="space-y-2">
        {sources.map(({ number, passage }) => (
          <li key={passage.id}>
            <button
              type="button"
              onClick={(e) => onOpen(passage, e.currentTarget)}
              className={`flex w-full items-start gap-3 rounded-xl border border-slate-200 bg-white p-3 text-left text-sm transition-colors hover:border-teal-600 dark:border-slate-800 dark:bg-slate-900 ${focusRing}`}
            >
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-teal-50 text-xs font-semibold text-teal-800 dark:bg-teal-950 dark:text-teal-200">
                {number}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-medium">{passage.title}</span>
                <span className="block text-slate-600 dark:text-slate-400">
                  {passage.publisher} · {passage.as_of} · {pageLabel(passage)}
                </span>
              </span>
              {passage.superseded && (
                <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-900 dark:bg-amber-900/50 dark:text-amber-100">
                  Superseded
                </span>
              )}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function AbstentionCard() {
  return (
    <article
      aria-label="Answer"
      className="flex gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 sm:p-6"
    >
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-teal-50 text-teal-800 dark:bg-teal-950 dark:text-teal-200">
        <BookIcon className="h-5 w-5" />
      </span>
      <div>
        <h2 className="font-semibold">Not covered by the documents</h2>
        <p className="mt-1 text-slate-600 dark:text-slate-400">
          MedQuery answers only from its ten source documents, and none of them answers this
          question.{" "}
          <a href="#/sources" className={`rounded text-teal-800 underline dark:text-teal-300 ${focusRing}`}>
            See what the sources cover
          </a>
          .
        </p>
      </div>
    </article>
  );
}

export function AnswerSkeleton() {
  return (
    <div
      role="status"
      aria-label="Finding an answer"
      className="space-y-3 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      {["w-11/12", "w-full", "w-9/12", "w-10/12"].map((w) => (
        <div key={w} className={`h-4 ${w} animate-pulse rounded bg-slate-200 motion-reduce:animate-none dark:bg-slate-800`} />
      ))}
      <span className="sr-only">Finding an answer…</span>
    </div>
  );
}

export function ErrorCard({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-100"
    >
      <AlertIcon className="mt-0.5 h-5 w-5 shrink-0" />
      <p className="flex-1">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className={`rounded-lg border border-red-300 bg-white px-3 py-1 text-sm font-medium hover:bg-red-100 dark:border-red-800 dark:bg-transparent dark:hover:bg-red-900 ${focusRing}`}
        >
          Retry
        </button>
      )}
    </div>
  );
}
