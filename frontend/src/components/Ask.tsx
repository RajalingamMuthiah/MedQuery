import { useState } from "react";
import type { Sample } from "../api";
import { SearchIcon, Spinner } from "./icons";
import { focusRing } from "./Layout";

export const MAX_CHARS = 500;

interface AskBoxProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => void;
  busy: boolean;
  large?: boolean;
}

export function AskBox({ value, onChange, onSubmit, busy, large = false }: AskBoxProps) {
  return (
    <form
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onSubmit(value);
      }}
      className={`flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition-shadow focus-within:border-teal-600 focus-within:shadow-md dark:border-slate-700 dark:bg-slate-900 ${large ? "p-3" : ""}`}
    >
      <label htmlFor="question" className="sr-only">
        Ask about diabetes, hypertension or tuberculosis
      </label>
      <textarea
        id="question"
        value={value}
        maxLength={MAX_CHARS}
        rows={large ? 3 : 1}
        placeholder="Ask about diabetes, hypertension or tuberculosis…"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (value.trim() && !busy) onSubmit(value);
          }
        }}
        className={`min-w-0 flex-1 resize-none bg-transparent px-2 py-1.5 outline-none placeholder:text-slate-500 dark:placeholder:text-slate-400 ${large ? "text-lg" : ""}`}
      />
      <div className="flex flex-col items-end gap-1">
        {large && (
          <span className="text-xs text-slate-500 dark:text-slate-400" aria-live="polite">
            {value.length}/{MAX_CHARS}
          </span>
        )}
        <button
          type="submit"
          disabled={busy || !value.trim()}
          aria-label="Ask"
          className={`flex h-10 w-10 items-center justify-center rounded-xl bg-teal-700 text-white transition-colors hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-teal-400 dark:text-slate-950 dark:hover:bg-teal-300 ${focusRing}`}
        >
          {busy ? <Spinner /> : <SearchIcon />}
        </button>
      </div>
    </form>
  );
}

const TOPICS = [
  { key: "diabetes", label: "Diabetes" },
  { key: "hypertension", label: "Hypertension" },
  { key: "tuberculosis", label: "Tuberculosis" },
];

interface EmptyStateProps {
  question: string;
  setQuestion: (q: string) => void;
  onAsk: (q: string) => void;
  busy: boolean;
  samples: Sample[];
}

export function EmptyState({ question, setQuestion, onAsk, busy, samples }: EmptyStateProps) {
  const [topic, setTopic] = useState("");
  const shown = topic ? samples.filter((s) => s.topic === topic) : samples;
  return (
    <section className="mx-auto max-w-2xl pt-10 sm:pt-16" aria-labelledby="home-title">
      <h1 id="home-title" className="text-center text-3xl font-semibold tracking-tight sm:text-4xl">
        What would you like to know?
      </h1>
      <p className="mt-2 text-center text-slate-600 dark:text-slate-400">
        Answers drawn only from public health guidelines, each claim linked to its source.
      </p>
      <div className="mt-8">
        <AskBox value={question} onChange={setQuestion} onSubmit={onAsk} busy={busy} large />
      </div>
      <div className="mt-5 flex flex-wrap justify-center gap-2" role="group" aria-label="Filter sample questions by topic">
        {TOPICS.map((t) => (
          <button
            key={t.key}
            type="button"
            aria-pressed={topic === t.key}
            onClick={() => setTopic(topic === t.key ? "" : t.key)}
            className={`rounded-full border px-3 py-1 text-sm transition-colors ${focusRing} ${
              topic === t.key
                ? "border-teal-700 bg-teal-700 text-white dark:border-teal-300 dark:bg-teal-300 dark:text-slate-950"
                : "border-slate-300 text-slate-700 hover:border-teal-600 dark:border-slate-700 dark:text-slate-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {shown.length > 0 && (
        <section aria-labelledby="samples-title" className="mt-8">
          <h2 id="samples-title" className="text-sm font-medium text-slate-600 dark:text-slate-400">
            Try a sample question
          </h2>
          <ul className="mt-3 grid gap-3 sm:grid-cols-2">
            {shown.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => {
                    setQuestion(s.question);
                    onAsk(s.question);
                  }}
                  className={`h-full w-full rounded-xl border border-slate-200 bg-white p-4 text-left text-sm shadow-sm transition hover:-translate-y-0.5 hover:border-teal-600 hover:shadow motion-reduce:transform-none dark:border-slate-800 dark:bg-slate-900 ${focusRing}`}
                >
                  <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-teal-800 dark:text-teal-300">
                    {s.topic}
                  </span>
                  {s.question}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </section>
  );
}
