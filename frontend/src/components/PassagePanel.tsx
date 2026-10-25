import { useEffect, useRef } from "react";
import type { Passage } from "../api";
import { pageLabel } from "../lib/citations";
import { CloseIcon } from "./icons";
import { focusRing } from "./Layout";

interface Props {
  passage: Passage;
  number?: number;
  highlights: [number, number][];
  onClose: () => void;
}

function withHighlights(text: string, spans: [number, number][]) {
  const parts: (string | { mark: string })[] = [];
  let at = 0;
  for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0])) {
    if (start < at) continue;
    parts.push(text.slice(at, start), { mark: text.slice(start, end) });
    at = end;
  }
  parts.push(text.slice(at));
  return parts.map((p, i) =>
    typeof p === "string" ? (
      p
    ) : (
      <mark key={i} className="rounded bg-yellow-200 px-0.5 text-slate-900 dark:bg-yellow-300/80">
        {p.mark}
      </mark>
    ),
  );
}

export function PassagePanel({ passage, number, highlights, onClose }: Props) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const pages = pageLabel(passage).replace("p. ", "page ").replace("pp. ", "pages ");
  return (
    <>
      <div className="fixed inset-0 z-20 bg-slate-950/30 backdrop-blur-[1px]" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="passage-title"
        className="fixed inset-x-0 bottom-0 z-30 max-h-[80vh] overflow-y-auto rounded-t-2xl border border-slate-200 bg-white p-5 shadow-2xl dark:border-slate-800 dark:bg-slate-900 sm:inset-y-0 sm:left-auto sm:right-0 sm:max-h-none sm:w-[30rem] sm:rounded-none sm:rounded-l-2xl"
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            {number !== undefined && (
              <span className="text-xs font-semibold uppercase tracking-wide text-teal-800 dark:text-teal-300">
                Source {number}
              </span>
            )}
            <h2 id="passage-title" className="font-semibold leading-snug">
              {passage.title}
            </h2>
          </div>
          <button
            ref={close}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className={`rounded-lg border border-slate-300 p-1.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800 ${focusRing}`}
          >
            <CloseIcon />
          </button>
        </div>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          {passage.publisher} · as of {passage.as_of} · {pages}
          {passage.section && <> · {passage.section}</>}
        </p>
        {passage.superseded && (
          <p className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100">
            <strong>Superseded guidance ({passage.as_of}).</strong> {passage.superseded_note}
          </p>
        )}
        <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-slate-800 dark:text-slate-200">
          {withHighlights(passage.text, highlights)}
        </p>
      </div>
    </>
  );
}
