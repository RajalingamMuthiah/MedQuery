import { useEffect, useRef } from "react";
import type { Passage } from "../api";

interface Props {
  passage: Passage;
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
      <mark key={i} className="rounded bg-yellow-200 px-0.5">
        {p.mark}
      </mark>
    ),
  );
}

export function PassagePanel({ passage, highlights, onClose }: Props) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const pages =
    passage.page_start === passage.page_end
      ? `page ${passage.page_start}`
      : `pages ${passage.page_start}-${passage.page_end}`;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="passage-title"
      className="fixed inset-x-0 bottom-0 z-20 max-h-[75vh] overflow-y-auto rounded-t-xl border border-slate-300 bg-white p-4 shadow-xl sm:inset-y-0 sm:left-auto sm:right-0 sm:max-h-none sm:w-[28rem] sm:rounded-none"
    >
      <div className="flex items-start justify-between gap-2">
        <h2 id="passage-title" className="font-semibold">
          {passage.title}
        </h2>
        <button
          ref={close}
          type="button"
          onClick={onClose}
          className="rounded border border-slate-400 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600"
        >
          Close
        </button>
      </div>
      <p className="mt-1 text-sm text-slate-700">
        {passage.publisher} · as of {passage.as_of} · {pages}
        {passage.section && <> · {passage.section}</>}
      </p>
      {passage.superseded && (
        <p className="mt-2 rounded border border-amber-600 bg-amber-50 p-2 text-sm text-amber-900">
          <strong>Superseded guidance ({passage.as_of}).</strong> {passage.superseded_note}
        </p>
      )}
      <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">
        {withHighlights(passage.text, highlights)}
      </p>
    </div>
  );
}
