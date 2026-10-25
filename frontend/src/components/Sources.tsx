import { useEffect, useState } from "react";
import { documents, type SourceDocument } from "../api";
import { friendlyError } from "../lib/errors";
import { ErrorCard } from "./AnswerCard";
import { focusRing } from "./Layout";

export function Sources() {
  const [docs, setDocs] = useState<SourceDocument[]>([]);
  const [error, setError] = useState("");
  const load = () => {
    setError("");
    documents()
      .then((r) => setDocs(r.documents))
      .catch((e) => setError(friendlyError(e)));
  };
  useEffect(load, []);
  return (
    <section aria-labelledby="sources-title" className="mx-auto max-w-2xl space-y-4 pt-8">
      <h1 id="sources-title" className="text-3xl font-semibold tracking-tight">
        Sources
      </h1>
      <p className="text-slate-600 dark:text-slate-400">
        MedQuery answers only from these {docs.length || 10} public documents. It does not use the
        web or general knowledge.
      </p>
      {error && <ErrorCard message={error} onRetry={load} />}
      <ul className="space-y-3">
        {docs.map((d) => (
          <li key={d.id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <a href={d.url} target="_blank" rel="noreferrer" className={`rounded font-medium text-teal-800 underline dark:text-teal-300 ${focusRing}`}>
                {d.title}
              </a>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs capitalize text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                {d.topic}
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              {d.publisher} · {d.year} · licence: {d.licence}
            </p>
            {d.superseded && (
              <p className="mt-2 text-sm text-amber-900 dark:text-amber-200">
                <span className="mr-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium dark:bg-amber-900/50">Superseded</span>
                {d.superseded}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
