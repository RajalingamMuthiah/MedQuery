import { useEffect, useState } from "react";
import { documents, type SourceDocument } from "../api";

export function Sources() {
  const [docs, setDocs] = useState<SourceDocument[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    documents()
      .then((r) => setDocs(r.documents))
      .catch((e: Error) => setError(e.message));
  }, []);
  return (
    <section aria-labelledby="sources-title" className="space-y-3">
      <h2 id="sources-title" className="text-lg font-semibold">
        Sources
      </h2>
      <p className="text-slate-700">
        MedQuery answers only from these {docs.length || 10} public documents. It does not use
        the web or general knowledge.
      </p>
      {error && <p role="alert" className="text-red-800">{error}</p>}
      <ul className="space-y-2">
        {docs.map((d) => (
          <li key={d.id} className="rounded-lg border border-slate-300 bg-white p-3">
            <a href={d.url} className="font-medium text-blue-800 underline" target="_blank" rel="noreferrer">
              {d.title}
            </a>
            <p className="text-sm text-slate-700">
              {d.publisher} · {d.year} · {d.topic} · licence: {d.licence}
            </p>
            {d.superseded && (
              <p className="mt-1 text-sm text-amber-900">
                <span className="rounded bg-amber-200 px-1">superseded</span> {d.superseded}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
