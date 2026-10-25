import type { AskResponse, Passage, VerifyStatus } from "../api";

// Citation markers like [tb-ltbi-2020:0020] are shown as chips, so they are removed from the text.
const CITATION_RE = /\s*[[【][^\]】]*:\d{4}[^\]】]*[\]】]/g;

export const STATUS_STYLE: Record<VerifyStatus, { label: string; className: string }> = {
  supported: { label: "Supported", className: "border-green-700 bg-green-50" },
  unverified: { label: "Unverified", className: "border-amber-600 bg-amber-50" },
  unsupported: { label: "Unsupported", className: "border-red-700 bg-red-50 line-through" },
  note: { label: "Note", className: "border-slate-400 bg-slate-100 text-slate-700" },
};

interface Props {
  response: AskResponse;
  statuses?: VerifyStatus[];
  onCite: (passage: Passage, sentenceIndex: number) => void;
}

export function Answer({ response, statuses, onCite }: Props) {
  if (response.abstained) {
    return (
      <section aria-label="Answer" className="rounded-lg border border-slate-300 bg-white p-4">
        <h2 className="font-semibold">The documents don't cover this</h2>
        <p className="mt-1 text-slate-700">
          MedQuery only answers from its ten source documents, and none of the passages it found
          answers this question. Try rephrasing, or see the Sources page for what is covered.
        </p>
      </section>
    );
  }
  const byId = new Map(response.passages.map((p) => [p.id, p]));
  return (
    <section aria-label="Answer" className="space-y-2 rounded-lg border border-slate-300 bg-white p-4">
      {response.sentences.map((sentence, i) => {
        const status = statuses?.[i];
        const style = status ? STATUS_STYLE[status] : undefined;
        return (
          <p
            key={i}
            data-status={status ?? "none"}
            className={`leading-relaxed ${style ? `border-l-4 pl-2 ${style.className}` : ""}`}
          >
            {style && <span className="sr-only">{style.label}: </span>}
            <span className={status === "unsupported" ? "line-through" : ""}>
              {sentence.text.replace(CITATION_RE, "")}
            </span>
            {sentence.citations.map((id) => {
              const passage = byId.get(id);
              if (!passage) return null;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => onCite(passage, i)}
                  className="ml-1 inline-flex items-center gap-1 rounded-full border border-blue-700 px-2 py-0.5 text-xs text-blue-800 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-600"
                  aria-label={`Source: ${passage.title}, page ${passage.page_start}${passage.superseded ? ", superseded" : ""}`}
                >
                  {passage.title.split(/[,:]/)[0]} p.{passage.page_start}
                  {passage.superseded && (
                    <span className="rounded bg-amber-200 px-1 text-amber-900">superseded</span>
                  )}
                </button>
              );
            })}
            {style && (
              <span className="ml-2 text-xs font-medium uppercase tracking-wide text-slate-600">
                {style.label}
              </span>
            )}
          </p>
        );
      })}
    </section>
  );
}
