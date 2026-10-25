import type { VerifyResponse } from "../api";
import { LEGEND_ORDER, STATUS, verifySummary } from "../lib/status";
import { CheckIcon, Spinner } from "./icons";
import { focusRing } from "./Layout";

interface Props {
  result: VerifyResponse | null;
  checking: boolean;
  disabled: boolean;
  onCheck: () => void;
  waitS?: number | null; // estimated rate-limit wait before the check starts
}

export function VerifyBar({ result, checking, disabled, onCheck, waitS = null }: Props) {
  return (
    <section aria-label="Check this answer" className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <button
          type="button"
          onClick={onCheck}
          disabled={disabled || checking}
          aria-busy={checking}
          className={`inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium transition-colors hover:border-teal-600 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 ${focusRing}`}
        >
          {checking ? <Spinner /> : <CheckIcon />}
          {checking ? "Checking…" : "Check this answer"}
        </button>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {checking && waitS !== null
            ? `Waiting about ${Math.ceil(waitS)} s for the rate limit, then checking…`
            : "can take up to a minute"}
        </span>
      </div>
      {result && (
        <div className="space-y-1 text-sm" aria-live="polite">
          {result.verified ? (
            <p className="font-medium">{verifySummary(result.sentences.map((s) => s.status))}</p>
          ) : (
            <p className="inline-block rounded-md bg-amber-100 px-2 py-0.5 text-amber-950 dark:bg-amber-900/50 dark:text-amber-100">
              Unverified: the check could not be completed
            </p>
          )}
          <ul aria-label="Legend" className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600 dark:text-slate-400">
            {LEGEND_ORDER.map((s) => (
              <li key={s} className="flex items-center gap-1.5">
                <span className={`h-2 w-2 rounded-full ${STATUS[s].dot}`} aria-hidden />
                {STATUS[s].label}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
