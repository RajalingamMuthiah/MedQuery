import { focusRing } from "./Layout";

const REPORT_URL =
  "https://github.com/RajalingamMuthiah/MedQuery/blob/main/medquery/eval/reports/2026-10-25_summary_test_dacaeb434340_vs_ff63d760267f.md";

const STEPS = [
  ["Hybrid search", "Your question is matched against ten public health documents by meaning and by keywords, and the best passages are fused into one ranking."],
  ["Grounded answer", "A language model writes a short answer from the top five passages only. If they don't contain the answer, it says so instead of guessing."],
  ["Numbered citations", "Every sentence cites the passage it came from. Click a number to read the passage itself."],
  ["Optional check", "“Check this answer” asks a second pass to test each sentence's claims against the passages and marks them supported, unsupported or unverified."],
];

// Headline numbers from the README's "Final results (test split)" (summary comparison of the
// baseline and the live config on the 23-question test split).
const RESULTS = [
  ["56% → 83%", "answer correctness, simple baseline → this app"],
  ["39% → 11%", "false refusals on answerable questions"],
  ["100%", "of unanswerable test questions declined"],
];

export function About() {
  return (
    <article className="mx-auto max-w-2xl space-y-10 pt-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">About MedQuery</h1>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          A research demo of question answering over public health guidelines, built to show its
          sources and to say when it doesn't know.
        </p>
      </header>

      <section aria-labelledby="how-title">
        <h2 id="how-title" className="text-lg font-semibold">How it works</h2>
        <ol className="mt-4 space-y-4">
          {STEPS.map(([title, text], i) => (
            <li key={title} className="flex gap-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-700 text-sm font-semibold text-white dark:bg-teal-400 dark:text-slate-950">
                {i + 1}
              </span>
              <div>
                <h3 className="font-medium">{title}</h3>
                <p className="text-slate-600 dark:text-slate-400">{text}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="results-title">
        <h2 id="results-title" className="text-lg font-semibold">Results on held-out test questions</h2>
        <dl className="mt-4 grid gap-3 sm:grid-cols-3">
          {RESULTS.map(([value, label]) => (
            <div key={label} className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
              <dt className="text-sm text-slate-600 dark:text-slate-400">{label}</dt>
              <dd className="mt-1 text-2xl font-semibold text-teal-800 dark:text-teal-300">{value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
          Measured once on 23 held-out questions after all design choices were made on separate
          development questions.{" "}
          <a href={REPORT_URL} className={`rounded text-teal-800 underline dark:text-teal-300 ${focusRing}`}>
            Read the report
          </a>
          .
        </p>
      </section>

      <section aria-labelledby="limits-title">
        <h2 id="limits-title" className="text-lg font-semibold">Limitations</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-700 dark:text-slate-300">
          <li>The test set is small (23 questions), so the numbers are uncertain.</li>
          <li>It was not shown to make fewer unsupported claims than the simple baseline; the baseline refused far more often.</li>
          <li>The test questions were written by an AI and checked by an AI, not by a clinician.</li>
          <li>It knows only ten US documents on three topics, and some are dated: the JNC 7 hypertension guidance (2003) has been superseded and is flagged as such.</li>
          <li>It is not medical advice and does not diagnose or recommend doses.</li>
        </ul>
      </section>
    </article>
  );
}
