import { useState } from "react";
import type { Route } from "../lib/route";
import { CloseIcon, LogoIcon } from "./icons";

export const NOTICE =
  "MedQuery answers only from its source documents and does not diagnose or give dosing instructions. Consult a qualified health professional.";

const NAV: { route: Route; label: string; href: string }[] = [
  { route: "ask", label: "Ask", href: "#/" },
  { route: "sources", label: "Sources", href: "#/sources" },
  { route: "about", label: "About", href: "#/about" },
];

export const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 focus-visible:ring-offset-2 dark:focus-visible:ring-teal-300 dark:focus-visible:ring-offset-slate-950";

function readDismissed(): boolean {
  try {
    return sessionStorage.getItem("medquery-banner") === "dismissed";
  } catch {
    return false;
  }
}

export function Banner() {
  const [hidden, setHidden] = useState(readDismissed);
  if (hidden) return null;
  return (
    <div role="note" className="border-b border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-100">
      <div className="mx-auto flex max-w-4xl items-center gap-3 px-4 py-1.5 text-sm">
        <p className="flex-1">
          <strong>Not medical advice.</strong> Answers come only from public health documents.
        </p>
        <button
          type="button"
          aria-label="Dismiss the notice for this session"
          onClick={() => {
            try {
              sessionStorage.setItem("medquery-banner", "dismissed");
            } catch {
              /* private mode: just hide it */
            }
            setHidden(true);
          }}
          className={`rounded p-1 hover:bg-amber-100 dark:hover:bg-amber-900 ${focusRing}`}
        >
          <CloseIcon />
        </button>
      </div>
    </div>
  );
}

export function Header({ route }: { route: Route }) {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <a href="#/" className={`flex items-center gap-2 rounded ${focusRing}`}>
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-700 text-white dark:bg-teal-400 dark:text-slate-950">
            <LogoIcon className="h-5 w-5" />
          </span>
          <span>
            <span className="block text-lg font-semibold tracking-tight">MedQuery</span>
            <span className="block text-xs text-slate-600 dark:text-slate-400">
              Answers from public health guidelines, with sources
            </span>
          </span>
        </a>
        <nav aria-label="Main">
          <ul className="flex gap-1 text-sm">
            {NAV.map((n) => (
              <li key={n.route}>
                <a
                  href={n.href}
                  aria-current={route === n.route ? "page" : undefined}
                  className={`rounded-md px-3 py-1.5 transition-colors ${focusRing} ${
                    route === n.route
                      ? "bg-teal-50 font-medium text-teal-800 dark:bg-teal-950 dark:text-teal-200"
                      : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                  }`}
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mt-16 border-t border-slate-200 py-6 text-sm text-slate-600 dark:border-slate-800 dark:text-slate-400">
      <div className="mx-auto max-w-4xl space-y-1 px-4">
        <p>
          <strong className="text-slate-800 dark:text-slate-200">Not medical advice.</strong> {NOTICE}
        </p>
        <p>
          Built on 10 public-domain CDC/NIH/VA documents ·{" "}
          <a href="#/sources" className={`rounded text-teal-800 underline dark:text-teal-300 ${focusRing}`}>
            Sources
          </a>
        </p>
      </div>
    </footer>
  );
}
