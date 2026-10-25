// Types and calls for the MedQuery API (medquery/api/app.py).

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface Passage {
  id: string;
  doc_id: string;
  title: string;
  publisher: string;
  as_of: string;
  superseded: boolean;
  superseded_note: string;
  page_start: number;
  page_end: number;
  section: string;
  kind: string;
  text: string;
}

export interface Sentence {
  text: string;
  citations: string[];
  highlights: Record<string, [number, number][]>;
}

export interface AskResponse {
  question: string;
  answer: string;
  abstained: boolean;
  sentences: Sentence[];
  passages: Passage[];
  latency: { retrieval_ms: number; generation_s: number };
  model: string;
  notice: string;
}

export type VerifyStatus = "supported" | "unsupported" | "note" | "unverified";

export interface VerifyResponse {
  sentences: { text: string; status: VerifyStatus }[];
  verified: boolean;
  verifier: string;
  latency_s: number;
}

export interface Sample {
  id: string;
  question: string;
  topic: string;
}

export interface SourceDocument {
  id: string;
  title: string;
  publisher: string;
  year: number;
  topic: string;
  as_of: string;
  licence: string;
  superseded: string;
  url: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export const ask = (question: string) =>
  call<AskResponse>("/ask", { method: "POST", body: JSON.stringify({ question }) });

export const verify = (question: string, answer: string, passage_ids: string[]) =>
  call<VerifyResponse>("/verify", {
    method: "POST",
    body: JSON.stringify({ question, answer, passage_ids }),
  });

export const samples = () => call<{ samples: Sample[] }>("/samples");

export const documents = () => call<{ documents: SourceDocument[] }>("/documents");
