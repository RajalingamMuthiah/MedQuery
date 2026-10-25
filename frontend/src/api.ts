// Types and calls for the MedQuery API (medquery/api/app.py).
import { createSSEParser } from "./lib/sse";

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
  latency: { retrieval_ms: number; generation_s: number; ttft_s?: number | null };
  model: string;
  cached?: boolean;
  notice: string;
}

export interface Health {
  status: "ok" | "starting" | "error";
  ready: boolean;
  error: string | null;
  daily_remaining: number;
}

export type VerifyStatus = "supported" | "unsupported" | "note" | "unverified";

export interface VerifyResponse {
  sentences: { text: string; status: VerifyStatus }[];
  verified: boolean;
  verifier: string;
  latency_s: number;
  wait_s?: number;
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

/** An error response from the API, with its HTTP status (network failures are plain Errors). */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(response.status, body.error ?? `Request failed (${response.status})`);
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

export const health = () => call<Health>("/health");

export const verifyEstimate = (question: string, answer: string, passage_ids: string[]) =>
  call<{ wait_s: number }>("/verify/estimate", {
    method: "POST",
    body: JSON.stringify({ question, answer, passage_ids }),
  });

export interface StreamHandlers {
  onDelta: (text: string) => void;
}

/** POST /ask/stream: calls onDelta for each piece of answer text and resolves with the final
 * response (the same JSON as /ask). HTTP errors before the stream starts throw ApiError. */
export async function askStream(question: string, handlers: StreamHandlers): Promise<AskResponse> {
  const response = await fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.error ?? `Request failed (${response.status})`);
  }
  const result: { final: AskResponse | null; failure: ApiError | null } = {
    final: null,
    failure: null,
  };
  const feed = createSSEParser(({ event, data }) => {
    const payload = JSON.parse(data);
    if (event === "delta") handlers.onDelta(payload.text);
    else if (event === "final") result.final = payload as AskResponse;
    else if (event === "error") result.failure = new ApiError(payload.status ?? 503, payload.error);
  });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    feed(value);
  }
  if (result.failure) throw result.failure;
  if (!result.final) throw new Error("the answer stream ended early");
  return result.final;
}

export const samples = () => call<{ samples: Sample[] }>("/samples");

export const documents = () => call<{ documents: SourceDocument[] }>("/documents");
