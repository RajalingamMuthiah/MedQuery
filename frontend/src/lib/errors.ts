import { ApiError } from "../api";

/** A message a person can act on, for any error from the API or the network. */
export function friendlyError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) {
      return /daily/i.test(error.message)
        ? "Daily demo limit reached — try again tomorrow."
        : "Too many requests — please wait a minute and try again.";
    }
    if (error.status === 422) return error.message;
    if (error.status === 503) return "The language model is unavailable right now — please try again shortly.";
    return `Something went wrong (${error.status}). Please try again.`;
  }
  return "Can't reach the MedQuery server — check it is running, then try again.";
}
