import type { VerifyStatus } from "../api";

export const STATUS: Record<VerifyStatus, { label: string; dot: string }> = {
  supported: { label: "Supported", dot: "bg-green-600" },
  unsupported: { label: "Unsupported", dot: "bg-red-600" },
  unverified: { label: "Unverified", dot: "bg-amber-500" },
  note: { label: "Note", dot: "bg-slate-400" },
};

export const LEGEND_ORDER: VerifyStatus[] = ["supported", "unsupported", "unverified", "note"];

/** "3 of 4 sentences supported", with notes counted separately ("· 1 note"). */
export function verifySummary(statuses: VerifyStatus[]): string {
  const notes = statuses.filter((s) => s === "note").length;
  const facts = statuses.length - notes;
  const supported = statuses.filter((s) => s === "supported").length;
  const noun = facts === 1 ? "sentence" : "sentences";
  let text = `${supported} of ${facts} ${noun} supported`;
  if (notes) text += ` · ${notes} ${notes === 1 ? "note" : "notes"}`;
  return text;
}
