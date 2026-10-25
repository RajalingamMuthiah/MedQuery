import type { Passage, Sentence } from "../api";

// Citation markers like [tb-ltbi-2020:0020] become numbered superscripts, so they are removed
// from the sentence text.
const CITATION_RE = /\s*[[【][^\]】]*:\d{4}[^\]】]*[\]】]/g;

export function stripCitations(text: string): string {
  return text.replace(CITATION_RE, "");
}

export interface NumberedSource {
  number: number;
  passage: Passage;
}

/** Number each cited passage by its first appearance in the answer (1, 2, ...). Citations to
 * passages that were not retrieved are ignored. */
export function numberCitations(sentences: Sentence[], passages: Passage[]): NumberedSource[] {
  const byId = new Map(passages.map((p) => [p.id, p]));
  const sources: NumberedSource[] = [];
  for (const sentence of sentences) {
    for (const id of sentence.citations) {
      const passage = byId.get(id);
      if (passage && !sources.some((s) => s.passage.id === id)) {
        sources.push({ number: sources.length + 1, passage });
      }
    }
  }
  return sources;
}

export function pageLabel(p: Passage): string {
  return p.page_start === p.page_end ? `p. ${p.page_start}` : `pp. ${p.page_start}–${p.page_end}`;
}
