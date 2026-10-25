// A small parser for server-sent events read from a fetch stream (POST, so no EventSource).

export interface SSEvent {
  event: string;
  data: string;
}

/** Feed text chunks as they arrive; complete events (ending in a blank line) are emitted. */
export function createSSEParser(onEvent: (e: SSEvent) => void) {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent({ event, data: data.join("\n") });
    }
  };
}

const ABSTAIN_TEXT = "I cannot answer this from the provided documents.";
const squash = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** While a streamed draft could still be the abstention sentence, it is not shown, so an
 * abstention goes straight from the skeleton to the "not covered" card. */
export function mayBeAbstention(draft: string): boolean {
  return squash(ABSTAIN_TEXT).startsWith(squash(draft));
}
