"""The baseline answer prompt (answer_v1). Changing the text changes its hash in every report."""

import hashlib
import re

from medquery.retrieval.dense import Hit

ANSWER_VERSION = "answer_v1"
ABSTAIN_TEXT = "I cannot answer this from the provided documents."

ANSWER_SYSTEM = f"""You answer questions about health topics using ONLY the context passages below.
You are not a doctor and you do not give medical advice.
Rules:
1. Use only facts stated in the passages. Do not use outside knowledge.
2. After each sentence that states a fact, cite the passage id(s) that support it in
   square brackets, for example [tb-ltbi-2020:0020]. Cite only ids shown below.
3. If the passages do not contain the answer, reply exactly:
   {ABSTAIN_TEXT}
4. Never give a diagnosis. State doses only exactly as a passage states them, with its citation.
5. If you use a passage marked SUPERSEDED, say that this guidance is from its "as of" year
   and has been superseded.
Answer in at most 5 sentences."""

USER_TEMPLATE = "Context passages:\n{passages}\n\nQuestion: {question}"

_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*:\d{4}")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def passage_header(chunk: dict) -> str:
    pages = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    parts = [f"[{chunk['id']}] {chunk['title']}", chunk["publisher"], f"as of {chunk['as_of']}"]
    if chunk.get("superseded"):
        parts.append("SUPERSEDED")
    parts.append(pages)
    if chunk.get("section"):
        parts.append(chunk["section"])
    return " | ".join(parts)


def render_passages(hits: list[Hit]) -> str:
    return "\n---\n".join(f"{passage_header(h.chunk)}\n{h.chunk['text']}" for h in hits)


def answer_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = USER_TEMPLATE.format(passages=render_passages(hits), question=question)
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n\x1e\n".join(parts).encode("utf-8")).hexdigest()[:12]


ANSWER_HASH = prompt_hash(ANSWER_SYSTEM, USER_TEMPLATE)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def is_abstention(text: str) -> bool:
    """True if the answer is the abstention sentence (ignoring case, spacing and punctuation)."""
    return _normalise(text).startswith(_normalise(ABSTAIN_TEXT))


def extract_citations(text: str) -> list[str]:
    """Chunk ids cited in square brackets, in order of first appearance, without duplicates."""
    cited: list[str] = []
    for group in _BRACKET_RE.findall(text):
        for chunk_id in _ID_RE.findall(group):
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited
