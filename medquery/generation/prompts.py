"""Answer prompts (answer_v1, answer_v2), selected by version. Changing a prompt's text changes
its hash in every report; earlier versions are kept unchanged for comparison."""

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

# answer_v2 (Phase 4): the diagnosis found added detail not in the passages, a unit misread from
# a table, a citation to the wrong passage and a missing superseded warning.
ANSWER_V2_SYSTEM = f"""\
You answer questions about health topics using ONLY the context passages below.
You are not a doctor and you do not give medical advice.
Rules:
1. Use only facts stated in the passages. Add no detail, example or list item that is not in a
   passage, even if you believe it is true.
2. End every sentence with the id(s) of the passage(s) it came from, in square brackets, written
   in plain ASCII, for example [tb-ltbi-2020:0020]. Cite only ids shown below, and cite the
   passage that actually states the fact.
3. Copy numbers, units and qualifiers exactly as the passage writes them (for example "per 1,000
   persons", "in millions", "at least", "or"). Before using a value from a table, check the
   table's caption, the column header and its unit.
4. If the passages answer only part of the question, answer that part and say plainly which part
   the passages do not cover. Do not guess.
5. If the passages do not contain the answer at all, reply exactly:
   {ABSTAIN_TEXT}
6. Never give a diagnosis. State doses only exactly as a passage states them, with its citation.
7. If any passage you use is marked SUPERSEDED, end with a sentence saying that this guidance is
   from its "as of" year and has been superseded.
Answer in at most 5 sentences."""

_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*:\d{4}")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
# The generator sometimes writes ids with Unicode hyphens (U+2010-2013, U+2212) or in fullwidth
# brackets; they are normalised before parsing so a valid citation is not lost.
_CITATION_FIXES = str.maketrans(
    {c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"} | {"\u3010": "[", "\u3011": "]"}
)


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


def answer_messages(question: str, hits: list[Hit], version: str = ANSWER_VERSION) -> list[dict]:
    system, template = PROMPTS[version]
    user = template.format(passages=render_passages(hits), question=question)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n\x1e\n".join(parts).encode("utf-8")).hexdigest()[:12]


ANSWER_HASH = prompt_hash(ANSWER_SYSTEM, USER_TEMPLATE)
PROMPTS = {  # version -> (system, user template)
    "answer_v1": (ANSWER_SYSTEM, USER_TEMPLATE),
    "answer_v2": (ANSWER_V2_SYSTEM, USER_TEMPLATE),
}


def prompt_id(version: str) -> str:
    """'<version>:<hash>' as recorded in RunConfig.answer_prompt and every report."""
    return f"{version}:{prompt_hash(*PROMPTS[version])}"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def is_abstention(text: str) -> bool:
    """True if the answer is the abstention sentence (ignoring case, spacing and punctuation)."""
    return _normalise(text).startswith(_normalise(ABSTAIN_TEXT))


def extract_citations(text: str) -> list[str]:
    """Chunk ids cited in square brackets, in order of first appearance, without duplicates."""
    cited: list[str] = []
    for group in _BRACKET_RE.findall(text.translate(_CITATION_FIXES)):
        for chunk_id in _ID_RE.findall(group):
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited
