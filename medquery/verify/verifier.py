"""Claim-level verification of a generated answer (prompts verify_v1, verify_v2).

One call per answered question to the generator model (never the judge model, so evaluation
stays independent): each sentence is classified as a fact or a note and split into atomic
claims, and each claim is checked against the passages the generator saw. A fact sentence with
any unsupported claim is removed; if no fact sentence survives, the answer abstains."""

import json
from dataclasses import dataclass, field

from medquery import config
from medquery.eval.golden_gen import parse_json
from medquery.generation.prompts import prompt_hash, render_passages
from medquery.llm.base import LLM, Completion
from medquery.retrieval.dense import Hit
from medquery.verify.claims import render, split_sentences

VERIFY_VERSION = "verify_v1"
VERIFY_SETTINGS = {"temperature": 0.0, "seed": 0, "reasoning_effort": "low", "max_tokens": 2000}

VERIFY_SYSTEM = """You check an answer against source passages, sentence by sentence.
You do not use outside knowledge, the question, or plausibility: only what the passages state."""

VERIFY_TEMPLATE = """Passages:
{passages}

Question: {question}

Answer sentences:
{sentences}

For each numbered sentence:
1. "kind": "note" if the sentence only says that the passages do not cover something, or only
   warns that guidance is superseded; otherwise "fact".
2. "claims": for a fact sentence, its atomic claims (one checkable fact each). A claim is
   "supported": true only if the passages state it. Paraphrase is fine; numbers, units and
   qualifiers must match. For a note, give an empty list.

Reply with JSON only, one entry per sentence, in order:
{{"sentences": [{{"n": 1, "kind": "fact", "claims": [{{"text": "...", "supported": true}}]}}]}}"""

RETRY_MESSAGE = "Your reply was not valid ({error}). Reply again with the JSON object only."
VERIFY_HASH = prompt_hash(VERIFY_SYSTEM, VERIFY_TEMPLATE, RETRY_MESSAGE)


# verify_v2 (Phase 6, live app): the same prompt text with a higher token limit, and an empty or
# reasoning-only reply (no content) counts as a failed attempt at once. The settings are part
# of its hash, so it is a separate version; verify_v1 and its hash are unchanged (Phase 5).
VERIFY_V2_SETTINGS = {**VERIFY_SETTINGS, "max_tokens": 4000}
VERIFY_V2_HASH = prompt_hash(
    VERIFY_SYSTEM, VERIFY_TEMPLATE, RETRY_MESSAGE, json.dumps(VERIFY_V2_SETTINGS, sort_keys=True)
)
VERSIONS = {  # version -> (hash, settings, empty reply is a failure)
    "verify_v1": (VERIFY_HASH, VERIFY_SETTINGS, False),
    "verify_v2": (VERIFY_V2_HASH, VERIFY_V2_SETTINGS, True),
}
EMPTY_REPLY = "empty reply (no content; reasoning only)"


def verify_id(version: str = VERIFY_VERSION) -> str:
    return f"{version}:{VERSIONS[version][0]}"


@dataclass
class Verification:
    text: str
    abstained: bool
    sentences: list[dict] = field(default_factory=list)  # sentence, kind, claims, kept
    removed: list[str] = field(default_factory=list)
    failed: bool = False
    error: str = ""
    completions: list[Completion] = field(default_factory=list)


def verify_messages(question: str, hits: list[Hit], sentences: list[str]) -> list[dict]:
    numbered = "\n".join(f"{n}. {s}" for n, s in enumerate(sentences, 1))
    user = VERIFY_TEMPLATE.format(
        passages=render_passages(hits), question=question, sentences=numbered
    )
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


def statuses(v: "Verification", n_sentences: int) -> list[str]:
    """Per-sentence display status: supported, unsupported, note, or unverified (failed)."""
    if v.failed:
        return ["unverified"] * n_sentences
    out = []
    for s in v.sentences:
        if s["kind"] == "note":
            out.append("note")
        else:
            out.append("supported" if all(c["supported"] for c in s["claims"]) else "unsupported")
    return out


def validate(data: dict, n_sentences: int) -> list[dict]:
    """Schema check; returns one {kind, claims} per sentence or raises ValueError."""
    if not isinstance(data, dict) or not isinstance(data.get("sentences"), list):
        raise ValueError("'sentences' must be a list")
    items = data["sentences"]
    if len(items) != n_sentences:
        raise ValueError(f"expected {n_sentences} sentences, got {len(items)}")
    out = []
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict) or item.get("n") != n:
            raise ValueError(f'entry {n} must have "n": {n}')
        if item.get("kind") not in ("fact", "note"):
            raise ValueError(f"sentence {n}: kind must be 'fact' or 'note'")
        claims = item.get("claims")
        if not isinstance(claims, list):
            raise ValueError(f"sentence {n}: claims must be a list")
        for c in claims:
            if not isinstance(c, dict) or not isinstance(c.get("supported"), bool):
                raise ValueError(f"sentence {n}: each claim needs supported true or false")
        out.append(
            {
                "kind": item["kind"],
                "claims": [
                    {"text": str(c.get("text", "")).strip(), "supported": c["supported"]}
                    for c in claims
                ],
            }
        )
    return out


def verify(
    question: str,
    hits: list[Hit],
    answer: str,
    llm: LLM,
    *,
    model: str = config.GENERATOR_MODEL,
    provider: str | None = None,
    version: str = VERIFY_VERSION,
) -> Verification:
    """Verify one answer. One retry on an invalid reply; if that fails too, the original answer
    is kept unchanged and the failure is recorded (never silently)."""
    if model == config.JUDGE_MODEL:
        raise ValueError("the verifier must not be the judge model")
    sentences = split_sentences(answer)
    messages = verify_messages(question, hits, sentences)
    extra = {"provider": provider} if provider else {}
    _, settings, empty_fails = VERSIONS[version]
    completions: list[Completion] = []
    error = ""
    for attempt in range(2):
        completion = llm.complete(
            messages,
            model=model,
            response_format={"type": "json_object"},
            **settings,
            **extra,
        )
        completions.append(completion)
        try:
            if empty_fails and not completion.text.strip():
                raise ValueError(EMPTY_REPLY)
            verdicts = validate(parse_json(completion.text), len(sentences))
            break
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
                ]
    else:
        return Verification(answer, False, failed=True, error=error, completions=completions)

    is_fact = [v["kind"] == "fact" for v in verdicts]
    keep = [v["kind"] == "note" or all(c["supported"] for c in v["claims"]) for v in verdicts]
    text, abstained = render(answer, sentences, keep, is_fact)
    return Verification(
        text,
        abstained,
        sentences=[
            {"sentence": s, **v, "kept": k}
            for s, v, k in zip(sentences, verdicts, keep, strict=True)
        ],
        removed=[s for s, k in zip(sentences, keep, strict=True) if not k],
        completions=completions,
    )
