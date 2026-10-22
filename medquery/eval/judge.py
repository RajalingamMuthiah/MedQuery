"""The LLM judge (judge_v1): one call per answered question.

Claim support is decided ONLY from the retrieved passages. The reference answer comes last, in
its own section, and is used only for the correctness field.
"""

import json
from dataclasses import dataclass

from medquery import config
from medquery.eval.golden_gen import parse_json
from medquery.generation.prompts import prompt_hash
from medquery.llm.base import LLM, Completion
from medquery.retrieval.dense import Hit

JUDGE_VERSION = "judge_v1"
CORRECTNESS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a question-answering system over health "
    "documents. Follow the instructions exactly and reply with JSON only."
)
JUDGE_TEMPLATE = """Question:
{question}

Retrieved passages (the ONLY evidence for support decisions):
{passages}

Answer to evaluate:
{answer}

Instructions for claims:
1. Split the answer into atomic claims: short, self-contained factual statements. Skip citation
   markers, disclaimers, and statements that the documents do not cover something.
2. "cited_ids": the passage ids the answer cites for that claim (the ids in square brackets on
   its sentence), or [] if none.
3. "supported_by_context": true only if at least one retrieved passage above fully supports the
   claim. Decide this ONLY from the passages, never from your own knowledge and never from the
   reference answer below.
4. "supported_by_cited": true if the passages in cited_ids fully support the claim, false if they
   do not, null if cited_ids is empty. Decide this ONLY from the passages.
5. "relevancy": 1-5, how directly the answer addresses the question (5 = fully and directly,
   1 = off-topic).

Reference answer (use it ONLY for the "correctness" field below; do NOT use it to decide
supported_by_context or supported_by_cited):
{reference}

6. "correctness": compare the answer with the reference answer. "correct" if it states the same
   key facts without contradicting it; "partial" if it gives some but not all key facts, or adds a
   minor error; "incorrect" if it misses the key facts or contradicts the reference.

Reply with JSON: {{"claims": [{{"text": "...", "cited_ids": [], "supported_by_context": true,
"supported_by_cited": null}}], "relevancy": 5, "correctness": "correct", "rationale": "..."}}"""
RETRY_MESSAGE = "Your reply was not valid ({error}). Reply again with the JSON object only."
JUDGE_HASH = prompt_hash(JUDGE_SYSTEM, JUDGE_TEMPLATE, RETRY_MESSAGE)
JUDGE_SETTINGS = {"temperature": 0.0, "seed": 0, "reasoning_effort": "low", "max_tokens": 2000}


@dataclass(frozen=True)
class Verdict:
    claims: list[dict]
    relevancy: int
    correctness: str
    rationale: str


class JudgeFailed(Exception):
    def __init__(self, error: str, completions: list[Completion]):
        super().__init__(error)
        self.completions = completions  # still counted in token and cost totals


def judge_messages(question: str, hits: list[Hit], answer: str, reference: str) -> list[dict]:
    passages = "\n---\n".join(f"[{h.chunk['id']}]\n{h.chunk['text']}" for h in hits)
    user = JUDGE_TEMPLATE.format(
        question=question, passages=passages, answer=answer, reference=reference
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def validate(data: dict) -> Verdict:
    """Schema check for a judge reply; raises ValueError with the first problem found."""
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list")
    for n, claim in enumerate(claims, 1):
        if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
            raise ValueError(f"claim {n} has no text")
        if not isinstance(claim.get("cited_ids", []), list):
            raise ValueError(f"claim {n}: cited_ids must be a list")
        if not isinstance(claim.get("supported_by_context"), bool):
            raise ValueError(f"claim {n}: supported_by_context must be true or false")
        if claim.get("supported_by_cited") not in (True, False, None):
            raise ValueError(f"claim {n}: supported_by_cited must be true, false or null")
    relevancy = data.get("relevancy")
    if not isinstance(relevancy, int) or not 1 <= relevancy <= 5:
        raise ValueError("'relevancy' must be an integer 1-5")
    if data.get("correctness") not in CORRECTNESS:
        raise ValueError(f"'correctness' must be one of {sorted(CORRECTNESS)}")
    return Verdict(
        claims=[
            {
                "text": str(c["text"]).strip(),
                "cited_ids": [str(i) for i in c.get("cited_ids", [])],
                "supported_by_context": c["supported_by_context"],
                "supported_by_cited": c.get("supported_by_cited") if c.get("cited_ids") else None,
            }
            for c in claims
        ],
        relevancy=relevancy,
        correctness=data["correctness"],
        rationale=str(data.get("rationale", "")),
    )


def judge(
    question: str, hits: list[Hit], answer: str, reference: str, llm: LLM
) -> tuple[Verdict, list[Completion]]:
    """Judge one answer. One retry on an invalid reply, then JudgeFailed (never silently dropped).

    The retry adds the bad reply and the error to the conversation, so it is a new request
    rather than a cache hit on the same invalid reply."""
    messages = judge_messages(question, hits, answer, reference)
    completions: list[Completion] = []
    for attempt in range(2):
        completion = llm.complete(
            messages,
            model=config.JUDGE_MODEL,
            response_format={"type": "json_object"},
            **JUDGE_SETTINGS,
        )
        completions.append(completion)
        try:
            return validate(parse_json(completion.text)), completions
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
                ]
    raise JudgeFailed(error, completions)
