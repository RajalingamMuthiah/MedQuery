"""Baseline answer generation: retrieve top-k, one prompt, citations parsed from the reply."""

from dataclasses import dataclass

from medquery import config
from medquery.generation.prompts import answer_messages, extract_citations, is_abstention
from medquery.llm.base import LLM, Completion
from medquery.retrieval.dense import Hit, Retriever

NOT_MEDICAL_ADVICE = (
    "Not medical advice. MedQuery answers only from its source documents and does not "
    "diagnose or give dosing instructions. Consult a qualified health professional."
)
GENERATION = {"temperature": 0.0, "seed": 0, "reasoning_effort": "low", "max_tokens": 600}


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    abstained: bool
    cited_ids: list[str]
    hits: list[Hit]
    completion: Completion
    retrieval_s: float

    @property
    def invalid_citations(self) -> list[str]:
        retrieved = {h.chunk["id"] for h in self.hits}
        return [c for c in self.cited_ids if c not in retrieved]


def answer(
    question: str,
    retriever: Retriever,
    llm: LLM,
    *,
    k: int = config.TOP_K,
    model: str = config.GENERATOR_MODEL,
) -> Answer:
    hits, retrieval_s = retriever.retrieve(question, k)
    return answer_from_hits(question, hits, retrieval_s, llm, model=model)


def answer_from_hits(
    question: str,
    hits: list[Hit],
    retrieval_s: float,
    llm: LLM,
    *,
    model: str = config.GENERATOR_MODEL,
    provider: str | None = None,
    prompt_version: str = "answer_v1",
) -> Answer:
    """Generate from already-retrieved passages (the eval runner retrieves 10, uses the top k).
    `provider` routes the model to a non-default provider (e.g. NVIDIA instead of Groq)."""
    extra = {"provider": provider} if provider else {}
    messages = answer_messages(question, hits, prompt_version)
    completion = llm.complete(messages, model=model, **GENERATION, **extra)
    text = completion.text.strip()
    return Answer(
        question=question,
        text=text,
        abstained=is_abstention(text),
        cited_ids=extract_citations(text),
        hits=hits,
        completion=completion,
        retrieval_s=retrieval_s,
    )
