"""Generate golden-set candidates (one-time; the reviewed result is frozen as golden_vN).

Questions are written by the golden author model (a different family from the judge) from
pages, not chunks. Every LLM call is cached, so a stopped run resumes by re-running.
"""

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from medquery import config, paths
from medquery.eval.evidence import Evidence, EvidenceMapper
from medquery.eval.golden import (
    Item,
    check_evidence,
    check_question,
    copied_spans,
    is_duplicate,
    review_row,
    shared_ngram,
)
from medquery.generation.prompts import prompt_hash
from medquery.ingest.manifest import Document, Manifest
from medquery.llm.base import LLM, Completion
from medquery.retrieval.dense import Retriever

PAGES = paths.DATA_PROCESSED / "pages.jsonl"
SEED = 20261021
MIN_PAGE_WORDS, MAX_PAGE_WORDS = 80, 1400
VERIFY_DENSE_K = VERIFY_KEYWORD_K = 6  # 6 + 6 passages keep one request under 8K tokens/min

# Per topic: (LLM calls, questions per call). Totals carry ~25% headroom for rejections.
PLAN = {"factual": (7, 2), "table": (3, 2), "multi-chunk": (5, 1), "unanswerable": (4, 2)}
SUPERSEDED_PLAN = (3, 2)  # hypertension only: JNC 7 Express
OCR_FACTUAL_CALLS = 1  # one tuberculosis factual call uses the scanned 1964 report
TYPE_ORDER = ("factual", "table", "multi-chunk", "superseded", "unanswerable")

GEN_SYSTEM = (
    "You write evaluation questions for a question-answering system over public health "
    "documents. Follow the requirements exactly and reply with JSON only."
)
TYPE_INSTRUCTIONS = {
    "factual": "Ask about one specific fact stated in the running text: a number, "
    "recommendation, definition, criterion or time period.",
    "table": "Ask about a value or row in a TABLE on this page (tables appear as Markdown "
    "rows with | separators, often after a 'Table' caption). The answer must come from the "
    "table. Quote the table cells as evidence.",
    "multi-chunk": "Use BOTH pages: the answer must combine one fact from the first page with "
    "one fact from the second page. Give at least one quote from each page.",
    "superseded": "This document is JNC 7 (2003), which has been superseded by later "
    "hypertension guidance. Ask what JNC 7 says, naming 'JNC 7' in the question. The reference "
    "answer must give the JNC 7 (2003) guidance and note that it has since been superseded.",
}
ANSWERABLE_TEMPLATE = """{pages}

Write {n} question(s) of type "{type}". {instruction}
Requirements:
- Each question must be answerable from {scope} alone, by someone who has not seen it.
- Write each question in your own words. Never copy 6 or more consecutive words from the page.
- Make each question specific and self-contained: name the condition, drug or group.
- No yes/no questions. If you write two questions, they must be about different facts.
- reference_answer: 1-3 sentences, correct and complete, using only the page text.
- evidence: 1-3 quotes copied EXACTLY (character for character) from the page, 5-25 words each,
  that together support the answer, each with the page number it comes from.
Reply with JSON: {{"items": [{{"question": "...", "reference_answer": "...",
"evidence": [{{"page": 0, "quote": "..."}}]}}]}}"""
UNANSWERABLE_TEMPLATE = """{pages}

The whole document collection contains only these documents:
{titles}

Write {n} question(s) that a reader of this page could plausibly ask next: about the same
condition, test or treatment, but whose specific answer is NOT on this page and is unlikely to
be in any document listed above (for example a specific subgroup, a newer drug, or a precise
figure these documents do not report). Stay on the same topic: no trick questions, no other
diseases, no questions about the documents themselves.
- Write each question in your own words. Never copy 6 or more consecutive words from the page.
- No yes/no questions.
Reply with JSON: {{"items": [{{"question": "...", "why_not_answerable": "..."}}]}}"""

COPY_REASON = "question copies"
REPHRASE_TEMPLATE = """Rewrite this question so that it keeps exactly the same meaning and the same
answer and still reads naturally, but contains none of these phrases (rephrase them or change
the word order):
{phrases}

Question: {question}

Reply with JSON: {{"question": "..."}}"""

VERIFY_SYSTEM = (
    "You check whether a question can be answered from a set of passages. Judge only from the "
    "passages. Reply with JSON only."
)
VERIFY_TEMPLATE = """Question: {question}

Passages:
{passages}

Does any passage contain the information needed to answer the question, fully or partly?
Reply with JSON: {{"answer_found": true|false, "partial": true|false, "passage_ids": [],
"explanation": "..."}}"""

GEN_HASH = prompt_hash(
    REPHRASE_TEMPLATE,
    GEN_SYSTEM,
    ANSWERABLE_TEMPLATE,
    UNANSWERABLE_TEMPLATE,
    *TYPE_INSTRUCTIONS.values(),
)
VERIFY_HASH = prompt_hash(VERIFY_SYSTEM, VERIFY_TEMPLATE)

_STOPWORD_TEXT = (
    "the and for with that this what which who whom how are was were from into about does did "
    "can could should would may might than then them they their there these those has have had "
    "not any all per according recommended recommend guideline guidelines document documents "
    "people persons patients patient adults"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


@dataclass(frozen=True)
class SourcePage:
    doc_id: str
    page: int
    text: str
    source: str
    ocr_confidence: float | None
    tables: int


@dataclass
class Task:
    type: str
    topic: str
    pages: list[SourcePage]
    n: int


@dataclass
class GenerationResult:
    candidates: list[dict] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)


def load_pages(path=PAGES) -> list[SourcePage]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m medquery.ingest build` first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        SourcePage(r["doc_id"], r["page"], r["text"], r["source"], r["ocr_confidence"], r["tables"])
        for r in rows
    ]


def page_coverage(chunks: list[dict]) -> dict[tuple[str, int], set[str]]:
    """The chunk kinds (text/table/quiz) covering each (doc_id, page)."""
    covered: dict[tuple[str, int], set[str]] = defaultdict(set)
    for chunk in chunks:
        for page in range(chunk["page_start"], chunk["page_end"] + 1):
            covered[(chunk["doc_id"], page)].add(chunk["kind"])
    return covered


def eligible(page: SourcePage, kinds: set[str]) -> bool:
    """Body pages with real content: no covers, quiz pages, reference lists or noisy scans."""
    text = page.text
    if not kinds or "quiz" in kinds or page.page <= 2 or len(text.split()) < MIN_PAGE_WORDS:
        return False
    if len(re.findall(r"et al\b", text)) >= 4 or text.lower().count("doi") >= 3:
        return False
    if page.source == "ocr":
        tokens = text.split()
        alpha = sum(1 for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z,.;:'()-]*", t))
        return alpha / len(tokens) >= 0.6 and (page.ocr_confidence or 0) >= 75
    return True


def plan_tasks(
    pages: list[SourcePage], chunks: list[dict], docs: dict[str, Document], seed: int = SEED
) -> list[Task]:
    """Seeded sampling of source pages: per topic and type, spread round-robin over documents."""
    rng = random.Random(seed)
    covered = page_coverage(chunks)
    usable = [p for p in pages if eligible(p, covered.get((p.doc_id, p.page), set()))]
    used: set[tuple[str, int]] = set()

    def pick(pool: list[SourcePage], calls: int, span: int = 1) -> list[SourcePage]:
        """Pick `calls` start pages, round-robin over documents. Each pick occupies `span`
        consecutive pages, and a page is never used by two tasks."""

        def free(p: SourcePage) -> bool:
            return all((p.doc_id, p.page + i) not in used for i in range(span))

        by_doc: dict[str, list[SourcePage]] = defaultdict(list)
        for p in pool:
            by_doc[p.doc_id].append(p)
        order = sorted(by_doc)
        rng.shuffle(order)
        chosen: list[SourcePage] = []
        while len(chosen) < calls:
            progressed = False
            for doc_id in order:
                options = [p for p in by_doc[doc_id] if free(p)]
                if options and len(chosen) < calls:
                    page = options[rng.randrange(len(options))]
                    chosen.append(page)
                    used.update((page.doc_id, page.page + i) for i in range(span))
                    progressed = True
            if not progressed:
                break
        return chosen

    tasks: list[Task] = []
    for topic in sorted({d.topic for d in docs.values()}):
        in_topic = [p for p in usable if docs[p.doc_id].topic == topic]
        current = [p for p in in_topic if not docs[p.doc_id].superseded]
        text_pages = [p for p in current if p.source == "text"]
        ocr_pages = [p for p in current if p.source == "ocr"]
        table_pages = [p for p in text_pages if "table" in covered[(p.doc_id, p.page)] and p.tables]

        calls, n = PLAN["factual"]
        ocr_calls = min(OCR_FACTUAL_CALLS, len(ocr_pages))
        for page in pick(ocr_pages, ocr_calls) + pick(text_pages, calls - ocr_calls):
            tasks.append(Task("factual", topic, [page], n))
        calls, n = PLAN["table"]
        tasks += [Task("table", topic, [p], n) for p in pick(table_pages, calls)]

        calls, n = PLAN["multi-chunk"]
        by_key = {(p.doc_id, p.page): p for p in text_pages}
        starts = [p for p in text_pages if (p.doc_id, p.page + 1) in by_key]
        for first in pick(starts, calls, span=2):
            second = by_key[(first.doc_id, first.page + 1)]
            tasks.append(Task("multi-chunk", topic, [first, second], n))

        superseded_pages = [p for p in in_topic if docs[p.doc_id].superseded and p.source == "text"]
        if superseded_pages:
            calls, n = SUPERSEDED_PLAN
            tasks += [Task("superseded", topic, [p], n) for p in pick(superseded_pages, calls)]

        calls, n = PLAN["unanswerable"]
        tasks += [Task("unanswerable", topic, [p], n) for p in pick(text_pages, calls)]
    return tasks


def render_pages(task: Task, docs: dict[str, Document]) -> str:
    blocks = []
    for page in task.pages:
        doc = docs[page.doc_id]
        text = " ".join(page.text.split(" ")[:MAX_PAGE_WORDS])
        blocks.append(
            f"Document: {doc.title} ({doc.publisher}, as of {doc.as_of})\n"
            f"Page {page.page}:\n<<<\n{text}\n>>>"
        )
    return "\n\n".join(blocks)


def generation_messages(task: Task, docs: dict[str, Document]) -> list[dict]:
    pages = render_pages(task, docs)
    if task.type == "unanswerable":
        titles = "\n".join(f"- {d.title}" for d in docs.values())
        user = UNANSWERABLE_TEMPLATE.format(pages=pages, titles=titles, n=task.n)
    else:
        scope = "these two pages together" if len(task.pages) == 2 else "this page"
        user = ANSWERABLE_TEMPLATE.format(
            pages=pages,
            n=task.n,
            type=task.type,
            instruction=TYPE_INSTRUCTIONS[task.type],
            scope=scope,
        )
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]


def parse_json(text: str) -> dict:
    """Parse a JSON reply, tolerating a code fence or prose around the object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in STOPWORDS}


class KeywordScan:
    """Term overlap between the question and each chunk. Only used to check that unanswerable
    questions really are unanswerable; it is not the Phase 3 BM25 retriever."""

    def __init__(self, chunks: list[dict]):
        self._chunks = [(c, _terms(c["text"])) for c in chunks]

    def top(self, question: str, k: int, exclude: set[str]) -> list[dict]:
        query = _terms(question)
        scored = [
            (len(query & terms), -len(c["text"]), c)
            for c, terms in self._chunks
            if c["id"] not in exclude and query & terms
        ]
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [c for _, _, c in scored[:k]]


def rephrase(question: str, source: str, llm: LLM) -> tuple[str, Completion]:
    """One rewrite of a question that copied phrases from its page (the 6-word rule stays)."""
    phrases = "\n".join(f'- "{span}"' for span in copied_spans(question, source))
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": REPHRASE_TEMPLATE.format(phrases=phrases, question=question)},
    ]
    completion = llm.complete(
        messages,
        model=config.GOLDEN_AUTHOR_MODEL,
        temperature=0.0,
        max_tokens=300,
        reasoning_effort="none",
        response_format={"type": "json_object"},
    )
    try:
        rewritten = str(parse_json(completion.text).get("question", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        rewritten = ""
    return rewritten, completion


def verify_unanswerable(
    question: str, retriever: Retriever, scan: KeywordScan, llm: LLM
) -> tuple[dict, Completion]:
    hits, _ = retriever.retrieve(question, VERIFY_DENSE_K)
    chunks = [h.chunk for h in hits]
    chunks += scan.top(question, VERIFY_KEYWORD_K, {c["id"] for c in chunks})
    passages = "\n---\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_TEMPLATE.format(question=question, passages=passages)},
    ]
    completion = llm.complete(
        messages,
        model=config.JUDGE_MODEL,
        max_tokens=800,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    verdict = parse_json(completion.text)
    verdict["checked_ids"] = [c["id"] for c in chunks]
    return verdict, completion


def generate(
    manifest: Manifest,
    pages: list[SourcePage],
    chunks: list[dict],
    retriever: Retriever,
    llm: LLM,
    log=print,
) -> GenerationResult:
    """Run every task, validate every candidate, and return all of them with a status."""
    docs = {d.id: d for d in manifest.documents}
    mapper = EvidenceMapper(chunks)
    scan = KeywordScan([c for c in chunks if c["kind"] != "quiz"])
    tasks = plan_tasks(pages, chunks, docs)
    result = GenerationResult()
    accepted: list[str] = []
    log(f"{len(tasks)} generation calls planned: {dict(Counter(t.type for t in tasks))}")

    for number, task in enumerate(tasks, 1):
        completion = llm.complete(
            generation_messages(task, docs),
            model=config.GOLDEN_AUTHOR_MODEL,
            temperature=0.0,
            max_tokens=1500,
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        result.completions.append(completion)
        source_pages = [(p.doc_id, p.page) for p in task.pages]
        page_texts = {(p.doc_id, p.page): p.text for p in task.pages}
        try:
            raw_items = parse_json(completion.text).get("items", [])
        except (json.JSONDecodeError, AttributeError):
            raw_items = []
            result.candidates.append(_candidate(task, {}, source_pages, "reply was not valid JSON"))
        for raw in raw_items[: task.n]:
            reason, extra = _validate(task, raw, page_texts, mapper, accepted)
            if reason and reason.startswith(COPY_REASON):
                source = " ".join(page_texts.values())
                rewritten, retry = rephrase(str(raw.get("question", "")), source, llm)
                result.completions.append(retry)
                extra = {"original_question": raw.get("question", ""), "copy_reason": reason}
                if rewritten:
                    raw = {**raw, "question": rewritten}
                    reason, more = _validate(task, raw, page_texts, mapper, accepted)
                    extra.update(more)
                    if reason:
                        reason = f"after one rewrite: {reason}"
                else:
                    reason = "rewrite reply was not valid JSON"
            if reason is None and task.type == "unanswerable":
                verdict, check = verify_unanswerable(raw["question"], retriever, scan, llm)
                result.completions.append(check)
                extra["verification"] = verdict
                if verdict.get("answer_found") or verdict.get("partial"):
                    reason = "answer found in corpus by verification check"
            if reason is None:
                accepted.append(raw["question"])
            result.candidates.append(_candidate(task, raw, source_pages, reason, extra))
        log(f"  [{number}/{len(tasks)}] {task.type:<12} {task.topic:<12} {source_pages}")
    return result


def _validate(task: Task, raw: dict, page_texts, mapper, accepted) -> tuple[str | None, dict]:
    question = str(raw.get("question", "")).strip()
    extra: dict = {}
    if reason := check_question(question):
        return reason, extra
    source = " ".join(page_texts.values())
    if copied := shared_ngram(question, source):
        return f"{COPY_REASON} {len(copied.split())} words from the page: {copied!r}", extra
    if is_duplicate(question, accepted):
        return "near-duplicate of an accepted question", extra
    if task.type == "unanswerable":
        return None, extra
    doc_id = task.pages[0].doc_id
    evidence = [
        Evidence(doc_id, int(e.get("page", 0)), str(e.get("quote", "")).strip())
        for e in raw.get("evidence", [])
        if isinstance(e, dict)
    ]
    if reason := check_evidence(evidence, page_texts, mapper):
        return reason, extra
    if task.type == "multi-chunk" and len({e.page for e in evidence}) < 2:
        return "multi-chunk evidence does not use both pages", extra
    if not str(raw.get("reference_answer", "")).strip():
        return "empty reference answer", extra
    return None, extra


def _candidate(task: Task, raw: dict, source_pages, reason, extra=None) -> dict:
    answerable = task.type != "unanswerable"
    doc_id = task.pages[0].doc_id
    return {
        "type": task.type,
        "topic": task.topic,
        "answerable": answerable,
        "question": str(raw.get("question", "")).strip(),
        "reference_answer": str(raw.get("reference_answer", "")).strip()
        if answerable
        else "Not answerable from the documents.",
        "evidence": [
            {"doc_id": doc_id, "page": int(e.get("page", 0)), "quote": str(e.get("quote", ""))}
            for e in raw.get("evidence", [])
            if isinstance(e, dict)
        ]
        if answerable
        else [],
        "generator_note": str(raw.get("why_not_answerable", "")).strip(),
        "source_pages": [{"doc_id": d, "page": p} for d, p in source_pages],
        "gen_model": config.GOLDEN_AUTHOR_MODEL,
        "gen_prompt": f"golden_gen_v1:{GEN_HASH}",
        "status": "accepted" if reason is None else "rejected",
        "reject_reason": reason or "",
        **(extra or {}),
    }


def to_items(candidates: list[dict]) -> list[Item]:
    """Accepted candidates as items with stable ids g001..., ordered by topic and type."""
    accepted = [c for c in candidates if c["status"] == "accepted"]
    accepted.sort(key=lambda c: (c["topic"], TYPE_ORDER.index(c["type"])))
    return [
        Item(
            id=f"g{n:03d}",
            question=c["question"],
            reference_answer=c["reference_answer"],
            evidence=[Evidence(e["doc_id"], e["page"], e["quote"]) for e in c["evidence"]],
            type=c["type"],
            topic=c["topic"],
            answerable=c["answerable"],
            source_pages=[(s["doc_id"], s["page"]) for s in c["source_pages"]],
            gen_model=c["gen_model"],
            generator_note=c["generator_note"],
        )
        for n, c in enumerate(accepted, 1)
    ]


def review_rows(items: list[Item], docs: dict[str, Document]) -> list[dict]:
    rows = []
    for item in items:
        doc_id = item.source_pages[0][0]
        pages = " and ".join(f"p.{p}" for _, p in item.source_pages)
        rows.append(review_row(item, f"{docs[doc_id].title}, {pages}"))
    return rows
