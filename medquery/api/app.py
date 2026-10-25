"""MedQuery API: the live config (hybrid + headers + corpus v2 + answer_v2, no reranker) with
generation on Groq, and an optional on-demand claim check (verify_v2). Everything reuses the
medquery package; this module only wires it to HTTP and adds the request limits."""

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from medquery import config as settings
from medquery import paths
from medquery.api.answer_cache import AnswerCache
from medquery.api.budget import ANSWER_EXPECTED_TOKENS, VERIFY_EXPECTED_TOKENS
from medquery.api.highlight import highlights
from medquery.api.limits import DailyCap, RateLimiter
from medquery.api.schemas import AskRequest, VerifyRequest
from medquery.generation.answer import GENERATION, NOT_MEDICAL_ADVICE, answer_from_hits
from medquery.generation.prompts import answer_messages, extract_citations, is_abstention
from medquery.llm import LLMError, QuotaExhausted
from medquery.retrieval.dense import Hit
from medquery.verify.claims import split_sentences
from medquery.verify.verifier import VERSIONS as VERIFY_VERSIONS
from medquery.verify.verifier import statuses, verify, verify_id, verify_messages

LIVE_PROVIDER = "groq"
VERIFY_VERSION = "verify_v2"
USAGE_PATH = paths.ROOT / "data" / "api_usage.json"
ANSWER_CACHE_PATH = paths.ROOT / "data" / "api_answer_cache.json"
SAMPLES = {"answerable_per_topic": 2, "unanswerable": 2}


def live_config():
    from medquery.eval.runner import CONFIGS

    return replace(CONFIGS["hybrid_headers_tables_answer_v2"], generator_provider=LIVE_PROVIDER)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def passage_view(chunk: dict) -> dict:
    return {
        "id": chunk["id"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "publisher": chunk["publisher"],
        "as_of": chunk["as_of"],
        "superseded": bool(chunk.get("superseded")),
        "superseded_note": chunk.get("superseded", ""),
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "section": chunk.get("section", ""),
        "kind": chunk["kind"],
        "text": chunk["text"],
    }


def sample_questions(items: list[dict]) -> list[dict]:
    """A fixed choice of dev questions only: the first answerable ones per topic (by id), then
    the first unanswerable ones. Test-split items are never considered."""
    dev = sorted((i for i in items if i["split"] == "dev"), key=lambda i: i["id"])
    chosen: list[dict] = []
    for topic in sorted({i["topic"] for i in dev}):
        answerable = [i for i in dev if i["topic"] == topic and i["answerable"]]
        chosen += answerable[: SAMPLES["answerable_per_topic"]]
    chosen += [i for i in dev if not i["answerable"]][: SAMPLES["unanswerable"]]
    return [{"id": i["id"], "question": i["question"], "topic": i["topic"]} for i in chosen]


def documents_view() -> list[dict]:
    from medquery.ingest.manifest import load_manifest

    return [
        {
            "id": d.id,
            "title": d.title,
            "publisher": d.publisher,
            "year": d.year,
            "topic": d.topic,
            "as_of": d.as_of,
            "licence": d.licence,
            "superseded": d.superseded,
            "url": d.url,
        }
        for d in load_manifest(paths.MANIFEST).documents
    ]


class AppState:
    """What the background loader fills in; requests that need it wait for `ready`."""

    def __init__(self):
        self.ready = False
        self.error = ""
        self.retriever = None
        self.chunks: dict[str, dict] = {}
        self.corpus_sha = ""


def default_loader(config):
    """Load corpus v2, build the hybrid retriever (embeddings from cache, BM25), load the
    embedding model and run one warm-up query so the first real request is fast."""
    from medquery.eval.freeze import sha256_file
    from medquery.eval.runner import build_retriever
    from medquery.retrieval.dense import chunks_path, load_chunks
    from medquery.retrieval.embed import FastEmbedder

    corpus = chunks_path(config.corpus)
    if not corpus.exists():
        raise RuntimeError(
            f"{corpus} is missing: build corpus v2 first "
            "(uv run python -m medquery.ingest build --corpus v2)"
        )
    loaded = load_chunks(corpus)
    retriever = build_retriever(config, loaded, FastEmbedder(config.embed_model))
    retriever.retrieve("warm-up query", config.k)
    return retriever, {c["id"]: c for c in loaded}, sha256_file(corpus)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    *,
    retriever=None,
    llm=None,
    chunks: dict[str, dict] | None = None,
    golden: list[dict] | None = None,
    loader=None,
    usage_path: Path = USAGE_PATH,
    answer_cache: AnswerCache | None = None,
    rate_per_min: int | None = None,
    daily_cap: int | None = None,
    cors_origin: str | None = None,
) -> FastAPI:
    """The app. Tests pass a fake retriever, LLM, chunks and golden items (ready at once) or a
    loader; by default the live config loads in the background at startup."""
    config = live_config()
    state = AppState()
    if retriever is not None and chunks is not None:
        state.retriever, state.chunks, state.ready = retriever, chunks, True
    if llm is None:
        from medquery.llm import get_client

        llm = get_client()
    if golden is None:
        from medquery.eval.freeze import load_golden

        golden = load_golden(config.golden_version)[0]
    cache = answer_cache if answer_cache is not None else AnswerCache(ANSWER_CACHE_PATH)

    def load() -> None:
        try:
            state.retriever, state.chunks, state.corpus_sha = (loader or default_loader)(config)
            state.ready = True
        except Exception as exc:  # reported by /health; requests get 503
            state.error = f"{type(exc).__name__}: {exc}"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not state.ready:
            threading.Thread(target=load, name="medquery-warm-start", daemon=True).start()
        yield

    limiter = RateLimiter(rate_per_min or _env_int("MEDQUERY_RATE_PER_MIN", 6))
    cap = DailyCap(usage_path, daily_cap or _env_int("MEDQUERY_DAILY_CAP", 60))
    app = FastAPI(title="MedQuery", version="0.7.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            cors_origin or os.environ.get("MEDQUERY_CORS_ORIGIN", "http://localhost:5173")
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message, "notice": NOT_MEDICAL_ADVICE}, status_code=status)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        messages = [str(e.get("msg", "")).removeprefix("Value error, ") for e in exc.errors()]
        return error(422, "; ".join(messages) or "invalid request")

    def require_ready() -> None:
        if not state.ready:
            detail = (
                f"MedQuery could not start: {state.error}"
                if state.error
                else "MedQuery is starting up; try again in a few seconds"
            )
            raise HTTPException(503, detail)

    def per_ip(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "too many requests: please wait a minute and try again")

    def daily() -> None:
        if cap.remaining() <= 0:
            raise HTTPException(
                429, "the daily request limit for this demo has been reached; try again tomorrow"
            )

    def count(completions) -> None:
        if any(not c.cached for c in completions):
            cap.record()

    def call_failed(exc: LLMError) -> HTTPException:
        if isinstance(exc, QuotaExhausted):
            return HTTPException(503, "the language model's daily quota is used up; try later")
        return HTTPException(503, f"the language model could not be reached: {exc}")

    def identity() -> str:
        return f"{config.answer_prompt}|{state.corpus_sha}|{config.generator}|{LIVE_PROVIDER}"

    def build_response(question, hits, retrieval_s, completion, ttft_s=None) -> dict:
        text = completion.text.strip()
        abstained = is_abstention(text)
        by_id = {h.chunk["id"]: h.chunk for h in hits}
        sentence_views = []
        for s in [] if abstained else split_sentences(text):
            cited = [c for c in extract_citations(s) if c in by_id]
            sentence_views.append(
                {
                    "text": s,
                    "citations": cited,
                    "highlights": {c: highlights(s, by_id[c]["text"]) for c in cited},
                }
            )
        return {
            "question": question,
            "answer": text,
            "abstained": abstained,
            "sentences": sentence_views,
            "passages": [passage_view(h.chunk) for h in hits],
            "latency": {
                "retrieval_ms": round(retrieval_s * 1000, 1),
                "generation_s": completion.latency_s,
                "ttft_s": ttft_s,
            },
            "model": config.generator,
            "cached": False,
            "notice": NOT_MEDICAL_ADVICE,
        }

    def cached_answer(question: str) -> dict | None:
        hit = cache.get(question, identity())
        return None if hit is None else {**hit, "question": question, "cached": True}

    def prompt_version() -> str:
        return config.answer_prompt.split(":")[0]

    @app.get("/health")
    def health():
        status = "ok" if state.ready else ("error" if state.error else "starting")
        return {
            "status": status,
            "ready": state.ready,
            "error": state.error or None,
            "config": {
                "name": "live",
                "retriever": config.retriever,
                "headers": config.headers,
                "reranker": config.rerank_model or None,
                "corpus": config.corpus,
                "corpus_sha256": state.corpus_sha,
                "answer_prompt": config.answer_prompt,
                "verifier": verify_id(VERIFY_VERSION),
                "generator": config.generator,
                "provider": LIVE_PROVIDER,
            },
            "keys_present": {
                name: bool(os.environ.get(p["key_env"])) for name, p in settings.PROVIDERS.items()
            },
            "daily_remaining": cap.remaining(),
            "answer_cache_entries": len(cache),
            "notice": NOT_MEDICAL_ADVICE,
        }

    @app.get("/documents")
    def documents():
        return {"documents": documents_view(), "notice": NOT_MEDICAL_ADVICE}

    @app.get("/samples")
    def samples():
        return {"samples": sample_questions(golden), "notice": NOT_MEDICAL_ADVICE}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        require_ready()
        per_ip(request)
        if (hit := cached_answer(body.question)) is not None:
            return hit
        daily()
        hits, retrieval_s = state.retriever.retrieve(body.question, config.k)
        try:
            result = answer_from_hits(
                body.question,
                hits,
                retrieval_s,
                llm,
                model=config.generator,
                provider=LIVE_PROVIDER,
                prompt_version=prompt_version(),
                expected_output_tokens=ANSWER_EXPECTED_TOKENS,
            )
        except LLMError as exc:
            raise call_failed(exc) from exc
        count([result.completion])
        response = build_response(body.question, hits, retrieval_s, result.completion)
        cache.put(body.question, identity(), response)
        return response

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, request: Request):
        require_ready()
        per_ip(request)
        hit = cached_answer(body.question)
        if hit is None:
            daily()

        def events():
            if hit is not None:
                yield sse("final", hit)
                return
            hits, retrieval_s = state.retriever.retrieve(body.question, config.k)
            yield sse(
                "retrieval",
                {
                    "passages": [passage_view(h.chunk) for h in hits],
                    "retrieval_ms": round(retrieval_s * 1000, 1),
                },
            )
            messages = answer_messages(body.question, hits, prompt_version())
            try:
                stream = llm.stream(
                    messages,
                    model=config.generator,
                    provider=LIVE_PROVIDER,
                    **GENERATION,
                    expected_output_tokens=ANSWER_EXPECTED_TOKENS,
                )
                for delta in stream:
                    yield sse("delta", {"text": delta})
            except LLMError as exc:
                yield sse("error", {"error": str(call_failed(exc).detail), "status": 503})
                return
            count([stream.completion])
            response = build_response(
                body.question, hits, retrieval_s, stream.completion, stream.first_token_s
            )
            cache.put(body.question, identity(), response)
            yield sse("final", response)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    def verify_inputs(body: VerifyRequest) -> tuple[list[Hit], list[str]]:
        unknown = [p for p in body.passage_ids if p not in state.chunks]
        if unknown:
            raise HTTPException(422, f"unknown passage ids: {unknown}")
        sentences = split_sentences(body.answer)
        if not sentences:
            raise HTTPException(422, "the answer has no sentences to check")
        hits = [Hit(state.chunks[p], 0.0, n) for n, p in enumerate(body.passage_ids, 1)]
        return hits, sentences

    @app.post("/verify/estimate")
    def verify_estimate(body: VerifyRequest):
        """How long a check would wait for the rate limit right now (no call, not counted)."""
        require_ready()
        hits, sentences = verify_inputs(body)
        wait_s = llm.estimate_wait(
            verify_messages(body.question, hits, sentences),
            model=config.generator,
            provider=LIVE_PROVIDER,
            max_tokens=VERIFY_VERSIONS[VERIFY_VERSION][1]["max_tokens"],
            expected_output_tokens=VERIFY_EXPECTED_TOKENS,
        )
        return {"wait_s": wait_s, "notice": NOT_MEDICAL_ADVICE}

    @app.post("/verify")
    def verify_answer(body: VerifyRequest, request: Request):
        require_ready()
        hits, sentences = verify_inputs(body)
        per_ip(request)
        daily()
        started = time.perf_counter()
        try:
            v = verify(
                body.question,
                hits,
                body.answer,
                llm,
                model=config.generator,
                provider=LIVE_PROVIDER,
                version=VERIFY_VERSION,
                expected_output_tokens=VERIFY_EXPECTED_TOKENS,
            )
        except LLMError as exc:
            raise call_failed(exc) from exc
        count(v.completions)
        status = statuses(v, len(sentences))
        return {
            "sentences": [
                {"text": s, "status": st} for s, st in zip(sentences, status, strict=True)
            ],
            "verified": not v.failed,
            "verifier": verify_id(VERIFY_VERSION),
            "latency_s": round(time.perf_counter() - started, 3),
            "wait_s": round(sum(c.wait_s for c in v.completions), 3),
            "notice": NOT_MEDICAL_ADVICE,
        }

    return app
