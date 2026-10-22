"""Resumable evaluation runs: one JSON record per golden item, appended as each finishes."""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from medquery import config as settings
from medquery import paths
from medquery.eval.evidence import Evidence, EvidenceMapper
from medquery.eval.judge import JUDGE_VERSION, JudgeFailed, judge, prompt_id
from medquery.fileio import atomic_write, atomic_write_bytes, exclusive_lock
from medquery.generation.answer import answer_from_hits
from medquery.generation.prompts import ANSWER_HASH, ANSWER_VERSION
from medquery.llm.base import LLM, Completion
from medquery.retrieval.dense import Retriever
from medquery.retrieval.fusion import RRF_K
from medquery.retrieval.pipeline import HybridRetriever
from medquery.retrieval.rerank import FastEmbedReranker

RUNS_DIR = paths.ROOT / "data" / "eval_runs"


@dataclass(frozen=True)
class RunConfig:
    name: str = "baseline"
    golden_version: str = "v1"
    k: int = settings.TOP_K
    recall_k: int = 10
    embed_model: str = settings.EMBED_MODEL
    generator: str = settings.GENERATOR_MODEL
    judge: str = settings.JUDGE_MODEL
    include_quiz: bool = False
    answer_prompt: str = f"{ANSWER_VERSION}:{ANSWER_HASH}"
    judge_prompt: str = prompt_id(JUDGE_VERSION)
    # Phase 3 retrieval settings. At their defaults they are left out of the run id, so the
    # Phase 2 baseline keeps its id and records.
    retriever: str = "dense"  # "dense" or "hybrid" (dense + BM25 with RRF)
    fuse_k: int = RRF_K
    rerank_model: str = ""
    rerank_n: int = 20
    headers: bool = False


RETRIEVAL_DEFAULTS = {
    "retriever": "dense",
    "fuse_k": RRF_K,
    "rerank_model": "",
    "rerank_n": 20,
    "headers": False,
}
_HYBRID = RunConfig(name="hybrid", retriever="hybrid")
_RERANK = replace(_HYBRID, name="hybrid_rerank", rerank_model="BAAI/bge-reranker-base")
_HEADERS = replace(_RERANK, name="hybrid_rerank_headers", headers=True)
CONFIGS = {
    "baseline": RunConfig(),
    "hybrid": _HYBRID,
    "hybrid_rerank": _RERANK,
    "hybrid_rerank_headers": _HEADERS,
    # k = 8 variants: only the best earlier row is run (decided on dev).
    "hybrid_k8": replace(_HYBRID, name="hybrid_k8", k=8),
    "hybrid_rerank_k8": replace(_RERANK, name="hybrid_rerank_k8", k=8),
    "hybrid_rerank_headers_k8": replace(_HEADERS, name="hybrid_rerank_headers_k8", k=8),
}


def run_id(config: RunConfig, split: str, golden_sha: str, chunks_sha: str) -> str:
    """Same config, split, golden set and corpus => same id => the run resumes."""
    fields = {
        key: value
        for key, value in asdict(config).items()
        if key not in RETRIEVAL_DEFAULTS or value != RETRIEVAL_DEFAULTS[key]
    }
    payload = {**fields, "split": split, "golden": golden_sha, "chunks": chunks_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def build_retriever(config: RunConfig, chunks: list[dict], embedder, reranker=None):
    """The retriever a config describes. The plain dense config uses the Phase 2 Retriever
    unchanged; anything else uses the hybrid pipeline."""
    if (config.retriever, config.rerank_model, config.headers) == ("dense", "", False):
        return Retriever(embedder, chunks)
    if config.rerank_model and reranker is None:
        reranker = FastEmbedReranker(config.rerank_model)
    return HybridRetriever(
        embedder,
        chunks,
        use_bm25=config.retriever == "hybrid",
        headers=config.headers,
        reranker=reranker if config.rerank_model else None,
        rerank_n=config.rerank_n,
        fuse_k=config.fuse_k,
    )


def _usage(completions: list[Completion]) -> dict:
    return {
        "calls": len(completions),
        "cached": sum(c.cached for c in completions),
        "prompt_tokens": sum(c.usage.get("prompt_tokens", 0) for c in completions),
        "completion_tokens": sum(c.usage.get("completion_tokens", 0) for c in completions),
        "reasoning_tokens": sum(c.usage.get("reasoning_tokens", 0) for c in completions),
    }


def _cost(model: str, usage: dict) -> float:
    return settings.cost_usd(model, usage["prompt_tokens"], usage["completion_tokens"])


def evaluate_item(
    item: dict, retriever: Retriever, llm: LLM, mapper: EvidenceMapper, config: RunConfig
) -> dict:
    hits, retrieval_s = retriever.retrieve(item["question"], config.recall_k)
    context = hits[: config.k]
    result = answer_from_hits(item["question"], context, retrieval_s, llm, model=config.generator)
    evidence_map = []
    for e in item["evidence"]:
        mapped = mapper.map(Evidence(e["doc_id"], e["page"], e["quote"]))
        evidence_map.append({**e, "chunk_ids": list(mapped.chunk_ids), "status": mapped.status})

    verdict, judge_completions, judge_error = None, [], ""
    if not result.abstained:
        try:
            v, judge_completions = judge(
                item["question"],
                context,
                result.text,
                item["reference_answer"],
                llm,
                version=config.judge_prompt.split(":")[0],
                model=config.judge,
            )
            verdict = asdict(v)
        except JudgeFailed as exc:
            judge_completions, judge_error = exc.completions, str(exc)

    gen_usage = _usage([result.completion])
    judge_usage = _usage(judge_completions)
    extra = {}
    if getattr(retriever, "last_stats", None):  # e.g. reranker inputs over the max length
        extra["retrieval_stats"] = dict(retriever.last_stats)
    return extra | {
        "id": item["id"],
        "split": item["split"],
        "topic": item["topic"],
        "type": item["type"],
        "answerable": item["answerable"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "retrieved_ids": [h.chunk["id"] for h in hits],
        "retrieved_scores": [round(h.score, 4) for h in hits],
        "context_ids": [h.chunk["id"] for h in context],
        "answer": result.text,
        "abstained": result.abstained,
        "cited_ids": result.cited_ids,
        "finish_reason": result.completion.finish_reason,
        "evidence_map": evidence_map,
        "gold_chunks": sorted({c for e in evidence_map for c in e["chunk_ids"]}),
        "judge": verdict,
        "judge_failed": bool(judge_error),
        "judge_error": judge_error,
        "latency": {
            "retrieval_s": round(retrieval_s, 4),
            "generation_s": result.completion.latency_s,
            "end_to_end_s": round(retrieval_s + result.completion.latency_s, 4),
            "wait_s": result.completion.wait_s,
        },
        "generation_usage": gen_usage,
        "judge_usage": judge_usage,
        "cost_usd": {
            "generation": round(_cost(config.generator, gen_usage), 8),
            "judge": round(_cost(config.judge, judge_usage), 8),
        },
    }


def load_records(path: Path) -> list[dict]:
    """Records from an items file. A torn final line (crash mid-write) is cut off so the next
    append starts on a clean line; that item is simply evaluated again."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        atomic_write_bytes(path, raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(
    config: RunConfig,
    split: str,
    items: list[dict],
    retriever: Retriever,
    llm: LLM,
    mapper: EvidenceMapper,
    run_dir: Path,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Evaluate every item of `split` not already in run_dir/items.jsonl; return all records.

    QuotaExhausted propagates after the finished items are saved, so re-running resumes."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "items.jsonl"
    with exclusive_lock(run_dir / ".lock"):
        with atomic_write(run_dir / "run.json") as f:
            json.dump({"config": asdict(config), "split": split}, f, indent=2)
        records = load_records(records_path)
        done = {r["id"] for r in records}
        todo = sorted(
            (i for i in items if i["split"] == split and i["id"] not in done), key=lambda i: i["id"]
        )
        total = sum(1 for i in items if i["split"] == split)
        log(f"{len(done)} of {total} {split} items already done; {len(todo)} to run")
        for n, item in enumerate(todo, len(done) + 1):
            record = evaluate_item(item, retriever, llm, mapper, config)
            _append(records_path, record)
            records.append(record)
            outcome = (
                "abstained"
                if record["abstained"]
                else (
                    "judge failed"
                    if record["judge_failed"]
                    else (record["judge"] or {}).get("correctness", "-")
                )
            )
            log(f"  [{n}/{total}] {item['id']} {item['type']:<12} {outcome}")
    return sorted(records, key=lambda r: r["id"])
