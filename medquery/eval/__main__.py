"""Evaluation commands: python -m medquery.eval golden generate [--dry-run] [--force]."""

import argparse
import json
import sys
from collections import Counter

from medquery import config, paths
from medquery.eval.golden import GOLDEN_DIR, read_review_csv, write_review_csv
from medquery.eval.golden_gen import (
    GEN_HASH,
    VERIFY_HASH,
    generate,
    load_pages,
    plan_tasks,
    review_rows,
    to_items,
)
from medquery.eval.usage import format_summary, summarize
from medquery.fileio import LockHeld, atomic_write, exclusive_lock
from medquery.ingest.manifest import load_manifest
from medquery.llm import LLMError, QuotaExhausted, get_client
from medquery.retrieval.dense import Retriever, load_chunks
from medquery.retrieval.embed import FastEmbedder

VERSION = "v1"
LOCK = paths.ROOT / "data" / "locks" / "golden_generate.lock"


def review_has_edits(path) -> bool:
    if not path.exists():
        return False
    return any(
        row.get("reviewed", "").strip()
        or row.get("notes", "").strip()
        or row.get("decision") != "keep"
        for row in read_review_csv(path)
    )


def cmd_golden_generate(dry_run: bool, force: bool) -> int:
    manifest = load_manifest(paths.MANIFEST)
    docs = {d.id: d for d in manifest.documents}
    pages, all_chunks = load_pages(), load_chunks(include_quiz=True)
    review_path = GOLDEN_DIR / f"review_{VERSION}.csv"

    if dry_run:
        tasks = plan_tasks(pages, all_chunks, docs)
        for t in tasks:
            where = " + ".join(f"{p.doc_id} p.{p.page}" for p in t.pages)
            print(f"{t.topic:<13} {t.type:<13} x{t.n}  {where}")
        questions = Counter()
        for t in tasks:
            questions[(t.topic, t.type)] += t.n
        print(
            f"\n{len(tasks)} calls, {sum(questions.values())} candidate questions before validation"
        )
        return 0

    if review_has_edits(review_path) and not force:
        print(f"{review_path.name} already has review edits; refusing to overwrite (use --force)")
        return 1
    try:
        retriever = Retriever(FastEmbedder(), load_chunks())
        with exclusive_lock(LOCK):
            result = generate(manifest, pages, all_chunks, retriever, get_client())
    except QuotaExhausted as exc:
        print(
            f"stopped: {exc}\nRe-run the same command later to resume (finished calls are cached)."
        )
        return 2
    except (LLMError, LockHeld) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with atomic_write(GOLDEN_DIR / f"candidates_{VERSION}.jsonl") as f:
        for candidate in result.candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    items = to_items(result.candidates)
    write_review_csv(review_path, review_rows(items, docs))

    status = Counter(c["status"] for c in result.candidates)
    print(f"\ncandidates: {len(result.candidates)} ({dict(status)})")
    print("accepted by topic and type:")
    for (topic, kind), n in sorted(Counter((i.topic, i.type) for i in items).items()):
        print(f"  {topic:<13} {kind:<13} {n}")
    unanswerable = sum(1 for i in items if not i.answerable)
    share = unanswerable / max(len(items), 1)
    print(f"  total {len(items)}, unanswerable {unanswerable} ({share:.0%})")
    print("rejected by reason:")
    reasons = Counter(
        c["reject_reason"].split(":")[0] for c in result.candidates if c["reject_reason"]
    )
    for reason, n in reasons.most_common():
        print(f"  {n:>3}  {reason}")
    print(f"tokens and list cost (prices {config.PRICES_DATE}):")
    print("\n".join(format_summary(summarize(result.completions))))
    print(f"prompts: golden_gen_v1:{GEN_HASH}, unanswerable_check_v1:{VERIFY_HASH}")
    print(f"\nwrote {review_path.relative_to(paths.ROOT)} for hand review")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when output is redirected
    parser = argparse.ArgumentParser(prog="python -m medquery.eval")
    sub = parser.add_subparsers(dest="group", required=True)
    golden = sub.add_parser("golden").add_subparsers(dest="action", required=True)
    gen = golden.add_parser("generate", help="write golden candidates and the review CSV")
    gen.add_argument("--dry-run", action="store_true", help="show sampled pages; no LLM calls")
    gen.add_argument("--force", action="store_true", help="overwrite a review CSV with edits")
    args = parser.parse_args(argv)
    if args.group == "golden" and args.action == "generate":
        return cmd_golden_generate(args.dry_run, args.force)
    return 1


if __name__ == "__main__":
    sys.exit(main())
