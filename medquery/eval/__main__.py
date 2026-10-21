"""Evaluation commands: python -m medquery.eval golden {generate [--dry-run] [--force], topup}."""

import argparse
import json
import sys
from collections import Counter

from medquery import config, paths
from medquery.eval.golden import GOLDEN_DIR, read_review_csv, write_review_csv
from medquery.eval.golden_gen import (
    GEN_HASH,
    SUPERSEDED_V2_HASH,
    TOPUP_BATCH,
    VERIFY_HASH,
    generate,
    load_pages,
    plan_tasks,
    review_rows,
    to_items,
    topup,
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


def next_id_number(rows: list[dict]) -> int:
    return max((int(r["id"].lstrip("g")) for r in rows), default=0) + 1


def append_review(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Existing rows unchanged and in order, then the new rows. Ids must not collide."""
    taken = {r["id"] for r in existing_rows}
    clash = [r["id"] for r in new_rows if r["id"] in taken]
    if clash:
        raise ValueError(f"new ids collide with existing rows: {clash}")
    return existing_rows + new_rows


def cmd_golden_topup() -> int:
    manifest = load_manifest(paths.MANIFEST)
    docs = {d.id: d for d in manifest.documents}
    candidates_path = GOLDEN_DIR / f"candidates_{VERSION}.jsonl"
    review_path = GOLDEN_DIR / f"review_{VERSION}.csv"
    existing = [
        json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    if any(c.get("batch") == TOPUP_BATCH for c in existing):
        print(f"{TOPUP_BATCH} is already in {candidates_path.name}; refusing to run it twice")
        return 1
    rows = read_review_csv(review_path)
    try:
        retriever = Retriever(FastEmbedder(), load_chunks())
        with exclusive_lock(LOCK):
            result = topup(
                manifest,
                load_pages(),
                load_chunks(include_quiz=True),
                retriever,
                get_client(),
                existing,
                [r["question"] for r in rows],
            )
    except QuotaExhausted as exc:
        print(
            f"stopped: {exc}\nRe-run the same command later to resume (finished calls are cached)."
        )
        return 2
    except (LLMError, LockHeld) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    new_items = to_items(result.candidates, start=next_id_number(rows), sort=False)
    all_rows = append_review(rows, review_rows(new_items, docs))
    with atomic_write(candidates_path) as f:
        for candidate in existing + result.candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    write_review_csv(review_path, all_rows)

    status = Counter(c["status"] for c in result.candidates)
    print(f"\ntop-up candidates: {len(result.candidates)} ({dict(status)})")
    if new_items:
        print(f"new ids: {new_items[0].id}-{new_items[-1].id}")
    print("rejected by reason:")
    reasons = Counter(
        c["reject_reason"].split(":")[0] for c in result.candidates if c["reject_reason"]
    )
    for reason, n in reasons.most_common():
        print(f"  {n:>3}  {reason}")
    print("\nreview_v1.csv totals by topic and type:")
    for (topic, kind), n in sorted(Counter((r["topic"], r["type"]) for r in all_rows).items()):
        print(f"  {topic:<13} {kind:<13} {n}")
    unanswerable = sum(1 for r in all_rows if r["answerable"] == "no")
    print(
        f"  total {len(all_rows)}, unanswerable {unanswerable} ({unanswerable / len(all_rows):.0%})"
    )
    print(f"tokens and list cost (prices {config.PRICES_DATE}):")
    print("\n".join(format_summary(summarize(result.completions))))
    print(f"prompt: golden_gen_v1+superseded_v2:{SUPERSEDED_V2_HASH} (superseded top-up only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when output is redirected
    parser = argparse.ArgumentParser(prog="python -m medquery.eval")
    sub = parser.add_subparsers(dest="group", required=True)
    golden = sub.add_parser("golden").add_subparsers(dest="action", required=True)
    gen = golden.add_parser("generate", help="write golden candidates and the review CSV")
    gen.add_argument("--dry-run", action="store_true", help="show sampled pages; no LLM calls")
    gen.add_argument("--force", action="store_true", help="overwrite a review CSV with edits")
    golden.add_parser("topup", help="append top-up 1 candidates to the review CSV")
    args = parser.parse_args(argv)
    if args.group == "golden" and args.action == "generate":
        return cmd_golden_generate(args.dry_run, args.force)
    if args.group == "golden" and args.action == "topup":
        return cmd_golden_topup()
    return 1


if __name__ == "__main__":
    sys.exit(main())
