"""Evaluation commands: python -m medquery.eval golden {generate [--dry-run] [--force], topup}."""

import argparse
import csv
import json
import math
import sys
from collections import Counter

from medquery import config, paths
from medquery.eval.compare import write_ablation
from medquery.eval.evidence import EvidenceMapper
from medquery.eval.freeze import FreezeError, freeze, load_golden, sha256_file, write_frozen
from medquery.eval.gitinfo import git_state
from medquery.eval.golden import (
    GOLDEN_DIR,
    append_review_csv,
    read_review_csv,
    write_review_csv,
)
from medquery.eval.golden_gen import (
    GEN_HASH,
    TOPUPS,
    VERIFY_HASH,
    generate,
    load_pages,
    plan_tasks,
    review_rows,
    to_items,
    topup,
)
from medquery.eval.judge import JUDGE_PROMPTS
from medquery.eval.judge import prompt_id as judge_prompt_id
from medquery.eval.judge_review import (
    MATCH_RATIO,
    N_ANSWERS,
    N_CLAIMS,
    REVIEW_VERSION,
    REVISIONS,
    LabelError,
    agreement_block,
    freeze_labels,
    judge_history,
    labelled_units,
    latest_revision,
    load_agreement,
    load_labels,
    measure_judge,
    measurement_path,
    probe_units,
    review_paths,
    sample,
    write_review,
)
from medquery.eval.judge_review import SEED as SAMPLE_SEED
from medquery.eval.judge_review import _key as judge_review_key
from medquery.eval.judge_review import review_rows as judge_review_rows
from medquery.eval.report import write_report
from medquery.eval.runner import CONFIGS, RUNS_DIR, build_retriever, load_records, run, run_id
from medquery.eval.usage import format_summary, summarize
from medquery.fileio import LockHeld, atomic_write, exclusive_lock
from medquery.ingest.manifest import load_manifest
from medquery.llm import LLMError, QuotaExhausted, get_client
from medquery.retrieval.dense import CHUNKS, Retriever, load_chunks
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


def cmd_golden_topup(batch: str) -> int:
    manifest = load_manifest(paths.MANIFEST)
    docs = {d.id: d for d in manifest.documents}
    candidates_path = GOLDEN_DIR / f"candidates_{VERSION}.jsonl"
    review_path = GOLDEN_DIR / f"review_{VERSION}.csv"
    existing = [
        json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    if any(c.get("batch") == batch for c in existing):
        print(f"{batch} is already in {candidates_path.name}; refusing to run it twice")
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
                batch=batch,
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
    new_rows = review_rows(new_items, docs)
    all_rows = append_review(rows, new_rows)  # id-collision check
    with atomic_write(candidates_path) as f:
        for candidate in existing + result.candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    append_review_csv(review_path, new_rows)  # existing rows' bytes are never rewritten

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
    prompts = sorted({c["gen_prompt"] for c in result.candidates})
    print(f"author prompts: {', '.join(prompts)}")
    checks = sorted(
        {
            c["verification"].get("version", f"unanswerable_check_v1:{VERIFY_HASH}")
            for c in result.candidates
            if "verification" in c
        }
    )
    if checks:
        print(f"unanswerability checks: {', '.join(checks)}")
    return 0


def cmd_golden_freeze(version: str, reviewer: str, exclude: set[str], check_only: bool) -> int:
    review_path = GOLDEN_DIR / f"review_{version}.csv"
    candidates_path = GOLDEN_DIR / f"candidates_{version}.jsonl"
    candidates = [
        json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    pages = {(p.doc_id, p.page): p.text for p in load_pages()}
    mapper = EvidenceMapper(load_chunks(include_quiz=True))
    try:
        frozen = freeze(review_path, version, pages, mapper, reviewer, exclude, candidates)
        paths_written = () if check_only else write_frozen(frozen)
    except FreezeError as exc:
        print(f"freeze refused: {exc}")
        return 1
    meta = frozen.meta
    review = meta["review"]
    print(
        f"review: {review['rows']} rows; reviewed {review['reviewed']} "
        f"({review['reviewed_share']:.0%}); "
        f"keep {review['keep']}, fix {review['fix']}, drop {review['drop']}; "
        f"excluded {review['excluded'] or 'none'}"
    )
    print(f"reviewer: {meta['reviewer']}")
    print(
        f"frozen items: {meta['items']} (unanswerable {meta['unanswerable']}, "
        f"{meta['unanswerable_share']:.0%}); splits {meta['by_split']}"
    )
    table = Counter((i["topic"], i["type"], i["split"]) for i in frozen.items)
    print(f"\n{'topic':<13} {'type':<13} {'dev':>4} {'test':>5} {'total':>6}")
    for topic, kind in sorted({(t, k) for t, k, _ in table}):
        dev, test = table[(topic, kind, "dev")], table[(topic, kind, "test")]
        print(f"{topic:<13} {kind:<13} {dev:>4} {test:>5} {dev + test:>6}")
    if check_only:
        print("\n--check: validation passed; nothing written")
    else:
        for path in paths_written:
            print(f"wrote {path.relative_to(paths.ROOT)}")
        print(f"sha256: {sha256_file(paths_written[0])}")
    return 0


def cmd_run(split: str, config_name: str) -> int:
    config = CONFIGS[config_name]
    try:
        items, golden_meta = load_golden(config.golden_version)
    except (FreezeError, FileNotFoundError) as exc:
        print(f"error: cannot load golden_{config.golden_version}: {exc}", file=sys.stderr)
        return 1
    git = git_state()
    if split == "test" and git["dirty"] is not False:
        print("refused: test-split runs are milestones and need a clean, committed git tree")
        return 1
    chunks_sha = sha256_file(CHUNKS)
    rid = run_id(config, split, golden_meta["sha256"], chunks_sha)
    run_dir = RUNS_DIR / rid
    print(f"run {rid}: config {config.name}, split {split}, golden {golden_meta['version']}")
    try:
        retriever = build_retriever(
            config, load_chunks(include_quiz=config.include_quiz), FastEmbedder(config.embed_model)
        )
        records = run(
            config,
            split,
            items,
            retriever,
            get_client(),
            EvidenceMapper(load_chunks(include_quiz=True)),
            run_dir,
        )
    except QuotaExhausted as exc:
        print(f"stopped: {exc}\nRe-run the same command later to resume (finished items are kept).")
        return 2
    except (LLMError, LockHeld) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    agreement = load_agreement(config.judge_prompt, config.judge)
    written = write_report(
        records,
        config,
        split,
        rid,
        golden_meta,
        chunks_sha,
        git,
        agreement,
        history=judge_history(),
    )
    for path in written:
        print(f"wrote {path.relative_to(paths.ROOT)}")
    return 0


FRESH_SAMPLE = {"claims": 25, "answers": 10, "min_unsupported": 5}  # Phase 3 re-validation


def cmd_judge_sample(run: str, version: str = "v1", exclude_labelled: str | None = None) -> int:
    records = load_records(RUNS_DIR / run / "items.jsonl")
    if not records:
        print(f"no records for run {run}")
        return 1
    review = review_paths(version)
    if review["csv"].exists() or review["labels"].exists():
        print(f"{review['csv'].name} already exists; refusing to overwrite a labelling file")
        return 1
    run_config = json.loads((RUNS_DIR / run / "run.json").read_text(encoding="utf-8"))
    judge_model = run_config["config"]["judge"]
    judge_prompt = run_config["config"]["judge_prompt"]
    exclude_claims: set[str] = set()
    exclude_answers: set[tuple] = set()
    if exclude_labelled:
        exclude_claims, exclude_answers = labelled_units(exclude_labelled)
    if version == "v1":
        n_claims, n_answers, min_unsupported = N_CLAIMS, N_ANSWERS, math.ceil(N_CLAIMS / 3)
    else:
        n_claims, n_answers = FRESH_SAMPLE["claims"], FRESH_SAMPLE["answers"]
        min_unsupported = FRESH_SAMPLE["min_unsupported"]
    chunks = {c["id"]: c for c in load_chunks(include_quiz=True)}
    natural_unsupported = sum(
        not c["supported_by_context"] and judge_review_key(c["text"]) not in exclude_claims
        for r in records
        if r["judge"]
        for c in r["judge"]["claims"]
    )
    need = max(0, min_unsupported - natural_unsupported)
    print(
        f"natural judge-unsupported claims available: {natural_unsupported}; probes needed: {need}"
    )
    try:
        probes, probe_calls = (
            probe_units(
                records,
                get_client(),
                chunks,
                need,
                judge_version=judge_prompt.split(":")[0],
                judge_model=judge_model,
                exclude_claims=exclude_claims,
            )
            if need
            else ([], [])
        )
    except QuotaExhausted as exc:
        print(f"stopped: {exc}\nRe-run later; finished probe calls are cached.")
        return 2
    units = sample(
        records,
        n_claims,
        n_answers,
        probes=probes,
        exclude_claims=exclude_claims,
        exclude_answers=exclude_answers,
    )
    rows, key = judge_review_rows(units, {cid: c["text"] for cid, c in chunks.items()})
    meta = {
        "version": version,
        "run_id": run,
        "split": run_config["split"],
        "config": run_config["config"]["name"],
        "judge_model": judge_model,
        "judge_prompt": judge_prompt,
        "excluded_labelled_in": exclude_labelled,
        "excluded": {"claims": len(exclude_claims), "answers": len(exclude_answers)},
        "seed": SAMPLE_SEED,
        "probes": {
            "method": "mismatched context: a dev answer re-judged against the passages "
            "retrieved for a dev question from a different topic",
            "probe_answers": len({p["record"]["id"] for p in probes}),
            "probe_claims": len(probes),
            "probe_claims_in_sample": sum(1 for k in key.values() if k["probe"]),
            "judge_calls": len(probe_calls),
        },
    }
    write_review(rows, key, meta, review)
    by_unit = Counter(
        (k["unit"], k["judge_label"], "probe" if k["probe"] else "natural") for k in key.values()
    )
    print(
        f"wrote {review['csv'].relative_to(paths.ROOT)} ({len(rows)} rows; judge verdicts hidden "
        f"in {review['key'].name})"
    )
    for (unit, label, source), n in sorted(by_unit.items()):
        print(f"  {unit:<7} judge said {label:<12} {source:<8} {n}")
    print(f"probes: {meta['probes']}")
    print("\n".join(format_summary(summarize(probe_calls))))
    return 0


def cmd_ablation(split: str, config_names: list[str]) -> int:
    """Ablation report over completed runs of the given configs (dev only, clean tree)."""
    if split != "dev":
        print("refused: Phase 3 ablations use the dev split only")
        return 1
    git = git_state()
    if git["dirty"] is not False:
        print("refused: ablation reports need a clean, committed git tree")
        return 1
    items, golden_meta = load_golden("v1")
    chunks_sha = sha256_file(CHUNKS)
    total = sum(1 for i in items if i["split"] == split)
    runs = []
    for name in config_names:
        cfg = CONFIGS[name]
        rid = run_id(cfg, split, golden_meta["sha256"], chunks_sha)
        records = sorted(load_records(RUNS_DIR / rid / "items.jsonl"), key=lambda r: r["id"])
        if len(records) != total:
            print(f"run {rid} ({name}) has {len(records)} of {total} items; finish it first")
            return 1
        runs.append((cfg, rid, records))
    for path in write_ablation(runs, split, git, golden_meta, chunks_sha):
        print(f"wrote {path.relative_to(paths.ROOT)}")
    return 0


def _fmt_ci(t) -> str:
    if t[0] is None:
        return "n/a"
    return f"{t[0]:.3f}" + (f" [{t[1]:.3f}, {t[2]:.3f}]" if t[1] is not None else "")


def print_agreement(block: dict, labels: list[dict], judge_name: str) -> None:
    print(f"\n{judge_name}:")
    print(f"{'group':<14} {'n':>3} {'probe':>5}  {'agreement [95% CI]':<26} {'kappa [95% CI]':<26}")
    for group in ("claim", "claim_natural", "claim_probe", "answer"):
        b = block[group]
        print(
            f"{group:<14} {b['n']:>3} {b['n_probe']:>5}  {_fmt_ci(b['agreement']):<26} "
            f"{_fmt_ci(b['kappa']):<26}"
        )
        print(f"{'':<14} human {b['human_class_counts']}  judge {b['judge_class_counts']}")
    caught = block["natural_unsupported_caught"]
    rejected = block["natural_supported_rejected"]
    print(
        f"natural unsupported caught: {caught['judge_caught']} of {caught['human_unsupported']}; "
        f"natural supported wrongly rejected: {rejected['judge_rejected']} of "
        f"{rejected['human_supported']} (matched units; unmatched: "
        f"{block['unmatched_units'] or 'none'})"
    )
    disagreements = [
        x for x in labels if x["judge_label"] is not None and x["human_label"] != x["judge_label"]
    ]
    print(f"{len(disagreements)} disagreement(s):")
    for x in disagreements:
        source = f"probe (passages from {x['donor_id']})" if x["probe"] else "natural"
        print(
            f"  {x['row_id']} {x['unit']:<6} {x['item_id']} {source}: "
            f"you={x['human_label']}, judge={x['judge_label']}"
        )
        print(f"      note: {x['notes'] or '(none)'}")


def cmd_judge_review_freeze(labeller: str, revision: str, version: str = REVIEW_VERSION) -> int:
    try:
        meta = freeze_labels(labeller, version=version, revision=revision)
    except LabelError as exc:
        print(f"freeze refused: {exc}")
        return 1
    review = review_paths(version, revision=revision)
    labels = [
        json.loads(line) for line in review["labels"].read_text(encoding="utf-8").splitlines()
    ]
    print(
        f"froze {meta['labels']} labels ({meta['revision']}) by {meta['labeller']} "
        f"-> {review['labels'].name}"
    )
    print(f"labels sha256 {meta['sha256']}; judge {meta['judge_model']} {meta['judge_prompt']}")
    if meta.get("supersedes"):
        print(
            f"supersedes {meta['supersedes']['file']} ({meta['supersedes']['sha256'][:12]}); "
            f"changed rows: {meta['changed_rows']}"
        )
    block = agreement_block(labels, meta["judge_prompt"], meta["judge_model"], meta)
    print_agreement(block, labels, meta["judge_prompt"])
    return 0


def cmd_judge_review_measure(
    judge_version: str, model: str, seed: int, version: str = REVIEW_VERSION
) -> int:
    revision = latest_revision(version)
    if revision is None:
        print("no frozen labels")
        return 1
    labels, meta = load_labels(version, GOLDEN_DIR, revision)
    review = review_paths(version, revision=revision)
    with review["csv"].open(encoding="utf-8-sig", newline="") as f:
        claim_text = {r["row_id"]: r["claim"] for r in csv.DictReader(f)}
    records = {r["id"]: r for r in load_records(RUNS_DIR / meta["run_id"] / "items.jsonl")}
    chunks = {c["id"]: c for c in load_chunks(include_quiz=True)}
    try:
        verdicts, unmatched, calls = measure_judge(
            judge_version, model, labels, claim_text, records, chunks, get_client(), seed=seed
        )
    except QuotaExhausted as exc:
        print(f"stopped: {exc}\nRe-run later; finished judge calls are cached.")
        return 2
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    data = {
        "judge_prompt": judge_prompt_id(judge_version),
        "judge_model": model,
        "seed": seed,
        "labels_revision": meta.get("revision", "v1"),
        "labels_sha256": meta["sha256"],
        "row_labels": verdicts,
        "unmatched": unmatched,
        "match_ratio": MATCH_RATIO,
        "usage": summarize(calls),
    }
    path = measurement_path(version, GOLDEN_DIR, judge_version, revision, model, seed)
    with atomic_write(path) as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {path.relative_to(paths.ROOT)} (labels {meta.get('revision', 'v1')}, "
        f"sha256 {meta['sha256'][:12]}); unmatched units: {unmatched or 'none'}"
    )
    print("\n".join(format_summary(summarize(calls))))
    measured = [{**x, "judge_label": verdicts.get(x["row_id"])} for x in labels]
    print_agreement(
        agreement_block(
            measured, data["judge_prompt"], model, {**meta, "judge_prompt": data["judge_prompt"]}
        ),
        measured,
        f"{model} {data['judge_prompt']} seed {seed} (re-judged on the same frozen labels)",
    )
    return 0


def cmd_judge_review_stability(judge_version: str, model: str, seeds: list[int]) -> int:
    """Verdict flips between seeds for the same judge on the same frozen labels."""
    revision = latest_revision()
    runs = []
    for seed in seeds:
        path = measurement_path(REVIEW_VERSION, GOLDEN_DIR, judge_version, revision, model, seed)
        if not path.exists():
            print(f"missing {path.name}: run `judge-review measure --seed {seed}` first")
            return 1
        runs.append(json.loads(path.read_text(encoding="utf-8"))["row_labels"])
    labels, _ = load_labels(REVIEW_VERSION, GOLDEN_DIR, revision)
    print(f"{model} {judge_version}: verdict stability across seeds {seeds} (labels {revision})")
    for unit in ("claim", "answer"):
        rows = [x["row_id"] for x in labels if x["unit"] == unit]
        both = [r for r in rows if all(run.get(r) is not None for run in runs)]
        flips = [r for r in both if len({run[r] for run in runs}) > 1]
        print(
            f"  {unit:<6} {len(flips)} of {len(both)} verdicts changed"
            f"{(' (' + ', '.join(flips) + ')') if flips else ''}"
            f"; {len(rows) - len(both)} unmatched in some seed"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when output is redirected
    parser = argparse.ArgumentParser(prog="python -m medquery.eval")
    sub = parser.add_subparsers(dest="group", required=True)
    golden = sub.add_parser("golden").add_subparsers(dest="action", required=True)
    gen = golden.add_parser("generate", help="write golden candidates and the review CSV")
    gen.add_argument("--dry-run", action="store_true", help="show sampled pages; no LLM calls")
    gen.add_argument("--force", action="store_true", help="overwrite a review CSV with edits")
    top = golden.add_parser("topup", help="append a top-up batch to the review CSV")
    top.add_argument("--batch", choices=sorted(TOPUPS), required=True)
    frz = golden.add_parser("freeze", help="validate the review CSV and freeze golden_<version>")
    frz.add_argument("--version", required=True)
    frz.add_argument("--reviewer", required=True, help="who reviewed the set, recorded in meta")
    frz.add_argument("--exclude", default="", help="comma-separated ids left out of the set")
    frz.add_argument("--check", action="store_true", help="validate and report only")
    run_parser = sub.add_parser("run", help="evaluate a config on a golden split (resumable)")
    run_parser.add_argument("--split", choices=["dev", "test"], required=True)
    run_parser.add_argument("--config", choices=sorted(CONFIGS), default="baseline")
    sample_parser = sub.add_parser("judge-sample", help="write the judge-review CSV from a run")
    sample_parser.add_argument("--run", required=True, help="run id (data/eval_runs/<id>)")
    sample_parser.add_argument(
        "--version", default=REVIEW_VERSION, help="review version (v2: fresh)"
    )
    sample_parser.add_argument(
        "--exclude-labelled", default=None, help="leave out units labelled in this review version"
    )
    abl = sub.add_parser("ablation", help="paired comparison of completed dev runs")
    abl.add_argument("--split", choices=["dev", "test"], default="dev")
    abl.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), required=True)
    jr = sub.add_parser("judge-review").add_subparsers(dest="action", required=True)
    jr_freeze = jr.add_parser("freeze", help="freeze hand labels and report agreement")
    jr_freeze.add_argument("--labeller", required=True, help="who labelled, recorded in meta")
    jr_freeze.add_argument("--version", default=REVIEW_VERSION)
    jr_freeze.add_argument(
        "--revision",
        default="",
        choices=list(REVISIONS),
        help="freeze a corrected label set (r2, ...) next to earlier ones",
    )
    jr_measure = jr.add_parser("measure", help="re-judge the frozen labels with a judge version")
    jr_measure.add_argument("--judge", required=True, choices=sorted(JUDGE_PROMPTS))
    jr_measure.add_argument("--model", default=config.JUDGE_MODEL)
    jr_measure.add_argument("--seed", type=int, default=0)
    jr_measure.add_argument("--version", default=REVIEW_VERSION)
    jr_stab = jr.add_parser("stability", help="verdict flips across seeds on the frozen labels")
    jr_stab.add_argument("--judge", required=True, choices=sorted(JUDGE_PROMPTS))
    jr_stab.add_argument("--model", default=config.JUDGE_MODEL)
    jr_stab.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args(argv)
    if args.group == "judge-review" and args.action == "freeze":
        return cmd_judge_review_freeze(args.labeller, args.revision, args.version)
    if args.group == "judge-review" and args.action == "measure":
        return cmd_judge_review_measure(args.judge, args.model, args.seed, args.version)
    if args.group == "judge-review" and args.action == "stability":
        return cmd_judge_review_stability(args.judge, args.model, args.seeds)
    if args.group == "run":
        return cmd_run(args.split, args.config)
    if args.group == "judge-sample":
        return cmd_judge_sample(args.run, args.version, args.exclude_labelled)
    if args.group == "ablation":
        return cmd_ablation(args.split, args.configs)
    if args.group == "golden" and args.action == "freeze":
        excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
        return cmd_golden_freeze(args.version, args.reviewer, excluded, args.check)
    if args.group == "golden" and args.action == "generate":
        return cmd_golden_generate(args.dry_run, args.force)
    if args.group == "golden" and args.action == "topup":
        return cmd_golden_topup(args.batch)
    return 1


if __name__ == "__main__":
    sys.exit(main())
