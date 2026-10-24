"""Corpus commands: python -m medquery.ingest {download,build,fixtures,sources}."""

import argparse
import sys

from medquery import paths
from medquery.fileio import LockHeld
from medquery.ingest.build import build
from medquery.ingest.download import download
from medquery.ingest.fixtures import cut_fixtures
from medquery.ingest.manifest import load_manifest, render_sources


def cmd_download() -> int:
    manifest = load_manifest(paths.MANIFEST)
    unpinned, failed = [], []
    for doc in manifest.documents:
        try:
            path, digest = download(doc, paths.DATA_RAW)
        except Exception as exc:  # report every failure, not just the first
            failed.append(doc.id)
            print(f"FAIL {doc.id:<22} {exc}")
            continue
        print(f"ok   {doc.id:<22} {path.stat().st_size / 1e6:6.1f} MB  sha256={digest}")
        if not doc.sha256:
            unpinned.append((doc.id, digest))
    if unpinned:
        print("\nThese documents have no pinned sha256. Add them to corpus/manifest.toml:")
        for doc_id, digest in unpinned:
            print(f'  {doc_id}: sha256 = "{digest}"')
    return 1 if failed else 0


def cmd_build(corpus: str = "v1") -> int:
    """Corpus v1 -> data/processed/; v2 (table fix) -> data/processed/v2/. v1 is never
    rewritten by a v2 build."""
    out_dir = paths.corpus_dir(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = build(load_manifest(paths.MANIFEST), paths.DATA_RAW, out_dir, table_format=corpus)
    except LockHeld as exc:
        print(f"build not started: {exc}", file=sys.stderr)
        return 1
    header = (
        f"{'document':<22} {'pages':>5} {'text':>5} {'ocr':>4} {'tables':>6} "
        f"{'chunks':>6} {'ocr conf':>8} {'secs':>6}"
    )
    print(header)
    print("-" * len(header))
    for d in summary["documents"] + [{"doc_id": "TOTAL", "seconds": "", **summary["totals"]}]:
        conf = d["ocr_mean_confidence"]
        print(
            f"{d['doc_id']:<22} {d['pages']:>5} {d['text_pages']:>5} {d['ocr_pages']:>4} "
            f"{d['tables']:>6} {d['chunks']:>6} {'-' if conf is None else conf:>8} "
            f"{d['seconds']:>6}"
        )
    totals = summary["totals"]
    print(
        f"\nshort text chunks (<30 words): {totals['short_text_chunks']} "
        f"({totals['short_text_share']:.1%} of text chunks)"
    )
    print(f"quiz chunks (kind=quiz, excluded from retrieval by default): {totals['quiz_chunks']}")
    print(f"dropped junk chunks: {sum(totals['dropped_junk'].values())}")
    for reason, count in sorted(totals["dropped_junk"].items()):
        print(f"  {count:>4}  {reason}")
    rel = paths.DATA_PROCESSED.relative_to(paths.ROOT)
    print(f"\nwrote {rel / 'chunks.jsonl'}, {rel / 'dropped.jsonl'}, {rel / 'build_summary.json'}")
    print("ocr conf = mean Tesseract word confidence (0-100) over OCR pages")
    return 0


def cmd_fixtures() -> int:
    for path in cut_fixtures(load_manifest(paths.MANIFEST), paths.DATA_RAW, paths.FIXTURES):
        print(f"wrote {path.relative_to(paths.ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


def cmd_sources() -> int:
    manifest = load_manifest(paths.MANIFEST)
    paths.SOURCES_MD.write_text(render_sources(manifest), encoding="utf-8", newline="\n")
    print(f"wrote {paths.SOURCES_MD.relative_to(paths.ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when output is redirected
    parser = argparse.ArgumentParser(prog="python -m medquery.ingest")
    commands = {
        "download": cmd_download,
        "build": cmd_build,
        "fixtures": cmd_fixtures,
        "sources": cmd_sources,
    }
    parser.add_argument("command", choices=list(commands))
    parser.add_argument(
        "--corpus", choices=["v1", "v2"], default="v1", help="build: v2 = table fix (Phase 4)"
    )
    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args.corpus)
    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
