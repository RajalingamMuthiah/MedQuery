"""Prepare the live app's data at image build time (Dockerfile 'corpus' stage):
uv run python -m medquery.api.prepare

Runs the live config's loader once, so the embedding cache and the fastembed model are stored
under data/cache/, and reports whether the corpus v2 just built is the one that was evaluated
(Debian's Tesseract can OCR the 1964 scan slightly differently from the local build)."""

import sys

from medquery.api.app import default_loader, live_config

EVALUATED_CORPUS_V2_SHA256 = "07d7860fac9eae08e3183084722c535b25948e33552979a7161da1686fd90c67"


def sha_note(sha: str) -> str:
    if sha == EVALUATED_CORPUS_V2_SHA256:
        return f"corpus v2 sha256 {sha}: identical to the evaluated corpus"
    return (
        f"WARNING: corpus v2 sha256 {sha} differs from the evaluated corpus "
        f"({EVALUATED_CORPUS_V2_SHA256}); the app works, but its answers may differ slightly "
        "from the reported results (OCR differences in scanned pages)"
    )


def main() -> int:
    retriever, chunks, sha = default_loader(live_config())
    print(f"loaded {len(chunks)} chunks; embeddings and model cached under data/cache/")
    print(sha_note(sha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
