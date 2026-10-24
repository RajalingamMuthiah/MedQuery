"""Project paths, resolved from the package location so commands work from any cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.toml"
SOURCES_MD = ROOT / "docs" / "SOURCES.md"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def corpus_dir(corpus: str = "v1") -> Path:
    """Corpus v1 lives in data/processed/ (as built since Phase 1); later versions in
    data/processed/<version>/, so building one never touches another."""
    return DATA_PROCESSED if corpus == "v1" else DATA_PROCESSED / corpus


FIXTURES = ROOT / "tests" / "fixtures"
