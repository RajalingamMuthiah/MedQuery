"""Project paths, resolved from the package location so commands work from any cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.toml"
SOURCES_MD = ROOT / "docs" / "SOURCES.md"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIXTURES = ROOT / "tests" / "fixtures"
