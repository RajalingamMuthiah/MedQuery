"""Load the corpus manifest and render docs/SOURCES.md from it."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED = (
    "id",
    "title",
    "publisher",
    "year",
    "topic",
    "url",
    "licence",
    "licence_evidence",
    "as_of",
)


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    publisher: str
    year: int
    topic: str
    url: str
    licence: str
    licence_evidence: str
    as_of: str
    superseded: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class Fixture:
    file: str
    doc_id: str
    page: int
    purpose: str


@dataclass(frozen=True)
class Manifest:
    documents: list[Document]
    fixtures: list[Fixture]

    def get(self, doc_id: str) -> Document:
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        raise KeyError(doc_id)


def load_manifest(path: Path) -> Manifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    documents = []
    for entry in data.get("documents", []):
        missing = [key for key in REQUIRED if not entry.get(key)]
        if missing:
            raise ValueError(f"manifest entry {entry.get('id', '?')} is missing {missing}")
        documents.append(Document(**entry))
    ids = [doc.id for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest has duplicate document ids")
    fixtures = [Fixture(**entry) for entry in data.get("fixtures", [])]
    for fixture in fixtures:
        if fixture.doc_id not in ids:
            raise ValueError(f"fixture {fixture.file} refers to unknown document {fixture.doc_id}")
    return Manifest(documents, fixtures)


def render_sources(manifest: Manifest) -> str:
    lines = [
        "# Corpus sources",
        "",
        "<!-- Generated from corpus/manifest.toml by `uv run python -m medquery.ingest sources`."
        " Do not edit by hand. -->",
        "",
        "Every document is public-domain, openly licensed, or synthetic. No real patient data.",
        "Raw PDFs are downloaded to the git-ignored `data/raw/` and are not committed.",
        "",
        "| ID | Document | Publisher | Year | Topic | As of | URL | Licence | Evidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in manifest.documents:
        lines.append(
            f"| `{d.id}` | {d.title} | {d.publisher} | {d.year} | {d.topic} | {d.as_of} "
            f"| {d.url} | {d.licence} | {d.licence_evidence} |"
        )
    superseded = [d for d in manifest.documents if d.superseded]
    if superseded:
        lines += ["", "## Superseded guidance", ""]
        lines += [f"- `{d.id}` (as of {d.as_of}): {d.superseded}" for d in superseded]
    if manifest.fixtures:
        lines += [
            "",
            "## Test fixtures",
            "",
            "Single pages cut from the documents above and committed under `tests/fixtures/`."
            " Fixtures contain no photos or third-party images.",
            "",
            "| File | From | Page | Purpose |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{f.file}` | `{f.doc_id}` | {f.page} | {f.purpose} |" for f in manifest.fixtures
        ]
    return "\n".join(lines) + "\n"
