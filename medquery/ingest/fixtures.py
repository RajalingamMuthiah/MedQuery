"""Cut single-page test fixtures out of verified corpus PDFs."""

from pathlib import Path

import pypdfium2 as pdfium

from medquery.ingest.download import verify
from medquery.ingest.manifest import Manifest


def cut_page(src: Path, page: int, dest: Path) -> None:
    """Copy one 1-based page of `src` into a new single-page PDF at `dest`."""
    source = pdfium.PdfDocument(src)
    out = pdfium.PdfDocument.new()
    out.import_pages(source, [page - 1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    out.close()
    source.close()


def cut_fixtures(manifest: Manifest, raw_dir: Path, fixtures_dir: Path) -> list[Path]:
    written = []
    for fixture in manifest.fixtures:
        doc = manifest.get(fixture.doc_id)
        src = raw_dir / f"{doc.id}.pdf"
        verify(src, doc.sha256)
        dest = fixtures_dir / fixture.file
        cut_page(src, fixture.page, dest)
        written.append(dest)
    return written
