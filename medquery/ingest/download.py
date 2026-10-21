"""Download corpus PDFs listed in the manifest and verify their sha256."""

import hashlib
import urllib.request
from pathlib import Path

from medquery.ingest.manifest import Document

# CDC's firewall rejects bare tool names; a contact URL identifies the client honestly.
USER_AGENT = "MedQuery/0.1 (+https://github.com/RajalingamMuthiah/MedQuery)"


class ChecksumError(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> str:
    """Return the file's sha256; raise ChecksumError if `expected` is set and differs."""
    actual = sha256_of(path)
    if expected and actual != expected:
        raise ChecksumError(f"{path.name}: expected sha256 {expected}, got {actual}")
    return actual


def download(doc: Document, dest_dir: Path) -> tuple[Path, str]:
    """Fetch `doc` into dest_dir/<id>.pdf unless already present. Returns (path, sha256)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{doc.id}.pdf"
    if not path.exists():
        part = path.with_suffix(".part")
        request = urllib.request.Request(doc.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
        if part.read_bytes()[:5] != b"%PDF-":
            part.unlink()
            raise ValueError(f"{doc.id}: {doc.url} did not return a PDF")
        part.replace(path)
    return path, verify(path, doc.sha256)
