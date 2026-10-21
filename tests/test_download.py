import hashlib

import pytest

from medquery.ingest.download import ChecksumError, download, verify
from medquery.ingest.manifest import Document

CONTENT = b"%PDF-1.4 test file"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


def doc(sha256: str = "") -> Document:
    return Document(
        id="x",
        title="X",
        publisher="P",
        year=2020,
        topic="t",
        url="https://invalid.example/x.pdf",  # never fetched in these tests
        licence="Public domain",
        licence_evidence="e",
        as_of="2020",
        sha256=sha256,
    )


def test_verify_returns_digest_when_unpinned(tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(CONTENT)
    assert verify(path, "") == DIGEST


def test_verify_raises_on_mismatch(tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(CONTENT)
    with pytest.raises(ChecksumError):
        verify(path, "0" * 64)


def test_existing_file_is_not_refetched_and_is_verified(tmp_path):
    (tmp_path / "x.pdf").write_bytes(CONTENT)
    path, digest = download(doc(DIGEST), tmp_path)
    assert path == tmp_path / "x.pdf"
    assert digest == DIGEST


def test_existing_file_with_wrong_hash_is_rejected(tmp_path):
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4 tampered")
    with pytest.raises(ChecksumError):
        download(doc(DIGEST), tmp_path)
