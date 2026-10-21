"""Committed fixtures must be single pages with no photos or third-party images."""

import pdfplumber
import pytest

from medquery import paths
from medquery.ingest.manifest import load_manifest
from tests.conftest import BOOKTABS_FIXTURE, SCAN_FIXTURE, TABLE_FIXTURE


def test_every_fixture_file_is_listed_in_the_manifest(fixtures_dir):
    listed = {f.file for f in load_manifest(paths.MANIFEST).fixtures}
    on_disk = {p.name for p in fixtures_dir.glob("*.pdf")}
    assert on_disk == listed


def test_fixtures_are_single_pages(fixtures_dir):
    for pdf_path in fixtures_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            assert len(pdf.pages) == 1, pdf_path.name


@pytest.mark.parametrize("fixture", [TABLE_FIXTURE, BOOKTABS_FIXTURE])
def test_table_fixtures_have_no_images(fixture):
    with pdfplumber.open(fixture) as pdf:
        assert pdf.pages[0].images == []


def test_scan_fixture_has_only_the_scan_bitmap():
    with pdfplumber.open(SCAN_FIXTURE) as pdf:
        page = pdf.pages[0]
        assert len(page.images) == 1
        image = page.images[0]
        coverage = (image["width"] * image["height"]) / (page.width * page.height)
        assert coverage > 0.9  # one full-page scan, not an embedded photo
