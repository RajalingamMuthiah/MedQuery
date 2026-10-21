"""Scanned pages: OCR fallback on a real image-only page, plus a ground-truth round trip."""

import difflib
import textwrap

import pdfplumber
import pytest
from PIL import Image, ImageDraw, ImageFont

from medquery.ingest.chunk import chunk_document
from medquery.ingest.ocr import needs_ocr
from medquery.ingest.pdf import extract_pages
from tests.conftest import SCAN_FIXTURE, TABLE_FIXTURE


@pytest.fixture(scope="module")
def scan_page():
    [page] = extract_pages(SCAN_FIXTURE)
    return page


def test_scan_has_no_text_layer_and_needs_ocr():
    with pdfplumber.open(SCAN_FIXTURE) as pdf:
        assert pdf.pages[0].chars == []
        assert needs_ocr(pdf.pages[0])


def test_text_page_does_not_need_ocr():
    with pdfplumber.open(TABLE_FIXTURE) as pdf:
        assert not needs_ocr(pdf.pages[0])


def test_scan_is_ocrd_with_confidence(scan_page):
    assert scan_page.source == "ocr"
    assert scan_page.ocr_words > 300
    assert scan_page.ocr_confidence is not None and scan_page.ocr_confidence > 70


# Phrases read by hand from the scanned page image (tb-reported-1964, page 5).
HAND_CHECKED = [
    "trends in tuberculosis",
    "the states reported 50,874 new",
    "54,042",
    "seventeen states and the district of columbia",
    "new york state",
    "public health service",
]


@pytest.mark.parametrize("phrase", HAND_CHECKED)
def test_ocr_text_contains_hand_checked_phrases(scan_page, phrase):
    text = " ".join(line.text for box in scan_page.boxes for line in box.lines).lower()
    assert phrase in text


def test_scan_yields_ocr_text_chunks(scan_page):
    chunks = chunk_document("scan", [scan_page])
    assert chunks
    for chunk in chunks:
        assert chunk.kind == "text"
        assert chunk.source == "ocr"
        assert chunk.ocr_confidence is not None
    joined = " ".join(c.text for c in chunks)
    assert "decreased markedly" in joined  # "de-" / "creased" rejoined across the line break


GROUND_TRUTH = (
    "Hypertension is a common condition in which the long-term force of the blood against "
    "the artery walls is high enough that it may eventually cause health problems. Blood "
    "pressure is determined both by the amount of blood the heart pumps and the amount of "
    "resistance to blood flow in the arteries. Regular measurement helps people and their "
    "health care teams decide on lifestyle changes such as eating less salt and being more "
    "physically active."
)


def test_ocr_round_trip_matches_ground_truth(tmp_path):
    """Render known text as an image-only PDF and check OCR recovers it."""
    width, height = 2550, 3300  # US Letter at 300 dpi
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    y = 300
    for line in textwrap.wrap(GROUND_TRUTH, width=70):
        draw.text((250, y), line, fill=0, font=font)
        y += 70
    pdf_path = tmp_path / "synthetic_scan.pdf"
    image.save(pdf_path, "PDF", resolution=300)

    [page] = extract_pages(pdf_path)
    assert page.source == "ocr"
    recovered = " ".join(line.text for box in page.boxes for line in box.lines)
    similarity = difflib.SequenceMatcher(None, recovered, GROUND_TRUTH).ratio()
    assert similarity >= 0.90, f"similarity {similarity:.3f}: {recovered!r}"
