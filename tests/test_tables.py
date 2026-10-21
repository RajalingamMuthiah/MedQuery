"""Table pages.

JNC 7 Express p.19: an open-sided ruled table (no outer borders, no bottom rule).
MMWR LTBI 2020 p.8: "booktabs" tables ruled with horizontal lines only.
"""

import pytest

from medquery.ingest.chunk import chunk_document
from medquery.ingest.pdf import extract_pages
from tests.conftest import BOOKTABS_FIXTURE, TABLE_FIXTURE


@pytest.fixture(scope="module")
def page():
    [page] = extract_pages(TABLE_FIXTURE)
    return page


def test_page_uses_text_layer(page):
    assert page.source == "text"
    assert page.number == 1


def test_exactly_one_real_table_is_detected(page):
    # The boxed section heading below the table must not count as a one-row "table".
    assert len(page.tables) == 1


def test_header_row(page):
    assert page.tables[0].rows[0][:5] == [
        "BP Classification",
        "SBP* mmHg",
        "DBP* mmHg",
        "Lifestyle Modification",
        "Initial drug therapy",
    ]


def test_row_labels_and_last_row_are_recovered(page):
    labels = [row[0] for row in page.tables[0].rows]
    assert labels[-4:] == [
        "Normal",
        "Prehypertension",
        "Stage 1 Hypertension",
        "Stage 2 Hypertension",
    ]


def test_known_cell_values(page):
    rows = {row[0]: row for row in page.tables[0].rows}
    assert rows["Normal"][1:3] == ["<120", "and <80"]
    assert rows["Stage 2 Hypertension"][1:3] == ["≥160", "or ≥100"]
    # Merged cell spanning Normal and Prehypertension is filled down.
    assert rows["Prehypertension"][4] == "No antihypertensive drug indicated."
    # A word hyphenated across a line break inside a cell is rejoined.
    assert "compelling indications" in rows["Stage 1 Hypertension"][5]


def test_caption_is_attached(page):
    assert page.tables[0].caption.startswith(
        "Table 1. Classification and management of blood pressure"
    )


def test_table_text_does_not_leak_into_body(page):
    body = " ".join(line.text for box in page.boxes for line in box.lines)
    assert "Thiazide-type diuretics for most" not in body
    assert "No antihypertensive drug indicated" not in body
    assert "120–139" not in body
    # Body text around the table is kept.
    assert "In clinical trials, antihypertensive therapy" in body


@pytest.fixture(scope="module")
def booktabs_page():
    [page] = extract_pages(BOOKTABS_FIXTURE)
    return page


def test_booktabs_tables_are_detected_with_all_columns(booktabs_page):
    """MMWR tables have horizontal rules only; columns come from the segmented rules."""
    assert len(booktabs_page.tables) == 2
    regimens, dosages = booktabs_page.tables
    assert regimens.caption.startswith("TABLE 3. Recommendations for regimens")
    assert dosages.caption.startswith("TABLE 4. Dosages")
    assert dosages.rows[0] == ["Drug", "Duration", "Dose and age group", "Frequency", "Total doses"]
    assert dosages.rows[1] == [
        "Isoniazid* and rifapentine†",
        "3 mos",
        "Adults and children aged ≥12 yrs",
        "Once weekly",
        "12",
    ]
    assert ["Preferred", "4 mos rifampin given daily", "Strong", "Moderate (HIV negative)†"] in (
        regimens.rows
    )


def test_booktabs_table_text_does_not_leak_into_body(booktabs_page):
    body = " ".join(line.text for box in booktabs_page.boxes for line in box.lines)
    assert "14.1–25.0 kg, 450 mg" not in body
    assert "Three Months of Daily Isoniazid Plus Rifampin" in body  # text below the tables


def test_table_becomes_one_markdown_chunk(page):
    chunks = chunk_document("jnc7", [page])
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    text = tables[0].text
    assert text.startswith("Table 1. Classification")
    assert "| Normal | <120 | and <80 |" in text
    assert "| Stage 2 Hypertension | ≥160 | or ≥100 |" in text
    assert tables[0].page_start == tables[0].page_end == 1
