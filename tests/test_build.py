from medquery.ingest.blocks import Line, Page, TextBox
from medquery.ingest.build import drop_toc_pages, is_toc_page, strip_running_text


def box(text: str, top: float) -> TextBox:
    return TextBox([Line(text, 9.0, False)], top, top + 0.02)


def test_repeated_headers_and_page_numbers_are_removed():
    pages = [
        Page(
            n,
            boxes=[
                box(f"MMWR / February 14, 2020 / Vol. 69 / No. {n}", 0.95),
                box("Body text that should stay on the page.", 0.4),
                box(str(n), 0.97),
            ],
        )
        for n in range(1, 4)
    ]
    strip_running_text(pages)
    for page in pages:
        assert [b.lines[0].text for b in page.boxes] == ["Body text that should stay on the page."]


def toc_page() -> Page:
    entries = [f"Chapter {n}: Topic number {n} ........ {10 * n}" for n in range(1, 8)]
    return Page(1, boxes=[box("TABLE OF CONTENTS", 0.1)] + [box(e, 0.2) for e in entries])


def test_table_of_contents_page_is_detected_and_dropped():
    page = toc_page()
    assert is_toc_page(page)
    drop_toc_pages([page])
    assert page.boxes == []


def test_table_of_contents_continuation_page_is_dropped():
    entries = [f"Chapter {n}: Topic number {n} ........ {10 * n}" for n in range(8, 14)]
    continuation = Page(2, boxes=[box(e, 0.2) for e in entries])
    body = Page(3, boxes=[box(e, 0.2) for e in entries])  # numbered lines, but not after a TOC
    assert not is_toc_page(body)
    drop_toc_pages([toc_page(), continuation])
    assert continuation.boxes == []


def test_body_page_that_mentions_contents_is_kept():
    text = "The contents of this guide reflect the 2020 recommendations for treatment."
    page = Page(1, boxes=[box(text, 0.2 + i / 100) for i in range(8)])
    assert not is_toc_page(page)


def test_page_text_keeps_reading_order_tables_and_rejoins_hyphens():
    from medquery.ingest.blocks import Table
    from medquery.ingest.build import page_text

    table = Table(
        rows=[["Drug", "Dose"], ["Rifampin", "10 mg/kg"]], caption="TABLE 4. Doses", top=0.5
    )
    page = Page(
        3,
        boxes=[
            TextBox(
                [Line("Before the com-", 9.0, False), Line("pelling table.", 9.0, False)], 0.2, 0.3
            ),
            TextBox([Line("After the table.", 9.0, False)], 0.7, 0.8),
        ],
        tables=[table],
    )
    assert page_text(page) == (
        "Before the compelling table.\n\n"
        "TABLE 4. Doses\n| Drug | Dose |\n|---|---|\n| Rifampin | 10 mg/kg |\n\n"
        "After the table."
    )


def test_unique_text_in_margin_is_kept():
    pages = [Page(1, boxes=[box("Table 3 continued from previous page", 0.03)])]
    strip_running_text(pages)
    assert len(pages[0].boxes) == 1
