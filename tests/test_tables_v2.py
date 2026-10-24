"""Corpus v2 table fix (Phase 4): caption, merged headers with units, empty columns, split
decimals, continuation chunks and long-table splits. Corpus v1 output is unchanged."""

from medquery.eval.evidence import normalise
from medquery.ingest import tables
from medquery.ingest.blocks import Line, Page, Table, TextBox
from medquery.ingest.chunk import chunk_document

TABLE_1B = [
    ["", "Diagnosed diabetes", "", "Undiagnosed diabetes"],
    ["Characteristic", "Number in Millions", "", "Number in Millions"],
    ["", "(95% CI)", "", "(95% CI)"],
    ["Total", "26 .8 (24 .4–29 .1)", "", "7 .3 (6 .3–8 .4)"],
    ["Asian, non-Hispanic", "1.6 (1.3–2.0)", "", "0.7 (0.4–1.0)"],
]


def test_decimals_are_rejoined_and_evidence_matches_both_ways():
    assert tables.rejoin_decimals("26 .8 (24 .4–29 .1)") == "26.8 (24.4–29.1)"
    assert tables.rejoin_decimals("Table 1 .") == "Table 1 ."  # needs a digit after the dot
    assert normalise("total 26 .8") == normalise("Total 26.8")


def test_empty_columns_are_dropped():
    rows = tables.drop_empty_columns(TABLE_1B)
    assert all(len(r) == 3 for r in rows) and rows[0] == [
        "",
        "Diagnosed diabetes",
        "Undiagnosed diabetes",
    ]


def test_header_rows_merge_so_each_column_carries_its_unit():
    rows = tables.clean_rows(TABLE_1B)
    head, body = tables.split_header(rows)
    assert len(head) == 3 and body[0][0] == "Total"
    assert tables.merge_header(head) == [
        "Characteristic",
        "Diagnosed diabetes Number in Millions (95% CI)",
        "Undiagnosed diabetes Number in Millions (95% CI)",
    ]
    assert body[1] == ["Asian, non-Hispanic", "1.6 (1.3–2.0)", "0.7 (0.4–1.0)"]


def test_a_table_is_never_all_header():
    head, body = tables.split_header([["a", "b"], ["c", "d"]])
    assert len(head) == 1 and len(body) == 1


def test_long_tables_split_into_parts_that_repeat_the_header():
    header = ["Name", "Value"]
    body = [[f"row{i}", " ".join(["word"] * 60)] for i in range(20)]  # ~1,220 words
    texts = tables.render_v2("Table 9. Long", header, body)
    assert len(texts) >= 3
    for i, text in enumerate(texts, 1):
        assert text.startswith(f"Table 9. Long (part {i} of {len(texts)})\n\n| Name | Value |")
        table_words = sum(len(r.split()) for r in text.splitlines()[3:])
        assert table_words <= tables.MAX_TABLE_WORDS + 4 * len(text.splitlines())
    assert tables.render_v2("T", header, body[:2]) == [tables.render_v2("T", header, body[:2])[0]]


def page_with_table(number, rows, caption="", caption_more=""):
    return Page(
        number,
        boxes=[TextBox([Line("Results", 16.0, True)], 0.05, 0.08)],
        tables=[Table(rows=rows, caption=caption, top=0.3, caption_more=caption_more)],
    )


def test_v2_chunks_complete_the_caption_and_continue_tables_across_pages():
    pages = [
        page_with_table(
            4,
            TABLE_1B,
            "Table 1b. Estimated number of adults with diabetes,",
            "by characteristic, United States, 2018",
        ),
        page_with_table(5, [["Hispanic", "4.9 (4.1–5.6)", "", "1.5 (1.0–1.9)"]]),
    ]
    v2 = [c for c in chunk_document("d", pages, table_format="v2") if c.kind == "table"]
    assert v2[0].text.startswith(
        "Table 1b. Estimated number of adults with diabetes, by characteristic, United States, 2018"
    )
    assert "Undiagnosed diabetes Number in Millions (95% CI)" in v2[0].text
    assert "26.8 (24.4–29.1)" in v2[0].text
    assert v2[1].text.startswith(
        "Table 1b. Estimated number of adults with diabetes, by characteristic, United States, "
        "2018 (continued)\n\n| Characteristic | Diagnosed diabetes Number in Millions (95% CI) |"
    )
    assert "| Hispanic | 4.9 (4.1–5.6) | 1.5 (1.0–1.9) |" in v2[1].text


def test_v1_output_is_unchanged_by_the_new_caption_field():
    pages = [page_with_table(4, TABLE_1B, "Table 1b. Estimated number", "of adults")]
    [v1] = [c for c in chunk_document("d", pages) if c.kind == "table"]
    assert v1.text == "Table 1b. Estimated number\n\n" + Table(TABLE_1B, "", 0).to_markdown()


def test_prose_chunks_are_identical_in_v1_and_v2():
    body = TextBox([Line("Prose " + "word " * 40 + "end.", 10.0, False)], 0.2, 0.25)
    pages = [
        Page(
            1,
            boxes=[TextBox([Line("Intro", 16.0, True)], 0.05, 0.08), body],
            tables=[Table(TABLE_1B, "Table 1. X", 0.5)],
        ),
        Page(2, boxes=[body]),
    ]

    def prose(fmt):
        return [c.text for c in chunk_document("d", pages, table_format=fmt) if c.kind != "table"]

    assert prose("v1") == prose("v2")


def test_text_only_tables_keep_their_first_row_as_the_header():
    rows = [
        ["Organization", "Names"],
        ["Department of Veterans Affairs", "Paul Conlin, MD"],
        ["Department of Defense", "Jane Doe, MD"],
    ]
    head, body = tables.split_header(rows)
    assert head == [rows[0]] and body == rows[1:]
    deep = [["Year", "Value"]] + [["text", "more text"]] * 5 + [["2018", "12.5"]]
    assert len(tables.split_header(deep)[0]) == 1  # first numeric row too far down


def test_a_section_label_row_ends_the_header():
    rows = [
        ["", "Diagnosed diabetes"],
        ["Characteristic", "Percentage"],
        ["Age in years", ""],
        ["18-44", "3.0 (2.6-3.6)"],
    ]
    head, body = tables.split_header(rows)
    assert len(head) == 2 and body[0] == ["Age in years", ""]
