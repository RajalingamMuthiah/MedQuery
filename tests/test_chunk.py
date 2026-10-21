from medquery.ingest.blocks import Line, Page, Table, TextBox
from medquery.ingest.chunk import (
    CEILING_WORDS,
    QUIZ_SECTION_RE,
    SHORT_WORDS,
    Chunk,
    chunk_document,
    join_lines,
    junk_reason,
    merge_lonely_siblings,
)

H1, H2 = 16.0, 12.0


def heading(text: str, size: float = H1) -> TextBox:
    return TextBox([Line(text, size, True)], 0.1, 0.12)


def para(text: str, top: float = 0.2) -> TextBox:
    return TextBox([Line(text, 10.0, False)], top, top + 0.05)


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}word{i}" for i in range(n)) + "."  # every token holds a real word


def sections_doc() -> list[Page]:
    return [
        Page(1, boxes=[heading("Diagnosis"), para(words(80, "diag")), heading("Tests", H2)]),
        Page(2, boxes=[para(words(60, "test")), heading("Treatment"), para(words(70, "treat"))]),
    ]


def test_chunks_follow_sections_and_never_cross_them():
    chunks = chunk_document("doc", sections_doc())
    assert [c.section for c in chunks] == ["Diagnosis", "Diagnosis > Tests", "Treatment"]
    assert all(len({w[:4] for w in c.text.split()}) == 1 for c in chunks)


def test_page_ranges_follow_content():
    chunks = chunk_document("doc", sections_doc())
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (2, 2)]


def test_section_spanning_pages_records_both_pages():
    pages = [
        Page(1, boxes=[heading("Diagnosis"), para(words(40, "a"))]),
        Page(2, boxes=[para(words(40, "b"))]),
    ]
    [chunk] = chunk_document("doc", pages)
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_ids_are_sequential_and_stable():
    first = chunk_document("tb-x", sections_doc())
    second = chunk_document("tb-x", sections_doc())
    assert [c.id for c in first] == ["tb-x:0000", "tb-x:0001", "tb-x:0002"]
    assert [(c.id, c.text) for c in first] == [(c.id, c.text) for c in second]


def test_size_limits_are_respected():
    boxes = [heading("Long section")] + [para(words(60, f"p{i}x")) for i in range(20)]
    chunks = chunk_document("doc", [Page(1, boxes=boxes)])
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= CEILING_WORDS for c in chunks)


def test_over_long_paragraph_without_sentences_is_split():
    run_on = " ".join(f"num{i}" for i in range(1200))  # e.g. OCR'd table digits, no periods
    chunks = chunk_document("doc", [Page(1, boxes=[para(run_on)])])
    assert len(chunks) >= 3
    assert all(len(c.text.split()) <= CEILING_WORDS for c in chunks)
    assert " ".join(c.text for c in chunks).split() == run_on.split()


def test_table_is_one_whole_chunk_in_its_section():
    rows = [["Drug", "Dose"]] + [[f"drug{i}", f"{i} mg"] for i in range(60)]
    table = Table(rows=rows, caption="Table 4. Regimens", top=0.5)
    page = Page(1, boxes=[heading("Treatment"), para(words(30, "t"), top=0.2)], tables=[table])
    chunks = chunk_document("doc", [page])
    [table_chunk] = [c for c in chunks if c.kind == "table"]
    assert table_chunk.section == "Treatment"
    assert table_chunk.text.startswith("Table 4. Regimens")
    assert table_chunk.text.count("\n| drug") == 60


def test_multi_line_heading_is_merged():
    pages = [Page(1, boxes=[heading("Stage 1"), heading("Hypertension"), para(words(20, "s"))])]
    [chunk] = chunk_document("doc", pages)
    assert chunk.section == "Stage 1 Hypertension"


def test_tiny_fragments_are_dropped():
    pages = [Page(1, boxes=[heading("Notes"), para("ok"), para(words(20, "real"), top=0.4)])]
    chunks = chunk_document("doc", pages)
    assert all(len(c.text.split()) >= 5 for c in chunks)


def word_counts(chunks) -> list[int]:
    return [len(c.text.split()) for c in chunks]


def test_short_tail_merges_into_previous_chunk_in_same_section():
    boxes = [heading("Dosing"), para(words(300, "a")), para(words(20, "b"), top=0.5)]
    chunks = chunk_document("doc", [Page(1, boxes=boxes)])
    assert word_counts(chunks) == [320]


def test_short_head_merges_into_next_chunk_when_it_has_no_left_neighbour():
    boxes = [heading("Dosing"), para(words(20, "a")), para(words(440, "b"), top=0.5)]
    chunks = chunk_document("doc", [Page(1, boxes=boxes)])
    assert word_counts(chunks) == [460]
    assert chunks[0].text.startswith("aword0")


def test_short_chunks_never_merge_across_sections():
    pages = [
        Page(
            1,
            boxes=[
                heading("Diagnosis"),
                para(words(20, "d")),
                heading("Treatment"),
                para(words(300, "t"), top=0.5),
            ],
        )
    ]
    chunks = chunk_document("doc", pages)
    assert [(c.section, len(c.text.split())) for c in chunks] == [
        ("Diagnosis", 20),
        ("Treatment", 300),
    ]
    assert SHORT_WORDS > 20  # the Diagnosis chunk stays short rather than crossing sections


def test_document_title_shared_by_every_chunk_is_stripped_from_sections():
    pages = [
        Page(
            1,
            boxes=[
                heading("Guidelines for the Treatment of Latent TB", 20.0),
                heading("Summary"),
                para(words(40, "s")),
                heading("Introduction"),
                para(words(40, "i"), top=0.5),
                heading("Evidence search", H2),
                para(words(40, "e"), top=0.7),
            ],
        )
    ]
    chunks = chunk_document("doc", pages)
    assert [c.section for c in chunks] == [
        "Summary",
        "Introduction",
        "Introduction > Evidence search",
    ]


def test_stacked_cover_titles_are_all_stripped():
    pages = [
        Page(
            1,
            boxes=[
                heading("Core Curriculum on Tuberculosis", 24.0),
                heading("WHAT THE CLINICIAN SHOULD KNOW", 20.0),
                heading("Chapter 1: Transmission"),
                para(words(40, "a")),
                heading("Chapter 2: Testing"),
                para(words(40, "b"), top=0.5),
            ],
        )
    ]
    chunks = chunk_document("doc", pages)
    assert [c.section for c in chunks] == ["Chapter 1: Transmission", "Chapter 2: Testing"]


def text_chunk(section: str, text: str, page: int = 1, kind: str = "text") -> Chunk:
    return Chunk(section, page, page, kind, "text", None, text)


SPUTUM = "phlegm from deep in the lungs; collected in a sterile container for examination"
GENOTYPING = "laboratory-based approach used to analyze the genetic material of M. tuberculosis"


def test_lonely_short_siblings_merge_under_parent_keeping_subheadings():
    chunks = [
        text_chunk("Glossary > sputum", SPUTUM, page=202),
        text_chunk("Glossary > genotyping", GENOTYPING, page=203),
    ]
    [merged] = merge_lonely_siblings(chunks)
    assert merged.section == "Glossary"
    assert merged.text == f"sputum\n{SPUTUM}\n\ngenotyping\n{GENOTYPING}"
    assert (merged.page_start, merged.page_end) == (202, 203)


def test_lonely_short_chunk_merges_into_its_parent_section_chunk():
    parent_text = words(100, "p")
    chunks = [
        text_chunk("Treatment", parent_text),
        text_chunk("Treatment > Monitoring", words(20, "m")),
    ]
    [merged] = merge_lonely_siblings(chunks)
    assert merged.section == "Treatment"
    assert merged.text.startswith(parent_text)
    assert "\n\nMonitoring\nmword0" in merged.text


def test_sibling_merge_never_crosses_top_level_sections():
    chunks = [
        text_chunk("Diagnosis > Tests", words(20, "d")),
        text_chunk("Treatment > Regimens", words(20, "t")),
    ]
    assert merge_lonely_siblings(chunks) == chunks


def test_top_level_short_chunk_has_no_parent_and_is_not_merged():
    chunks = [text_chunk("Diagnosis", words(20, "d")), text_chunk("Treatment", words(300, "t"))]
    assert merge_lonely_siblings(chunks) == chunks


def test_sibling_merge_respects_the_size_ceiling():
    chunks = [
        text_chunk("Treatment > Dosing", words(CEILING_WORDS - 5, "big")),
        text_chunk("Treatment > Notes", words(20, "n")),
    ]
    assert merge_lonely_siblings(chunks) == chunks


def test_tables_never_take_part_in_sibling_merges():
    chunks = [
        text_chunk("Treatment > Notes", words(20, "n")),
        text_chunk("Treatment > Table 4", "| Drug | Dose |\n|---|---|", kind="table"),
    ]
    assert merge_lonely_siblings(chunks) == chunks


def test_junk_reasons():
    assert junk_reason(text_chunk("s", "12 34 56 7 89")) == (
        "no real words (bare numbers or references)"
    )
    toc = "\n\n".join(f"Topic number {n} ........................ {n}" for n in range(6))
    assert junk_reason(text_chunk("s", toc)) == "contents listing"
    rotated = "\n\n".join(["HSAD hti", "W erusserP", "doolB", "l", "ruoY", "gnirewoL", "ot"])
    assert junk_reason(text_chunk("s", rotated)) == (
        "scattered labels (figure, axis or rotated text)"
    )


def test_real_short_content_is_not_junk():
    assert junk_reason(text_chunk("Glossary > sputum", SPUTUM)) is None
    quiz = "A. Fever B. Weight loss C. Fatigue D. Runny nose"
    assert junk_reason(text_chunk("Knowledge Check", quiz)) is None
    ocr_table_row = "Alabama 1,234 56.7 1,190 55.2 Alaska 210 82.3 198 79.0"
    assert junk_reason(text_chunk("", ocr_table_row)) is None
    assert junk_reason(text_chunk("s", "| 1 | 2 |\n|---|---|", kind="table")) is None


def test_label_heavy_chunks_with_prose_are_kept():
    """Found in the htn-dash audit: recipes and nutrient panels were wrongly dropped."""
    recipe = "\n\n".join(
        [
            "41/ 2",
            "cups",
            "cooked brown rice (cooked in unsalted water)",
            "11/ 2 1/ 4",
            "tsp cup",
            "1. Cook rice according to directions on the package.",
        ]
    )
    nutrients = "\n\n".join(
        ["Total fat", "27% of calories", "Sodium", "Saturated fat", "2,300 mg*", "4,700 mg"]
        + ["* 1,500 mg sodium was a lower goal tested and found to be even better"]
    )
    assert junk_reason(text_chunk("Recipes > Scallion Rice", recipe)) is None
    assert junk_reason(text_chunk("Daily Nutrient Goals", nutrients)) is None


def test_chunk_document_reports_dropped_junk_with_reasons():
    rotated = [
        para(t, top=0.2 + i / 50)
        for i, t in enumerate(["HSAD hti", "erusserP", "doolB", "ruoY", "gnirewoL", "ediuG"])
    ]
    pages = [Page(1, boxes=[heading("Figure"), *rotated, heading("Body"), para(words(40, "b"))])]
    dropped: list[tuple[str, Chunk]] = []
    chunks = chunk_document("doc", pages, dropped)
    assert [c.section for c in chunks] == ["Body"]
    assert [(reason, c.section) for reason, c in dropped] == [
        ("scattered labels (figure, axis or rotated text)", "Figure")
    ]


def quiz_doc() -> list[Page]:
    return [
        Page(
            1,
            boxes=[
                heading("Chapter 1: Transmission"),
                para(words(200, "body")),
                heading("Knowledge Check", H2),
                para("HIV-positive persons are not at high risk for active TB disease."),
                para("A. True B. False", top=0.5),
            ],
        )
    ]


def test_knowledge_check_sections_become_quiz_chunks_and_are_not_dropped():
    dropped: list[tuple[str, Chunk]] = []
    chunks = chunk_document("doc", quiz_doc(), dropped)
    assert [(c.section, c.kind) for c in chunks] == [
        ("Chapter 1: Transmission", "text"),
        ("Chapter 1: Transmission > Knowledge Check", "quiz"),
    ]
    assert dropped == []


def test_quiz_chunks_never_merge_into_real_content():
    # The quiz chunk is short and alone in its section, but must not join the chapter text.
    chunks = chunk_document("doc", quiz_doc())
    body = next(c for c in chunks if c.kind == "text")
    assert "HIV-positive" not in body.text
    assert "Knowledge Check" not in body.text


def test_quiz_label_matches_only_exercise_sections():
    labelled = [
        "Chapter 3 > Knowledge Check > 5. Which of the following",
        "Self-Assessment",
        "Review Questions",
    ]
    for section in labelled:
        assert QUIZ_SECTION_RE.search(section), section
    assert not QUIZ_SECTION_RE.search("Checking blood sugar")
    assert not QUIZ_SECTION_RE.search("Knowledge gaps in LTBI treatment")


def test_paragraph_split_by_a_column_or_page_break_is_rejoined():
    pages = [
        Page(1, boxes=[heading("Introduction"), para("Most infected persons are asymptomatic")]),
        Page(2, boxes=[para("and classified as having latent tuberculosis infection.")]),
    ]
    [chunk] = chunk_document("doc", pages)
    assert chunk.text == (
        "Most infected persons are asymptomatic and classified as having latent "
        "tuberculosis infection."
    )
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_complete_paragraphs_stay_separate():
    pages = [
        Page(
            1,
            boxes=[
                heading("Introduction"),
                para("Treatment of LTBI is effective in preventing disease."),
                para("rifampin is one option, written here in lowercase on purpose.", top=0.4),
            ],
        )
    ]
    [chunk] = chunk_document("doc", pages)
    assert "disease.\n\nrifampin" in chunk.text


def test_join_lines_rejoins_hyphenated_words_only_before_lowercase():
    assert join_lines(["the com-", "pelling case"]) == "the compelling case"
    assert join_lines(["see Table", "8 below"]) == "see Table 8 below"
    assert join_lines(["anti-", "TNF therapy"]) == "anti- TNF therapy"
