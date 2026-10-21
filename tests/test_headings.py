import pytest

from medquery.ingest.blocks import Line, Page, TextBox
from medquery.ingest.headings import HeadingDetector, body_size, is_bold_block

BODY = "Tuberculosis is spread through the air from one person to another person."


def make_pages(*lines: Line) -> list[Page]:
    body = [Line(BODY, 10.0, False) for _ in range(20)]
    return [Page(number=1, boxes=[TextBox(body + list(lines), 0.1, 0.9)])]


def test_body_size_is_most_common_size():
    pages = make_pages(Line("Big Title", 18.0, False))
    assert body_size(pages) == 10.0


def test_levels_rank_by_size_then_bold():
    h1, h2 = Line("Treatment", 16.0, True), Line("Latent TB infection", 12.0, False)
    bold = Line("Rifampin regimen", 10.0, True)
    detector = HeadingDetector(make_pages(h1, h2))
    assert detector.level(h1, bold_block=False) == 1
    assert detector.level(h2, bold_block=False) == 2
    assert detector.level(bold, bold_block=True) == 3


def test_bold_line_outside_a_bold_block_is_not_a_heading():
    detector = HeadingDetector(make_pages())
    assert detector.level(Line("Important", 10.0, True), bold_block=False) is None


def test_is_bold_block():
    bold = Line("What is diabetes?", 10.0, True)
    plain = Line(BODY, 10.0, False)
    assert is_bold_block(TextBox([bold], 0.1, 0.12))
    assert not is_bold_block(TextBox([bold, plain], 0.1, 0.15))  # run-in label + paragraph
    assert not is_bold_block(TextBox([bold] * 4, 0.1, 0.2))  # a bold paragraph, not a heading


def test_body_text_is_not_a_heading():
    detector = HeadingDetector(make_pages())
    assert detector.level(Line(BODY, 10.0, False), bold_block=False) is None


def test_long_or_sentence_like_large_lines_are_not_headings():
    long_line = Line(" ".join(["word"] * 20), 16.0, False)
    sentence = Line("This is a sentence.", 16.0, False)
    detector = HeadingDetector(make_pages(long_line, sentence))
    assert detector.level(long_line, bold_block=False) is None
    assert detector.level(sentence, bold_block=False) is None


@pytest.mark.parametrize(
    "text",
    [
        "• Gestational diabetes – Some women",
        "– Check your feet",
        "http://www.cdc.gov/diabetes/ndep/",
        "For more information call 1-800-232-4636",
        "Email: info@example.gov",
    ],
)
def test_bullets_links_and_phone_numbers_are_not_headings(text):
    detector = HeadingDetector(make_pages())
    assert detector.level(Line(text, 10.0, True), bold_block=True) is None


def test_ocr_lines_without_size_are_never_headings():
    detector = HeadingDetector(make_pages())
    assert detector.level(Line("TRENDS IN TUBERCULOSIS", 0.0, False), bold_block=True) is None
