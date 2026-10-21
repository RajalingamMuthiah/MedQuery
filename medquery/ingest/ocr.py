"""OCR fallback for scanned pages (Tesseract via pytesseract)."""

import pytesseract
from PIL import Image

from medquery.ingest.blocks import Line, Page, TextBox

MIN_TEXT_CHARS = 50  # fewer extractable characters than this means "no usable text layer"
OCR_DPI = 300
MIN_BOX_WORDS = 3  # shorter OCR paragraphs are mostly noise from charts and rules


def needs_ocr(plumber_page) -> bool:
    """A page needs OCR when it has an image but (almost) no extractable text."""
    chars = sum(1 for c in plumber_page.chars if not c["text"].isspace())
    return chars < MIN_TEXT_CHARS and bool(plumber_page.images)


def ocr_image(image: Image.Image, number: int) -> Page:
    """OCR one rendered page. Tesseract paragraphs become text boxes; no table structure."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    height = image.height
    paragraphs: dict[tuple[int, int], dict[int, list[int]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        paragraphs.setdefault(key, {}).setdefault(data["line_num"][i], []).append(i)

    page = Page(number=number, source="ocr")
    confidences = []
    for lines in paragraphs.values():
        indices = [i for line in lines.values() for i in line]
        if len(indices) < MIN_BOX_WORDS:
            continue
        tops = [data["top"][i] for i in indices]
        bottoms = [data["top"][i] + data["height"][i] for i in indices]
        box_lines = [
            Line(text=" ".join(data["text"][i] for i in line), size=0.0, bold=False)
            for line in lines.values()
        ]
        page.boxes.append(TextBox(box_lines, top=min(tops) / height, bottom=max(bottoms) / height))
        confidences += [float(data["conf"][i]) for i in indices]
    page.ocr_words = len(confidences)
    page.ocr_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    return page
