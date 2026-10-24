"""Extract text boxes (in reading order), tables and page numbers from a PDF."""

import re
from bisect import bisect_right
from pathlib import Path
from statistics import median

import pdfplumber
import pypdfium2 as pdfium

from medquery.ingest import ocr
from medquery.ingest.blocks import Line, Page, Table, TextBox

RULE_TOLERANCE = 3  # points; ruling lines closer than this count as touching
CAPTION_GAP = 30  # points; a caption must end this close above its table
CAPTION_RE = re.compile(r"^(table|box|exhibit)\b", re.IGNORECASE)
BOLD_RE = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)


def extract_pages(pdf_path: Path) -> list[Page]:
    pages = []
    scan_doc = None
    with pdfplumber.open(pdf_path, laparams={}) as pdf:
        for number, plumber_page in enumerate(pdf.pages, start=1):
            if ocr.needs_ocr(plumber_page):
                scan_doc = scan_doc or pdfium.PdfDocument(pdf_path)
                image = scan_doc[number - 1].render(scale=ocr.OCR_DPI / 72).to_pil()
                pages.append(ocr.ocr_image(image, number))
            else:
                pages.append(text_page(plumber_page, number))
            plumber_page.close()
    if scan_doc is not None:
        scan_doc.close()
    return pages


def text_page(page, number: int) -> Page:
    found = find_tables(page)
    table_bboxes = [bbox for bbox, _ in found]
    boxes = []
    for obj in page.objects.get("textboxhorizontal", []):
        if any(_centre_inside(obj, bbox) for bbox in table_bboxes):
            continue
        lines = _box_lines(page, obj)
        if lines:
            boxes.append(
                TextBox(lines, top=obj["top"] / page.height, bottom=obj["bottom"] / page.height)
            )
    tables = [
        Table(
            rows=rows,
            caption=_caption(page, bbox),
            top=bbox[1] / page.height,
            caption_more=_caption_more(page, bbox),
        )
        for bbox, rows in found
    ]
    return Page(number=number, boxes=boxes, tables=tables)


def find_tables(page) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """Find ruled tables, including "open" tables that lack outer borders.

    Ruling lines that touch are grouped into frames; each frame gets explicit outer edges so
    pdfplumber can close the first/last columns and the last row.
    """
    vertical, horizontal = _rules(page)
    edges_v, edges_h = [], []
    for x0, top, x1, bottom in _frames(vertical, horizontal):
        for x in (x0, x1):
            edges_v.append(_edge(x, top, x, bottom))
        for y in (top, bottom):
            edges_h.append(_edge(x0, y, x1, y))
    settings = {"explicit_vertical_lines": edges_v, "explicit_horizontal_lines": edges_h}
    results = []
    for table in page.find_tables(settings):
        rows = _clean_rows(table.extract())
        if len(rows) >= 2 and _filled_columns(rows) >= 2:
            results.append((table.bbox, rows))
    for bbox, rows in _booktabs_tables(page, horizontal):
        if not any(_overlap(bbox, other) for other, _ in results):
            results.append((bbox, rows))
    return results


def _booktabs_tables(page, horizontal):
    """Tables ruled with horizontal lines only (top, header and bottom rules).

    Anchored on a "Table N" caption with at least three rule rows below it. Column boundaries
    come from where the header/bottom rules are broken into segments; rows come from text lines.
    """
    rule_rows: list[list[tuple]] = []
    for segment in sorted(horizontal, key=lambda s: s[1]):
        if rule_rows and abs(rule_rows[-1][0][1] - segment[1]) <= 1:
            rule_rows[-1].append(segment)
        else:
            rule_rows.append([segment])
    captions = [
        line
        for line in page.extract_text_lines(strip=True)
        if CAPTION_RE.match(line["text"]) and len(line["text"].split()) > 2
    ]
    results = []
    for i, caption in enumerate(captions):
        limit = captions[i + 1]["top"] if i + 1 < len(captions) else page.height
        rows_below = [
            r
            for r in rule_rows
            if caption["bottom"] <= r[0][1] < limit
            and min(s[0] for s in r) < caption["x1"]
            and max(s[2] for s in r) > caption["x0"]
        ]
        if len(rows_below) < 3 or rows_below[0][0][1] - caption["bottom"] > CAPTION_GAP:
            continue
        left = min(s[0] for r in rows_below for s in r)
        right = max(s[2] for r in rows_below for s in r)
        ends = sorted({round(x) for r in rows_below for s in r for x in (s[0], s[2])})
        inner = [x for x in ends if left + RULE_TOLERANCE < x < right - RULE_TOLERANCE]
        breaks = [x for j, x in enumerate(inner) if j == 0 or x - inner[j - 1] > RULE_TOLERANCE]
        if not breaks:
            continue
        bbox = (left, rows_below[0][0][1], right, rows_below[-1][0][1])
        page_left, page_top, page_right, page_bottom = page.bbox
        padded = (
            max(left - RULE_TOLERANCE, page_left),
            max(bbox[1], page_top),
            min(right + RULE_TOLERANCE, page_right),
            min(bbox[3], page_bottom),
        )
        rows = _rows_from_columns(page.within_bbox(padded), [left, *breaks, right])
        if len(rows) >= 2 and _filled_columns(rows) >= 2:
            results.append((bbox, rows))
    return results


def _rows_from_columns(region, bounds: list[float]) -> list[list[str]]:
    """One row per text line; each word goes to the column containing its centre."""
    lines: list[tuple[float, list[dict]]] = []
    for word in sorted(region.extract_words(), key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0] - word["top"]) <= RULE_TOLERANCE:
            lines[-1][1].append(word)
        else:
            lines.append((word["top"], [word]))
    rows = []
    for _, words in lines:
        cells: list[list[str]] = [[] for _ in range(len(bounds) - 1)]
        for word in sorted(words, key=lambda w: w["x0"]):
            centre = (word["x0"] + word["x1"]) / 2
            column = min(max(bisect_right(bounds, centre) - 1, 0), len(cells) - 1)
            cells[column].append(word["text"])
        rows.append([" ".join(cell) for cell in cells])
    return [row for row in rows if any(row)]


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _rules(page):
    vertical, horizontal = [], []
    for obj in page.lines + page.rects:
        width, height = obj["x1"] - obj["x0"], obj["bottom"] - obj["top"]
        segment = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
        if width <= 2 and height > 10:
            vertical.append(segment)
        elif height <= 2 and width > 30:
            horizontal.append(segment)
    return vertical, horizontal


def _touch(a, b) -> bool:
    t = RULE_TOLERANCE
    return not (a[2] + t < b[0] or b[2] + t < a[0] or a[3] + t < b[1] or b[3] + t < a[1])


def _frames(vertical, horizontal):
    segments = [(True, s) for s in vertical] + [(False, s) for s in horizontal]
    seen, frames = set(), []
    for start in range(len(segments)):
        if start in seen:
            continue
        seen.add(start)
        stack, group = [start], []
        while stack:
            i = stack.pop()
            group.append(segments[i])
            for j in range(len(segments)):
                if j not in seen and _touch(segments[i][1], segments[j][1]):
                    seen.add(j)
                    stack.append(j)
        n_vertical = sum(1 for is_v, _ in group if is_v)
        if n_vertical >= 2 and len(group) - n_vertical >= 2:
            coords = [s for _, s in group]
            frames.append(
                (
                    min(c[0] for c in coords),
                    min(c[1] for c in coords),
                    max(c[2] for c in coords),
                    max(c[3] for c in coords),
                )
            )
    return frames


def _edge(x0, top, x1, bottom) -> dict:
    return {
        "object_type": "line",
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "width": x1 - x0,
        "height": bottom - top,
    }


def _clean_rows(raw: list[list[str | None]]) -> list[list[str]]:
    """Normalise cell text and fill merged cells (None) from above, else from the left."""
    rows: list[list[str]] = []
    for r, raw_row in enumerate(raw):
        row = []
        for c, cell in enumerate(raw_row):
            if cell is None:
                if r > 0 and c < len(rows[-1]):
                    cell = rows[-1][c]
                elif c > 0:
                    cell = row[c - 1]
            row.append(_normalise(re.sub(r"(\w)-\n(?=[a-z])", r"\1", cell or "")))
        rows.append(row)
    return [row for row in rows if any(row)]


def _filled_columns(rows: list[list[str]]) -> int:
    width = max(len(row) for row in rows)
    return sum(1 for c in range(width) if any(c < len(row) and row[c] for row in rows))


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _centre_inside(obj, bbox) -> bool:
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _box_lines(page, box) -> list[Line]:
    pad = 0.5
    left, top, right, bottom = page.bbox
    bbox = (
        max(box["x0"] - pad, left),
        max(box["top"] - pad, top),
        min(box["x1"] + pad, right),
        min(box["bottom"] + pad, bottom),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:  # box lies outside the visible page
        return []
    region = page.within_bbox(bbox)
    lines = []
    for raw in region.extract_text_lines(return_chars=True, strip=True):
        text = _normalise(raw["text"])
        chars = [c for c in raw["chars"] if not c["text"].isspace()]
        if not text or not chars:
            continue
        bold = sum(1 for c in chars if BOLD_RE.search(c["fontname"])) >= 0.8 * len(chars)
        size = median(c["size"] for c in chars)
        lines.append(Line(text=text, size=round(size, 1), bold=bold))
    return lines


def _caption_line(page, bbox) -> dict | None:
    best = None
    for line in page.extract_text_lines(strip=True):
        above = bbox[1] - CAPTION_GAP <= line["bottom"] <= bbox[1] + 2
        overlaps = line["x0"] < bbox[2] and line["x1"] > bbox[0]
        if above and overlaps and CAPTION_RE.match(line["text"]):
            best = line
    return best


def _caption_more(page, bbox) -> str:
    """The caption's wrapped lines: text lines between the 'Table N' line and the table top
    (used by corpus v2 only; corpus v1 keeps the single caption line)."""
    first = _caption_line(page, bbox)
    if first is None:
        return ""
    more = [
        line["text"]
        for line in page.extract_text_lines(strip=True)
        if line["top"] >= first["bottom"] - 1
        and line["bottom"] <= bbox[1] + 2
        and line["x0"] < bbox[2]
        and line["x1"] > bbox[0]
        and line["text"] != first["text"]
    ]
    return _normalise(" ".join(more))


def _caption(page, bbox) -> str:
    """The 'Table N…' line just above a table, if there is one."""
    best = ""
    for line in page.extract_text_lines(strip=True):
        above = bbox[1] - CAPTION_GAP <= line["bottom"] <= bbox[1] + 2
        overlaps = line["x0"] < bbox[2] and line["x1"] > bbox[0]
        if above and overlaps and CAPTION_RE.match(line["text"]):
            best = _normalise(line["text"])
    return best
