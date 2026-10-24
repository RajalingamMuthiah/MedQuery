"""Sentence units for claim verification, and re-rendering an answer from the sentences kept.

answer_v2 cites every sentence, so a sentence (with its citations) is the unit that is kept or
removed; the verifier splits each sentence into atomic claims itself."""

import re

from medquery.generation.prompts import ABSTAIN_TEXT

# A boundary is whitespace after '.', '!', '?' or a closing citation bracket, followed by the
# start of a new sentence (a capital letter or digit). Abbreviations below never end a sentence.
_BOUNDARY_RE = re.compile(r"(?<=[.!?\]】])\s+(?=[A-Z0-9])")
_ABBREVIATIONS = ("e.g.", "i.e.", "U.S.", "Dr.", "vs.", "etc.", "approx.", "No.", "Fig.")


def split_sentences(text: str) -> list[str]:
    """Sentences of an answer, each keeping the citations that follow it."""
    pieces = _BOUNDARY_RE.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece.strip():
            sentences.append(piece.strip())
    return sentences


def render(
    text: str, sentences: list[str], keep: list[bool], is_fact: list[bool]
) -> tuple[str, bool]:
    """(answer, abstained) after removing the sentences not kept.

    Nothing removed: the original text, byte for byte. Every fact sentence removed: the exact
    abstention sentence (notes such as a superseded warning do not answer anything alone)."""
    if all(keep):
        return text, False
    if not any(k and f for k, f in zip(keep, is_fact, strict=True)):
        return ABSTAIN_TEXT, True
    return " ".join(s for s, k in zip(sentences, keep, strict=True) if k), False
