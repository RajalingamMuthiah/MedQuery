# MedQuery v2

Question answering over public health documents, with cited answers, abstention when evidence is weak, and claim-level hallucination checking.

> **Not medical advice.** MedQuery answers only from its source documents and does not diagnose conditions or give dosing instructions. Always consult a qualified health professional.

## Setup (PowerShell)

Requires [uv](https://docs.astral.sh/uv/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with English language data (`tesseract --list-langs` must show `eng`).

```powershell
uv sync
Copy-Item .env.example .env   # then add your ANTHROPIC_API_KEY
uv run pytest
```

## Corpus and ingestion

The corpus is 10 public-domain US government documents on diabetes, hypertension and tuberculosis, listed with sources and licences in [docs/SOURCES.md](docs/SOURCES.md). The single source of truth is [corpus/manifest.toml](corpus/manifest.toml), which pins every file by sha256.

```powershell
uv run python -m medquery.ingest download  # fetch PDFs into data/raw/ and verify sha256
uv run python -m medquery.ingest build     # data/processed/chunks.jsonl + build_summary.json
```

The build keeps page numbers, section headings and tables, falls back to OCR on scanned pages, and prints a per-document summary that includes mean OCR confidence.

Status: work in progress. Results will be added here only once a re-runnable script produces them.

## Known limitations

- **Superseded guidance.** JNC 7 Express (2003) is superseded by later hypertension guidance, notably the 2017 ACC/AHA guideline, which lowered the hypertension threshold from 140/90 to 130/80 mmHg. Its chunks carry an `as_of` date and a `superseded` note.
- **Tables are detected only when ruled.** Grid tables, open-sided tables and caption-anchored "booktabs" tables (horizontal rules only) become table chunks. Tables laid out with whitespace alone stay as body text.
- **Scanned pages are OCR'd as plain text.** No table structure is recovered from scans. The 1964 tuberculosis report is mostly tables, so its OCR quality is lower. Confidence is reported in the build summary rather than hidden.
- **Heading detection is heuristic,** based on font size and bold weight. OCR pages have no headings. Mis-levelled headings occur: for example, NDSR 2020's large "SUGGESTED CITATION" heading becomes the parent of its appendices.
- **Floating tables take the section where they appear in reading order.** That can be the subsection just before them rather than the one that discusses them (MMWR Table 4 is one example).
- **Quiz content is kept but labelled.** The Core Curriculum's "Knowledge Check" items are unanswered questions, and some state false claims on purpose. They are stored as `kind = "quiz"` and excluded from retrieval by default.
- **OCR errors remain in scanned text.** Examples are dropped decimal points and misread years. Each chunk carries its OCR confidence.
- **Small, US-focused corpus.** It has three topics and ten documents, mostly from US federal agencies.
