import pytest

from medquery.eval.evidence import Evidence, EvidenceMapper, coverage, normalise


def chunk(chunk_id: str, text: str, pages: tuple[int, int] = (8, 8), doc: str = "tb") -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc,
        "page_start": pages[0],
        "page_end": pages[1],
        "text": text,
    }


CHUNKS = [
    chunk("tb:0001", "Rifampin is given daily for 4 months. Adults: 10 mg/kg."),
    chunk("tb:0002", "| Rifampin¶ | 4 mos | Adults: 10 mg/kg | Daily | 120 |\n|---|---|"),
    chunk("tb:0003", "The first half of a long sentence about treatment completion rates"),
    chunk("tb:0004", "rates were higher with shorter regimens than with longer ones overall."),
    chunk("tb:0005", "Text on another page entirely about isoniazid.", pages=(9, 10)),
    chunk("dm:0001", "Rifampin is given daily for 4 months.", doc="dm"),
]
mapper = EvidenceMapper(CHUNKS)


def test_exact_quote_maps_to_its_chunk():
    m = mapper.map(Evidence("tb", 8, "Rifampin is given daily for 4 months."))
    assert m.status == "exact"
    assert m.chunk_ids == ("tb:0001",)


@pytest.mark.parametrize(
    "quote",
    [
        "rifampin  is given\ndaily for 4 months",  # case and whitespace
        "Rifampin is given daily for 4 mon- ths",  # hyphenated across a line break
        "Rifampin is given daily for 4 months. Adults: 10 mg/kg.",
    ],
)
def test_normalised_quotes_still_match_exactly(quote):
    assert mapper.map(Evidence("tb", 8, quote)).status == "exact"


def test_quote_from_a_table_matches_across_markdown_cells():
    m = mapper.map(Evidence("tb", 8, "Rifampin¶ 4 mos Adults: 10 mg/kg Daily 120"))
    assert m.status == "exact"
    assert m.chunk_ids == ("tb:0002",)


def test_quote_split_across_two_chunks_maps_to_both():
    quote = "long sentence about treatment completion rates were higher with shorter regimens"
    m = mapper.map(Evidence("tb", 8, quote))
    assert m.status == "fuzzy"
    assert set(m.chunk_ids) == {"tb:0003", "tb:0004"}


def test_quote_not_in_any_chunk_is_missing():
    m = mapper.map(Evidence("tb", 8, "Pyrazinamide is recommended for pregnant women."))
    assert m.status == "missing"
    assert m.chunk_ids == ()


def test_page_range_and_document_are_respected():
    assert (
        mapper.map(Evidence("tb", 9, "Rifampin is given daily for 4 months.")).status == "missing"
    )
    assert mapper.map(Evidence("tb", 10, "another page entirely about isoniazid")).status == "exact"
    m = mapper.map(Evidence("dm", 8, "Rifampin is given daily for 4 months."))
    assert m.chunk_ids == ("dm:0001",)


def test_coverage_counts_each_status():
    mapped = [
        mapper.map(Evidence("tb", 8, "Rifampin is given daily for 4 months.")),
        mapper.map(Evidence("tb", 8, "Nothing like this exists anywhere here.")),
    ]
    assert coverage(mapped) == {"exact": 1, "fuzzy": 0, "missing": 1}


def test_normalise_handles_ligatures_quotes_and_dashes():
    assert normalise("ﬁrst “quoted” 10–14 kg") == 'first "quoted" 10-14 kg'
