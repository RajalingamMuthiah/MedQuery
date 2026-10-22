import json
from collections import Counter

import pytest

from medquery.eval.evidence import EvidenceMapper
from medquery.eval.freeze import (
    FreezeError,
    assign_splits,
    freeze,
    load_golden,
    write_frozen,
)
from medquery.eval.golden import REVIEW_COLUMNS, write_review_csv

PAGE = "Adults receive 10 mg/kg of rifampin daily for four months, up to 600 mg."
PAGES = {("tb", 8): PAGE}
MAPPER = EvidenceMapper(
    [{"id": "tb:0001", "doc_id": "tb", "page_start": 8, "page_end": 8, "text": PAGE}]
)
EVIDENCE = 'tb | 8 | "Adults receive 10 mg/kg of rifampin daily"'


def row(rid, decision="keep", reviewed="y", answerable="yes", evidence=EVIDENCE, **kw):
    base = {c: "" for c in REVIEW_COLUMNS}
    return (
        base
        | {
            "id": rid,
            "decision": decision,
            "reviewed": reviewed,
            "question": "What daily rifampin dose do adults with LTBI receive?",
            "reference_answer": "10 mg/kg daily, up to 600 mg.",
            "evidence": evidence if answerable == "yes" else "",
            "type": "factual" if answerable == "yes" else "unanswerable",
            "topic": "tuberculosis",
            "answerable": answerable,
        }
        | kw
    )


def run(tmp_path, rows, exclude=None):
    path = tmp_path / "review_v9.csv"
    write_review_csv(path, rows)
    return freeze(path, "v9", PAGES, MAPPER, "test reviewer", exclude)


def test_valid_review_freezes_kept_and_fixed_rows_only(tmp_path):
    rows = [
        row("g001"),
        row("g002", decision="fix"),
        row("g003", decision="drop"),
        row("g004", answerable="no"),
    ]
    frozen = run(tmp_path, rows)
    assert [i["id"] for i in frozen.items] == ["g001", "g002", "g004"]
    assert frozen.meta["review"] | {"excluded": []} == {
        "rows": 4,
        "reviewed": 4,
        "reviewed_share": 1.0,
        "keep": 2,
        "fix": 1,
        "drop": 1,
        "excluded": [],
    }
    assert frozen.meta["reviewer"] == "test reviewer"


def test_at_least_a_quarter_must_be_reviewed(tmp_path):
    rows = [row(f"g{n:03d}", reviewed="y" if n == 1 else "") for n in range(1, 6)]
    with pytest.raises(FreezeError, match="only 1 of 5 rows are marked reviewed"):
        run(tmp_path, rows)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (row("g001", decision="maybe"), "decision 'maybe'"),
        (row("g001", evidence='tb | 8 | "Adults receive 15 mg/kg"'), "quote not on tb p.8"),
        (row("g001", evidence='tb | 9 | "Adults receive 10 mg/kg"'), "no page tb p.9"),
        (row("g001", evidence=""), "needs at least one evidence quote"),
        (row("g001", answerable="no") | {"evidence": EVIDENCE}, "must have no evidence"),
        (row("g001", question="Rifampin dose"), "does not end with '?'"),
    ],
)
def test_invalid_kept_rows_are_refused(tmp_path, bad, message):
    with pytest.raises(FreezeError, match=message):
        run(tmp_path, [bad])


def test_dropped_and_excluded_rows_are_not_validated(tmp_path):
    broken = row("g002", evidence='tb | 8 | "not on the page at all"')
    frozen = run(tmp_path, [row("g001"), broken | {"decision": "drop"}])
    assert [i["id"] for i in frozen.items] == ["g001"]
    frozen = run(tmp_path, [row("g001"), broken], exclude={"g002"})
    assert [i["id"] for i in frozen.items] == ["g001"]
    assert frozen.meta["review"]["excluded"] == ["g002"]


def test_excluding_an_unknown_id_is_refused(tmp_path):
    with pytest.raises(FreezeError, match="unknown id g999"):
        run(tmp_path, [row("g001")], exclude={"g999"})


def test_split_is_deterministic_and_stratified():
    def items():
        return [
            {"id": f"g{n:03d}", "topic": t, "type": k}
            for n, (t, k) in enumerate(
                [("tb", "factual")] * 10 + [("dm", "table")] * 6 + [("htn", "unanswerable")] * 3
            )
        ]

    first, second = items(), items()
    assign_splits(first)
    assign_splits(second)
    assert [i["split"] for i in first] == [i["split"] for i in second]
    tests = Counter((i["topic"], i["split"]) for i in first)
    assert tests[("tb", "test")] == 3 and tests[("dm", "test")] == 2 and tests[("htn", "test")] == 1


def test_written_set_is_never_overwritten_and_tampering_is_detected(tmp_path):
    frozen = run(tmp_path, [row("g001"), row("g002", answerable="no")])
    data_path, meta_path = write_frozen(frozen, root=tmp_path)
    items, meta = load_golden("v9", root=tmp_path)
    assert [i["id"] for i in items] == ["g001", "g002"]
    assert meta["sha256"] and json.loads(meta_path.read_text())["items"] == 2
    with pytest.raises(FreezeError, match="never overwritten"):
        write_frozen(frozen, root=tmp_path)
    data_path.write_text(data_path.read_text().replace("10 mg/kg", "15 mg/kg"))
    with pytest.raises(FreezeError, match="sha256"):
        load_golden("v9", root=tmp_path)
