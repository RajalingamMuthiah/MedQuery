import re

import pytest

from medquery import paths
from medquery.ingest.manifest import load_manifest, render_sources


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(paths.MANIFEST)


def test_corpus_is_the_approved_ten_documents(manifest):
    assert len(manifest.documents) == 10
    assert {d.topic for d in manifest.documents} == {"diabetes", "hypertension", "tuberculosis"}


def test_every_document_is_pinned_and_licensed(manifest):
    for doc in manifest.documents:
        assert re.fullmatch(r"[0-9a-f]{64}", doc.sha256), doc.id
        assert doc.licence and doc.licence_evidence, doc.id
        assert doc.url.startswith("https://") and doc.url.endswith(".pdf"), doc.id
        assert doc.as_of, doc.id


def test_superseded_guidance_is_marked(manifest):
    jnc7 = manifest.get("htn-jnc7-express")
    assert jnc7.as_of == "2003"
    assert "superseded" in jnc7.superseded.lower()


def test_vadod_licence_evidence_is_the_federal_work_statute(manifest):
    assert "17 U.S.C. §105" in manifest.get("dm-vadod-cpg").licence_evidence


def test_sources_md_matches_manifest(manifest):
    expected = render_sources(manifest)
    actual = paths.SOURCES_MD.read_text(encoding="utf-8")
    assert actual == expected, "docs/SOURCES.md is stale: run `python -m medquery.ingest sources`"


def test_missing_required_field_is_rejected(tmp_path):
    bad = tmp_path / "manifest.toml"
    bad.write_text('[[documents]]\nid = "x"\ntitle = "X"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_manifest(bad)


def test_duplicate_ids_are_rejected(tmp_path):
    entry = (
        '[[documents]]\nid = "x"\ntitle = "X"\npublisher = "P"\nyear = 2020\ntopic = "t"\n'
        'url = "https://e/x.pdf"\nlicence = "PD"\nlicence_evidence = "e"\nas_of = "2020"\n'
    )
    bad = tmp_path / "manifest.toml"
    bad.write_text(entry * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(bad)
