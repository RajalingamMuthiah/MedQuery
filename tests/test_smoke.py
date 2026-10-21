import importlib

import pytest

import medquery

SUBPACKAGES = ["ingest", "retrieval", "generation", "verify", "api", "eval", "cache"]


def test_version():
    assert medquery.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"medquery.{name}")
