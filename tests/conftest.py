from pathlib import Path

import pytest

from medquery import paths

TABLE_FIXTURE = paths.FIXTURES / "jnc7-express-p19-table.pdf"
BOOKTABS_FIXTURE = paths.FIXTURES / "tb-ltbi-2020-p8-booktabs.pdf"
SCAN_FIXTURE = paths.FIXTURES / "tb-reported-1964-p5-scan.pdf"


@pytest.fixture
def fixtures_dir() -> Path:
    return paths.FIXTURES
