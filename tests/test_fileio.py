import pytest

from medquery import paths
from medquery.fileio import LockHeld, atomic_write, exclusive_lock
from medquery.ingest.build import LOCK_NAME, build
from medquery.ingest.manifest import load_manifest


def test_atomic_write_replaces_file_with_complete_content(tmp_path):
    target = tmp_path / "chunks.jsonl"
    target.write_text("old\n", encoding="utf-8")
    with atomic_write(target) as f:
        f.write("new line 1\n")
        assert target.read_text(encoding="utf-8") == "old\n"  # readers still see the old file
        f.write("new line 2\n")
    assert target.read_text(encoding="utf-8") == "new line 1\nnew line 2\n"
    assert list(tmp_path.iterdir()) == [target]  # no temp file left behind


def test_atomic_write_leaves_original_untouched_on_crash(tmp_path):
    target = tmp_path / "chunks.jsonl"
    target.write_text("complete old content\n", encoding="utf-8")
    with pytest.raises(RuntimeError), atomic_write(target) as f:
        f.write("half a rec")
        raise RuntimeError("crash mid-write")
    assert target.read_text(encoding="utf-8") == "complete old content\n"
    assert list(tmp_path.iterdir()) == [target]


def test_lock_is_exclusive_and_released(tmp_path):
    lock = tmp_path / ".build.lock"
    with exclusive_lock(lock):
        assert lock.exists()
        with pytest.raises(LockHeld, match=r"\.build\.lock exists \(pid \d+"), exclusive_lock(lock):
            pass
        assert lock.exists()  # the failed attempt must not remove the holder's lock
    assert not lock.exists()


def test_lock_is_released_when_the_body_fails(tmp_path):
    lock = tmp_path / ".build.lock"
    with pytest.raises(ValueError), exclusive_lock(lock):
        raise ValueError("build failed")
    assert not lock.exists()


def test_build_refuses_to_run_while_locked(tmp_path):
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    (out_dir / LOCK_NAME).write_text("pid 999, started earlier\n", encoding="utf-8")
    with pytest.raises(LockHeld, match="Another run is in progress"):
        build(load_manifest(paths.MANIFEST), tmp_path / "raw", out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [LOCK_NAME]  # nothing was written
