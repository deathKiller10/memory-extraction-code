"""
Cache tests.

The two-layer design exists because of a real failure: on Colab with the cache
pointed at a mounted Google Drive folder, every read came back a miss, so each
API call was paid for twice. These tests cover both layers and the diagnostics
that would have told us immediately instead of after three runs.
"""

import json

import pytest

from bapca.cache import DiskCache, clear_memo


@pytest.fixture(autouse=True)
def _isolate():
    """Each test starts with an empty in-process layer."""
    clear_memo()
    yield
    clear_memo()


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------

def test_real_answer_round_trips(tmp_path):
    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "ready")
    assert cache.get(key) == "ready"
    assert cache.hits == 1


def test_empty_string_is_not_treated_as_a_cached_answer(tmp_path):
    """complete() uses `if cached:` so a cached '' re-fetches rather than
    returning a blank answer that looks perfectly healthy in a results table."""
    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "")
    assert not cache.get(key)


def test_missing_key_is_a_miss(tmp_path):
    cache = DiskCache("t", root=tmp_path)
    assert cache.get(DiskCache.make_key(prompt="never-written")) is None
    assert cache.misses == 1


def test_cache_key_changes_with_every_input():
    base = dict(provider="groq", model="m", prompt="p", temperature=0.0)
    keys = {
        DiskCache.make_key(**base),
        DiskCache.make_key(**{**base, "model": "other"}),
        DiskCache.make_key(**{**base, "prompt": "q"}),
        DiskCache.make_key(**{**base, "temperature": 0.7}),
    }
    assert len(keys) == 4


def test_key_is_order_independent():
    assert DiskCache.make_key(a=1, b=2) == DiskCache.make_key(b=2, a=1)


# --------------------------------------------------------------------------
# The in-process layer -- what actually protects the quota
# --------------------------------------------------------------------------

def test_memo_serves_a_repeat_even_if_the_file_vanishes(tmp_path):
    """The Drive failure mode: the write does not become visible. Within one
    run we must still never pay twice."""
    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "answer")
    cache._path(key).unlink()               # simulate the file not being there
    assert cache.get(key) == "answer"
    assert cache.memo_hits == 1


def test_memo_is_shared_between_clients_on_the_same_root(tmp_path):
    """Generator and judge in one process should not re-fetch each other's work."""
    first = DiskCache("shared", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    first.put(key, "answer")

    second = DiskCache("shared", root=tmp_path)
    assert second.get(key) == "answer"
    assert second.memo_hits == 1


def test_different_namespaces_do_not_collide(tmp_path):
    a = DiskCache("gen", root=tmp_path)
    b = DiskCache("judge", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    a.put(key, "from-generator")
    assert b.get(key) is None


def test_disk_read_populates_the_memo(tmp_path):
    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "answer")
    clear_memo()

    assert cache.get(key) == "answer"       # from disk
    assert cache.memo_hits == 0
    assert cache.get(key) == "answer"       # now from memo
    assert cache.memo_hits == 1


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    """A Colab runtime killed mid-write must not break the next session."""
    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "value")
    cache._path(key).write_text("{ truncated")
    clear_memo()
    assert cache.get(key) is None


def test_write_falls_back_when_rename_is_refused(tmp_path, monkeypatch):
    """Drive's FUSE mount can reject rename-over; a plain write must still land."""
    from pathlib import Path

    def no_rename(self, target):
        raise OSError("rename not supported on this filesystem")

    monkeypatch.setattr(Path, "replace", no_rename)

    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    cache.put(key, "answer")

    assert cache.disk_ok is True
    clear_memo()
    assert cache.get(key) == "answer"


def test_unwritable_disk_still_serves_this_run(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    def no_open(self, *args, **kwargs):
        raise OSError("read-only file system")

    cache = DiskCache("t", root=tmp_path)
    key = DiskCache.make_key(prompt="x")
    monkeypatch.setattr(Path, "open", no_open)
    cache.put(key, "answer")

    assert cache.disk_ok is False
    assert cache.get(key) == "answer"       # memo carries it
    assert "not persisting to disk" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def test_selftest_reports_a_healthy_cache(tmp_path):
    result = DiskCache("t", root=tmp_path).selftest()
    assert result["written"] is True
    assert result["readable"] is True


def test_selftest_reads_past_the_memo(tmp_path):
    """It must prove the DISK works, not that the dict works."""
    cache = DiskCache("t", root=tmp_path)
    result = cache.selftest()
    key = DiskCache.make_key(namespace="t", probe="selftest")
    on_disk = json.loads(cache._path(key).read_text())
    assert result["readable"] is True
    assert on_disk["value"] == {"ok": True}


def test_stats_expose_disk_health(tmp_path):
    cache = DiskCache("t", root=tmp_path)
    cache.put(DiskCache.make_key(prompt="x"), "answer")
    stats = cache.stats()
    assert stats["disk_ok"] is True
    assert stats["entries_on_disk"] >= 1
