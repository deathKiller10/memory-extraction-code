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


# ---- 8 Nov: one Drive round trip per read, and parallel batch reads ----------

def test_get_many_equals_get_in_a_loop(tmp_path):
    from bapca.cache import DiskCache, clear_memo
    cache = DiskCache("t", root=tmp_path)
    keys = [DiskCache.make_key(i=i) for i in range(40)]
    for i, k in enumerate(keys[:30]):
        cache.put(k, {"v": i})
    torn = cache._path(keys[30], create=True)
    torn.write_text('{"value": ', encoding="utf-8")              # killed mid-write
    clear_memo()
    one = DiskCache("t", root=tmp_path)
    loop = [one.get(k) for k in keys]
    clear_memo()
    many = DiskCache("t", root=tmp_path)
    batch = many.get_many(keys)
    assert batch == loop
    assert batch[:30] == [{"v": i} for i in range(30)] and batch[30:] == [None] * 10
    assert (many.hits, many.misses) == (one.hits, one.misses) == (30, 10)
    again = many.get_many(keys)                                   # now from memory
    assert again == batch and many.memo_hits == 30


def test_a_read_creates_no_folders(tmp_path):
    from bapca.cache import DiskCache, clear_memo
    cache = DiskCache("t", root=tmp_path)
    clear_memo()
    assert cache.get(DiskCache.make_key(x=1)) is None
    assert cache.get_many([DiskCache.make_key(x=2)]) == [None]
    assert list((tmp_path / "t").iterdir()) == []                 # nothing made on read
    cache.put(DiskCache.make_key(x=1), 5)
    clear_memo()
    assert cache.get(DiskCache.make_key(x=1)) == 5


def test_embedder_reads_its_cache_in_one_batch_and_matches_a_fresh_encode(tmp_path, monkeypatch):
    """The real Embedder path (model stubbed): fresh vectors are cached, and a
    later batch read returns exactly them, in order, without the model."""
    import numpy as np
    from bapca import cache as cache_mod
    from bapca.cache import clear_memo
    from bapca.embeddings import Embedder
    monkeypatch.setenv("BAPCA_CACHE_DIR", str(tmp_path))

    class Model:
        calls = 0
        def encode(self, texts, **kw):
            Model.calls += 1
            return np.array([[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts])

    emb = Embedder()
    monkeypatch.setattr(emb, "_load", lambda: Model())
    texts = ["alpha", "beta gamma", "alpha", "delta"]
    first = emb.encode(texts)
    assert Model.calls == 1
    clear_memo()
    again = Embedder()
    monkeypatch.setattr(again, "_load", lambda: (_ for _ in ()).throw(AssertionError("model used")))
    second = again.encode(texts)
    assert np.allclose(first, second) and second.shape == (4, 3)
