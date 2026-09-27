"""
Cache for every LLM and embedding call.

This is the most important file in the project. Our entire experiment budget is
two free API tiers; the full study is roughly 7,000 calls. That fits -- but only
if a re-run costs nothing.

TWO LAYERS, because one is not enough:

  * In-process memo. Guarantees that repeating a call inside one run never
    spends quota twice, no matter what the filesystem does.
  * Disk. Carries results across Colab restarts.

The two-layer design is not belt-and-braces for its own sake. Our first Colab
run wrote the cache to a mounted Google Drive folder and every read came back a
miss: Drive's FUSE layer does not reliably make a just-written file visible to
an immediate `exists()` check, and it can reject the atomic rename that a
careful writer uses. On disk-only that silently doubles every cost. So writes
now fall back from rename to a direct write, verify themselves, and report
honestly via `disk_ok` instead of failing quietly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

_LOCK = threading.Lock()

# Shared across every DiskCache pointing at the same root, so two clients in one
# process (generator and judge, say) never re-fetch each other's results.
_MEMO: dict[tuple[str, str], Any] = {}

_WARNED: set[str] = set()


def _default_cache_dir() -> Path:
    return Path(os.environ.get("BAPCA_CACHE_DIR", "./cache")).expanduser()


class DiskCache:
    def __init__(self, namespace: str, root: Optional[Path] = None):
        self.namespace = namespace
        self.root = (root or _default_cache_dir()) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.memo_hits = 0
        self.disk_ok: Optional[bool] = None   # None until we have tried a write
        self.disk_error: Optional[str] = None

    # ---- keys ------------------------------------------------------------

    @staticmethod
    def make_key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        shard = self.root / key[:2]
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key}.json"

    def _memo_key(self, key: str) -> tuple[str, str]:
        return (str(self.root), key)

    # ---- read / write ----------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        memo_key = self._memo_key(key)
        if memo_key in _MEMO:
            self.hits += 1
            self.memo_hits += 1
            return _MEMO[memo_key]

        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                value = json.load(fh)["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A truncated file (Colab runtime killed mid-write) is a miss,
            # never a crash.
            self.misses += 1
            return None

        _MEMO[memo_key] = value
        self.hits += 1
        return value

    def _write_file(self, path: Path, payload: dict) -> None:
        """Atomic rename where possible; plain write where the FS refuses it."""
        try:
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            tmp.replace(path)
        except OSError:
            # Google Drive's FUSE mount can reject rename-over. A direct write
            # risks a torn file on a hard kill, which get() already treats as a
            # miss -- an acceptable trade for a cache that actually persists.
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)

    def put(self, key: str, value: Any, meta: Optional[dict] = None) -> None:
        # The memo is set first and unconditionally: even if the disk is
        # unusable, this run must not pay twice for the same call.
        _MEMO[self._memo_key(key)] = value

        payload = {"value": value, "meta": meta or {}}
        path = self._path(key)
        with _LOCK:
            try:
                self._write_file(path, payload)
                self.disk_ok = path.exists()
                if not self.disk_ok:
                    self.disk_error = "file not visible immediately after write"
            except OSError as exc:
                self.disk_ok = False
                self.disk_error = str(exc)

        if self.disk_ok is False and self.namespace not in _WARNED:
            _WARNED.add(self.namespace)
            print(
                f"WARNING: cache '{self.namespace}' is not persisting to disk "
                f"({self.disk_error}).\n"
                f"  Path: {self.root}\n"
                "  This run is still safe (results are held in memory), but they "
                "will be lost on restart\n"
                "  and re-fetching them will cost API quota. Point BAPCA_CACHE_DIR "
                "at local storage."
            )

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Any],
        meta: Optional[dict] = None,
    ) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value, meta=meta)
        return value

    # ---- diagnostics -----------------------------------------------------

    def selftest(self) -> dict:
        """
        Write a value and read it back off disk, bypassing the memo entirely.

        Run this before a long experiment. A cache that looks fine in memory but
        never lands on disk turns a one-off cost into a cost per session, and
        you find out on day four when the quota runs dry.
        """
        key = DiskCache.make_key(namespace=self.namespace, probe="selftest")
        self.put(key, {"ok": True})
        path = self._path(key)

        result = {"path": str(self.root), "written": path.exists(),
                  "readable": False, "error": self.disk_error}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    result["readable"] = json.load(fh)["value"] == {"ok": True}
            except Exception as exc:  # noqa: BLE001
                result["error"] = str(exc)
        return result

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "namespace": self.namespace,
            "hits": self.hits,
            "memo_hits": self.memo_hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "disk_ok": self.disk_ok,
            "entries_on_disk": sum(1 for _ in self.root.rglob("*.json")),
        }


def clear_memo() -> None:
    """Drop the in-process layer. For tests that need to exercise disk reads."""
    _MEMO.clear()
