"""Manual garbage collection for the SessionBank SSD cold tier (#493).

``SessionBankColdTier._cleanup_untracked_cache_once`` (``cache_bank/cold_tier.py``)
already reconciles the on-disk store against ``manifest.sqlite`` and deletes
whatever the manifest no longer references -- entry directories, blob files,
``evicted_entries/`` leftovers. It works; the gap is that it only ever runs
as a side effect of a write that finds the store close to its configured
size cap. On a Mac with generous free disk that cap is rarely approached, so
orphans a user never triggers the cap against just accumulate forever: #493
measured 394,155 orphaned blobs (44.1 GB) against 17 live manifest entries
after three weeks of normal use, with no way to reclaim them short of
deleting the whole session bank by hand.

This module reimplements the same reconciliation as a small, standalone,
MLX-free utility rather than importing ``SessionBankColdTier`` directly:
``cache_bank/cold_tier.py`` imports ``cache_bank/codec.py``, which imports
``mlx.core`` at module scope, purely for the tensor encode/decode path this
reconciliation never touches (it only reads ``manifest.sqlite`` and
``payload.json`` files, then deletes what nothing references). Routing a
disk-cleanup command through that import chain would mean ``mtplx gc``
stops working exactly when it might matter most -- a broken or absent MLX
install -- which contradicts the CLI's own "doctor and inspect run on any
machine" contract (see ``mtplx/cli.py``, ``test_no_mlx_imports.py``).

Read-only by default (``dry_run=True``): the CLI surface only deletes with
an explicit ``--apply``, since this walks and can remove tens of thousands
of files.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_COLD_TIER_DIR = Path("~/.mtplx/session-bank").expanduser()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _manifest_entry_dirs(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT entry_dir FROM entries").fetchall()
    return {str(row["entry_dir"]) for row in rows}


def _entry_blob_hashes(entry_dir: Path) -> set[str]:
    payload_path = entry_dir / "payload.json"
    if not payload_path.exists():
        return set()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    tensor_blobs = payload.get("tensor_blobs") or {}
    hashes: set[str] = set()
    if isinstance(tensor_blobs, dict):
        for blob in tensor_blobs.values():
            if isinstance(blob, dict) and blob.get("sha256"):
                hashes.add(str(blob["sha256"]))
    return hashes


def _manifest_blob_hashes(conn: sqlite3.Connection, base_dir: Path) -> set[str]:
    rows = conn.execute("SELECT entry_dir FROM entries").fetchall()
    hashes: set[str] = set()
    for row in rows:
        hashes.update(_entry_blob_hashes(base_dir / str(row["entry_dir"])))
    return hashes


def _dir_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def collect_garbage(
    base_dir: Path | str = DEFAULT_COLD_TIER_DIR, *, dry_run: bool = True
) -> dict[str, Any]:
    """Reconcile ``base_dir`` against its manifest; delete what dry_run allows.

    Mirrors ``SessionBankColdTier._cleanup_untracked_cache_once`` exactly:
    an entry directory not listed in the manifest is orphaned, a blob file
    whose digest no entry's ``payload.json`` references is orphaned, and
    everything under ``evicted_entries/`` is a stale archive copy (see
    ``_archive_orphan_entry_dir``) safe to remove outright. Returns a report
    dict with counts and bytes; nothing is deleted unless ``dry_run=False``.

    A concurrently running server writes new entries under the same lock
    this reconciliation does not hold across processes -- run this only
    while no server is serving this session bank, or a session committed
    between the scan and the delete could have its blobs removed. The CLI
    wrapper checks for a running server and requires ``--force`` to proceed
    anyway.
    """
    base_dir = Path(base_dir).expanduser()
    db_path = base_dir / "manifest.sqlite"
    report: dict[str, Any] = {
        "base_dir": str(base_dir),
        "dry_run": bool(dry_run),
        "manifest_found": db_path.exists(),
        "orphan_entry_dirs": [],
        "orphan_blob_files": 0,
        "orphan_blob_bytes": 0,
        "evicted_entries_bytes": 0,
        "deleted": False,
        "elapsed_s": 0.0,
    }
    if not base_dir.exists():
        return report
    started = time.perf_counter()

    manifest_entry_dirs: set[str] = set()
    manifest_blob_hashes: set[str] = set()
    if db_path.exists():
        conn = _connect(db_path)
        try:
            manifest_entry_dirs = _manifest_entry_dirs(conn)
            manifest_blob_hashes = _manifest_blob_hashes(conn, base_dir)
        finally:
            conn.close()

    evicted_root = base_dir / "evicted_entries"
    if evicted_root.exists():
        report["evicted_entries_bytes"] = _dir_bytes(evicted_root)
        if not dry_run:
            import shutil

            shutil.rmtree(evicted_root, ignore_errors=True)

    entries_root = base_dir / "entries"
    if entries_root.exists():
        for prefix_dir in entries_root.iterdir():
            if not prefix_dir.is_dir():
                continue
            for entry_dir in prefix_dir.iterdir():
                if not entry_dir.is_dir():
                    continue
                rel = str(entry_dir.relative_to(base_dir))
                if rel in manifest_entry_dirs:
                    continue
                report["orphan_entry_dirs"].append(rel)
                if not dry_run:
                    import shutil

                    shutil.rmtree(entry_dir, ignore_errors=True)

    blobs_root = base_dir / "blobs"
    if blobs_root.exists():
        for prefix_dir in blobs_root.iterdir():
            if not prefix_dir.is_dir():
                continue
            for path in prefix_dir.glob("*.bin"):
                digest = path.stem
                if digest in manifest_blob_hashes:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                report["orphan_blob_files"] += 1
                report["orphan_blob_bytes"] += size
                if not dry_run:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

    report["deleted"] = not dry_run
    report["elapsed_s"] = time.perf_counter() - started
    return report
