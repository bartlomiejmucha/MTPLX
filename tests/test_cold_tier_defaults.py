"""SSD cold-tier defaults accepted from PR #496 (Dizzler7).

The RAM-tiered SSD cap stays 32 GiB on a 64 GB Mac unless the disk has room
to spare (>= 150 GiB free lifts it to the 100 GiB big-machine default); the
effective cap is still min(cap, free/4) at write time, so the lift only
matters on disks with >= 400 GiB free. The hourly SSD write budget default
is 128 GiB (was 64): a deep multi-turn session dedupes most blocks, but the
first snapshot of several long conversations in one hour could exceed 64.
"""

from __future__ import annotations

import os
from collections import namedtuple

from mtplx.cache_bank import cold_tier

GIB = 1024**3
_Usage = namedtuple("_Usage", "total used free")


def _pin(monkeypatch, *, ram_gib: int, free_gib: int) -> None:
    monkeypatch.setattr(cold_tier, "detect_total_ram_bytes", lambda: ram_gib * GIB)
    monkeypatch.setattr(
        cold_tier.shutil,
        "disk_usage",
        lambda _path: _Usage(2000 * GIB, (2000 - free_gib) * GIB, free_gib * GIB),
    )


def test_64gb_mac_keeps_32gib_cap_on_a_tight_disk(monkeypatch):
    _pin(monkeypatch, ram_gib=64, free_gib=120)
    assert cold_tier.default_cold_tier_max_bytes() == 32 * GIB


def test_64gb_mac_gets_the_big_cap_with_150gib_free(monkeypatch):
    _pin(monkeypatch, ram_gib=64, free_gib=150)
    assert cold_tier.default_cold_tier_max_bytes() == cold_tier.DEFAULT_COLD_TIER_MAX_BYTES


def test_64gb_mac_keeps_32gib_when_the_disk_cannot_be_read(monkeypatch):
    monkeypatch.setattr(cold_tier, "detect_total_ram_bytes", lambda: 64 * GIB)

    def boom(_path):
        raise OSError("no statvfs")

    monkeypatch.setattr(cold_tier.shutil, "disk_usage", boom)
    assert cold_tier.default_cold_tier_max_bytes() == 32 * GIB


def test_smaller_tiers_are_unchanged(monkeypatch):
    _pin(monkeypatch, ram_gib=32, free_gib=900)
    assert cold_tier.default_cold_tier_max_bytes() == 24 * GIB
    _pin(monkeypatch, ram_gib=16, free_gib=900)
    assert cold_tier.default_cold_tier_max_bytes() == 16 * GIB


def test_hourly_write_budget_default_is_128gib(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_SSD_WRITE_BUDGET_PER_HOUR", raising=False)
    tier = cold_tier.SessionBankColdTier(base_dir=tmp_path / "bank", mode="on")
    try:
        assert tier.stats()["write_budget_per_hour_bytes"] == 128 * GIB
    finally:
        tier.close()
    assert os.environ.get("MTPLX_SSD_WRITE_BUDGET_PER_HOUR") is None
