import json, math
import pytest
from paper import export, store
from paper.paper_filter import apply_filters, preset_filters
from paper.replay import settled_pool
from test_pipeline import seed_day


def test_exports_are_valid_strict_json_and_shards_round_trip(tmp_path):
    seed_day(tmp_path, "2026-11-24", closing_line=24.5)
    out = export.export_site(tmp_path)
    summary = json.loads((out / "summary.json").read_text())          # allow_nan=False: no NaN/Infinity
    index = json.loads((out / "pool" / "index.json").read_text())
    shard = json.loads((out / "pool" / "2026-11-24.json").read_text())
    pool = settled_pool(tmp_path)[0]
    bettable_slots = [r for r in pool if r["slot"] != "closing"]
    assert index["shards"] == [{"date": "2026-11-24", "rows": len(bettable_slots), "graded": True}]
    assert len(shard["rows"]) == len(bettable_slots) > 0 and summary["candidates"] == len(pool)
    assert all(len(r) == len(shard["columns"]) for r in shard["rows"])
    assert "closing" not in shard["tables"]["slots"]                   # reference-only boards are not shipped
    assert {p["id"] for p in summary["personas"]} >= {"firehose", "shopper", "sharp"}
    assert [p["id"] for p in summary["presets"]] == ["realistic", "sharp", "ladder", "everything"]
    assert summary["defaults"]["min_edge"] == 0.03 and summary["latestDate"] == "2026-11-24"
    cols = {c: i for i, c in enumerate(shard["columns"])}
    src, row = bettable_slots[0], shard["rows"][0]
    assert abs(row[cols["implied"]] / 10000 - src["implied"]) < 1e-4   # packed x10000, restores within rounding
    assert shard["tables"]["players"][row[cols["player"]]]["id"] == src["player_id"]


def test_a_finished_days_shard_is_byte_identical_on_every_later_run(tmp_path):
    """Otherwise the data branch would re-commit the whole history every 15 minutes."""
    seed_day(tmp_path, "2026-11-23", closing_line=24.5)
    out = export.export_site(tmp_path)
    first = (out / "pool" / "2026-11-23.json").read_bytes()
    seed_day(tmp_path, "2026-11-24", closing_line=24.5)
    export.export_site(tmp_path)
    assert (out / "pool" / "2026-11-23.json").read_bytes() == first


def test_nan_and_inf_never_reach_the_json():
    assert export.clean({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -float("inf")}}) == \
        {"a": None, "b": [None, 1.0], "c": {"d": None}}


def test_personas_carry_an_equity_curve_and_latest_bets(tmp_path):
    seed_day(tmp_path, "2026-11-24", closing_line=24.5)
    s = json.loads((export.export_site(tmp_path) / "summary.json").read_text())
    fh = next(p for p in s["personas"] if p["id"] == "firehose")
    assert fh["equity"] and fh["equity"][-1]["date"] == "2026-11-24" and fh["recent"]
    assert fh["equity"][-1]["pnl"] == pytest.approx(fh["pnl"], abs=1e-3)
