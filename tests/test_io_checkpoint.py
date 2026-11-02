import json

import pytest

from carprice.utils.checkpoint import Manifest, run_unit
from carprice.utils.io import atomic_path, stable_hash, write_json, write_text


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    target = tmp_path / "out.txt"
    write_text(target, "old")
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_text("partial new")
        raise RuntimeError("crash mid-write")
    assert target.read_text() == "old"  # previous complete file intact
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]  # temp file cleaned up


def test_stable_hash_is_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def _manifest(tmp_path):
    return Manifest.load(tmp_path / "artifacts" / "checkpoints" / "manifest.json")


def test_unit_runs_once_then_skips_when_verified(tmp_path):
    m = _manifest(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return [write_json(tmp_path / "out.json", {"v": 1})], {"m": 1}

    r1 = run_unit(m, "stage", fn, "cfg1", root=tmp_path)
    r2 = run_unit(m, "stage", fn, "cfg1", root=tmp_path)
    assert not r1["skipped"] and r2["skipped"] and len(calls) == 1


def test_changed_config_or_corrupt_output_triggers_recompute(tmp_path):
    m = _manifest(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return [write_json(tmp_path / "out.json", {"v": len(calls)})], {}

    run_unit(m, "s", fn, "cfgA", root=tmp_path)
    run_unit(m, "s", fn, "cfgB", root=tmp_path)  # config changed
    assert len(calls) == 2
    (tmp_path / "out.json").write_text("tampered")  # corrupt output
    run_unit(m, "s", fn, "cfgB", root=tmp_path)
    assert len(calls) == 3


def test_missing_output_file_triggers_recompute_and_failure_is_recorded(tmp_path):
    m = _manifest(tmp_path)
    n = []

    def fn():
        n.append(1)
        return [write_json(tmp_path / "o.json", {})], {}

    run_unit(m, "s", fn, "c", root=tmp_path)
    (tmp_path / "o.json").unlink()
    run_unit(m, "s", fn, "c", root=tmp_path)
    assert len(n) == 2

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        run_unit(m, "bad", boom, "c", root=tmp_path)
    assert m.entries["bad"]["status"] == "failed"


def test_manifest_survives_corrupt_file(tmp_path):
    path = tmp_path / "artifacts" / "checkpoints" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    m = Manifest.load(path)
    assert m.entries == {}  # starts fresh instead of crashing
    assert any("corrupt" in p.name for p in path.parent.iterdir())


def test_downstream_input_hash_invalidates(tmp_path):
    m = _manifest(tmp_path)
    run_unit(m, "up", lambda: ([write_json(tmp_path / "u.json", {"a": 1})], {}), "c", root=tmp_path)
    h1 = m.output_hash("up")
    run_unit(m, "down", lambda: ([write_json(tmp_path / "d.json", {})], {}), "c", h1, root=tmp_path)
    ok, _ = m.verify("down", "c", h1)
    assert ok
    ok, why = m.verify("down", "c", "different-upstream")
    assert not ok and "upstream" in why
    assert json.loads((tmp_path / "artifacts/checkpoints/manifest.json").read_text())["entries"]["down"]["status"] == "complete"
