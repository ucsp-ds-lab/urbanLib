"""
Tests for the CLI's path helper functions.

These helpers can be tested without a GPU because operator imports are
deferred inside each command function rather than happening at module load.
"""
import os

import pytest

from urbanLib.cli import (
    _city_slug,
    _data_dir,
    _ensure_streets_file,
    _infer_city_center,
    _output_dir,
    _parse_modes,
    _streets_file,
)

_PRESETS = {"fast": 0.0, "balanced": 0.5, "safe": 1.0}


def test_output_dir_uses_city_as_subfolder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _output_dir("chicago")
    assert path == os.path.join("output", "chicago")
    assert os.path.isdir(path)  # _output_dir also creates the directory


def test_data_dir_uses_city_as_subfolder():
    assert _data_dir("manhattan") == os.path.join("data", "manhattan")


def test_streets_file_path():
    assert _streets_file("chicago") == os.path.join("data", "chicago", "streets.csv")


def test_city_slug_from_full_place_name():
    assert _city_slug("Manhattan, New York, USA") == "manhattan"


def test_city_slug_from_bare_slug_is_idempotent():
    # A slug that already went through _city_slug once must map to itself,
    # so a cached city keeps resolving to the same path on later runs.
    assert _city_slug("manhattan") == "manhattan"


def _fake_get_city_streets(calls):
    def fn(city_name, output_csv, network_type):
        calls.append((city_name, output_csv, network_type))
        with open(output_csv, "w") as f:
            f.write("Node1_ID,Node1_Latitude,Node1_Longitude,Node2_ID,Node2_Latitude,Node2_Longitude\n")
    return fn


def test_ensure_streets_file_fetches_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("urbanLib.network.get_city_streets", _fake_get_city_streets(calls))

    slug, path = _ensure_streets_file("Manhattan, New York, USA")

    assert slug == "manhattan"
    assert path == os.path.join("data", "manhattan", "streets.csv")
    assert os.path.exists(path)
    assert len(calls) == 1


def test_ensure_streets_file_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("urbanLib.network.get_city_streets", _fake_get_city_streets(calls))

    _ensure_streets_file("Manhattan, New York, USA")
    _ensure_streets_file("Manhattan, New York, USA")

    assert len(calls) == 1  # the second call reused the cached file


def test_ensure_streets_file_works_with_bare_slug_if_already_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("data", "manhattan"), exist_ok=True)
    with open(os.path.join("data", "manhattan", "streets.csv"), "w") as f:
        f.write("header\n")

    def fail_if_called(*a, **kw):
        raise AssertionError("should not fetch when the cache already exists")

    monkeypatch.setattr("urbanLib.network.get_city_streets", fail_if_called)

    slug, path = _ensure_streets_file("manhattan")
    assert slug == "manhattan"
    assert os.path.exists(path)


def test_ensure_streets_file_exits_cleanly_on_fetch_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def raise_error(*a, **kw):
        raise RuntimeError("city not found")

    monkeypatch.setattr("urbanLib.network.get_city_streets", raise_error)

    with pytest.raises(SystemExit):
        _ensure_streets_file("Nonexistent Place, Nowhere")

    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_infer_city_center_averages_cached_network_nodes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("data", "chicago"), exist_ok=True)
    with open(os.path.join("data", "chicago", "streets.csv"), "w", newline="") as f:
        f.write(
            "Node1_ID,Node1_Latitude,Node1_Longitude,Node2_ID,Node2_Latitude,Node2_Longitude\n"
            "1,41.0,-87.0,2,42.0,-88.0\n"
            "2,42.0,-88.0,3,43.0,-89.0\n"
        )

    lat, lon = _infer_city_center("chicago")

    assert lat == pytest.approx((41.0 + 42.0) / 2)
    assert lon == pytest.approx((-87.0 + -88.0) / 2)


def test_infer_city_center_falls_back_when_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _infer_city_center("nowhere") == (40.754, -73.984)


def test_parse_modes_accepts_presets_and_float_alpha():
    assert _parse_modes("fast,0.3,safe", _PRESETS) == ["fast", 0.3, "safe"]


def test_parse_modes_rejects_unknown_token(capsys):
    with pytest.raises(SystemExit):
        _parse_modes("quick", _PRESETS)
    assert "invalid --modes value 'quick'" in capsys.readouterr().out


def test_parse_modes_rejects_out_of_range_alpha(capsys):
    with pytest.raises(SystemExit):
        _parse_modes("1.5", _PRESETS)
    assert "invalid --modes value '1.5'" in capsys.readouterr().out
