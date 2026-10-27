"""Tests for bench/analyze.py using synthetic result records."""

import json


def make_record(
    candidate="uv", host="h1", digest="aaa", cold=100.0, status="ok", asof=None
):
    record = {
        "candidate": candidate,
        "status": status,
        "failed_phase": None if status == "ok" else "install_cold",
        "cell": {"id": "mac-cpu", "python": "3.12", "group": "full"},
        "installer": {"selected": None},
        "host": {"hostname_hash": host, "system": "Darwin", "machine": "arm64"},
        "env": {
            "size_bytes": 2048 * 1024 * 1024,
            "n_packages": 300,
            "package_set_hash": digest,
            "python_dist_hash": "pp",
        },
        "phases": {
            "install_cold": {
                "status": "ok",
                "seconds": [cold],
                "cache_size_bytes": {"/c": 1024 * 1024 * 1024},
            },
            "import": {"status": "ok", "seconds": [1.0, 2.0, 3.0]},
            "unit_tests": {
                "status": "ok",
                "seconds": [50.0],
                "pytest": {"passed": 90, "failed": 1, "errors": 0, "skipped": 4},
            },
        },
    }
    if status == "failed":
        record["phases"]["install_cold"]["status"] = "failed"
    if asof:
        record["phases"]["resolve_asof"] = {
            "status": "failed",
            "tail": ["error: no solution"],
            "asof": asof,
        }
    return record


def test_render_reports_median_min_max_and_success_rate(analyze):
    records = [
        make_record(cold=100.0),
        make_record(cold=200.0),
        make_record(cold=300.0, status="failed"),
    ]

    text = analyze.render(records)

    assert "## mac-cpu, Python 3.12, group full" in text
    assert "| install_cold | 150.0 (100.0-200.0) |" in text
    assert "| import | 2.0 (1.0-3.0) |" in text
    assert "| uv | 3 | 2 | 1 | 0 | 67% | install_cold |" in text
    assert "| uv | 2048 | 1024 | 300 |" in text
    assert "| uv | unit_tests | 90 | 1 | 0 | 4 |" in text


def test_reproducibility_flags_differing_hashes_across_hosts(analyze):
    records = [
        make_record(host="h1", digest="aaa"),
        make_record(host="h1", digest="aaa"),
        make_record(host="h2", digest="bbb"),
    ]

    text = analyze.reproducibility_table({"uv": records})

    assert "h1: identical (n=2)" in text
    assert "Darwin/arm64: DIFFERENT (2 hashes in 3) over 2 host(s)" in text


def test_reproducibility_reports_identical_hashes(analyze):
    records = [make_record(host="h1"), make_record(host="h2")]

    text = analyze.reproducibility_table({"uv": records})

    assert "Darwin/arm64: identical (n=2) over 2 host(s)" in text


def test_asof_table_shows_failure_detail(analyze):
    text = analyze.asof_table({"uv": [make_record(asof="2026-04-27")]})

    assert (
        "| uv | - | failed | - | error: no solution |" in text
        or "error: no solution" in text
    )


def test_load_records_skips_malformed_lines(analyze, tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(make_record()) + "\n\nnot json\n")

    records = analyze.load_records([path])

    assert len(records) == 1
    assert "warning" in capsys.readouterr().err


def test_main_filters_by_candidate_and_prints_markdown(analyze, tmp_path, capsys):
    path = tmp_path / "r.jsonl"
    path.write_text(
        "\n".join(json.dumps(make_record(c)) for c in ("uv", "pixi")) + "\n"
    )

    assert analyze.main([str(path), "--candidate", "pixi"]) == 0
    out = capsys.readouterr().out

    assert "pixi" in out and "| uv |" not in out
