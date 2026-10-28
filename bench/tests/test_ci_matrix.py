"""Tests for bench/ci/run_matrix.sh using the real harness in dry-run mode."""

import os
import re
import subprocess
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1]
SCRIPT = BENCH_DIR / "ci" / "run_matrix.sh"


def run_matrix(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    """Run the matrix script with a minimal environment and return the finished process."""
    env = {
        "PATH": os.environ["PATH"],
        "HARNESS_DIR": str(BENCH_DIR.parent),
        "CAND_ROOT": str(tmp_path / "cand"),
        "WORK_DIR": str(tmp_path / "work"),
        "RESULTS": str(tmp_path / "results.jsonl"),
        "OUT_ROOT": str(tmp_path / "out"),
        "HARDWARE": "cpu",
        "PYTHONS": "3.12",
        "BENCH_GROUPS": "infer",
        "CANDIDATES": "baseline-conda uv",
        "ROUNDS": "1",
        "DRY_RUN": "1",
    }
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False)


def group_labels(stdout: str) -> list[str]:
    return re.findall(r"^::group::(.+)$", stdout, flags=re.MULTILINE)


def test_dry_run_visits_each_candidate_once_per_round(tmp_path: Path):
    result = run_matrix(tmp_path, ROUNDS="2")

    assert result.returncode == 0
    assert group_labels(result.stdout) == [
        "r0-baseline-conda-py3.12-cpu-infer",
        "r0-uv-py3.12-cpu-infer",
        "r1-uv-py3.12-cpu-infer",
        "r1-baseline-conda-py3.12-cpu-infer",
    ]


def test_dry_run_creates_no_work_directory(tmp_path: Path):
    run_matrix(tmp_path)

    assert not (tmp_path / "work").exists()


def test_covers_every_python_and_group(tmp_path: Path):
    result = run_matrix(tmp_path, PYTHONS="3.12 3.13", BENCH_GROUPS="full infer", CANDIDATES="baseline-conda")

    assert sorted(group_labels(result.stdout)) == [
        "r0-baseline-conda-py3.12-cpu-full",
        "r0-baseline-conda-py3.12-cpu-infer",
        "r0-baseline-conda-py3.13-cpu-full",
        "r0-baseline-conda-py3.13-cpu-infer",
    ]


def test_draft_adapter_is_refused_and_no_cell_runs(tmp_path: Path):
    result = run_matrix(tmp_path, CANDIDATES="uv", DRY_RUN="")

    assert result.returncode == 3
    assert "refused: 1" in result.stdout


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CANDIDATES", "uv; touch pwned"),
        ("CANDIDATES", "no-such-adapter"),
        ("PYTHONS", "3.12 $(id)"),
        ("BENCH_GROUPS", "full;infer"),
        ("HARDWARE", "tpu"),
        ("ROUNDS", "0"),
        ("ASOF_DATE", "yesterday"),
    ],
)
def test_rejects_unvalidated_input_before_running(tmp_path: Path, variable: str, value: str):
    result = run_matrix(tmp_path, **{variable: value})

    assert result.returncode == 2
    assert group_labels(result.stdout) == []
    assert not (tmp_path / "pwned").exists()
