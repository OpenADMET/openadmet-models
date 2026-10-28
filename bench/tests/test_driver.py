"""Tests for bench/run.py using fake adapters whose commands are echo, true, and mkdir."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1]

FAKE_ADAPTER = """
schema = 1
name = "fake"
status = "ready"
tool_version = ["echo fake-tool 1.0"]
run_prefix = "env"
package_list = "echo Foo_Bar==1.0"
clear_cache = ["touch {work_dir}/cleared-marker"]
cache_dirs = ["{cache_dir}"]
lock_asof = "echo asof {date}"

[isolated_env]
FAKE_CACHE = "{cache_dir}/x"

[install.any.any]
note = "fake install"
steps = ["mkdir -p {env_dir}/bin", "touch {env_dir}/bin/marker"]
"""


def write_adapter(
    tmp_path: Path, body: str = FAKE_ADAPTER, label: str = "fake"
) -> Path:
    path = tmp_path / f"{label}.toml"
    path.write_text(body)
    return path


@pytest.fixture
def mac_host(run, monkeypatch):
    host = run.HostInfo(system="Darwin", machine="arm64", has_nvidia_smi=False)
    monkeypatch.setattr(run, "detect_host", lambda: host)
    return host


@pytest.fixture
def repo_dir(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return repo


def cli_args(tmp_path: Path, adapter: Path, repo: Path, *extra: str) -> list[str]:
    return [
        "--adapter", str(adapter),
        "--repo", str(repo),
        "--work-dir", str(tmp_path / "work"),
        "--out-dir", str(tmp_path / "out"),
        "--results", str(tmp_path / "results.jsonl"),
        "--hardware", "cpu",
        *extra,
    ]  # fmt: skip


@pytest.fixture
def python_on_path(tmp_path, monkeypatch):
    """Expose the running interpreter as `python` so `env python` works in fake runs."""
    bindir = tmp_path / "pybin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


# Adapter parsing and validation ------------------------------------------------


@pytest.mark.parametrize("label", ["baseline-conda", "conda-lock", "pixi", "uv"])
def test_shipped_adapters_validate(run, label):
    adapter = run.load_adapter(BENCH_DIR / "adapters" / f"{label}.toml")

    assert adapter.name == label
    assert adapter.status in {"ready", "draft"}


def test_baseline_selects_gpu_yaml_only_for_cuda(run):
    adapter = run.load_adapter(BENCH_DIR / "adapters" / "baseline-conda.toml")

    cuda = adapter.install_for("cuda", "full").steps[0].template
    cpu = adapter.install_for("cpu", "full").steps[0].template
    mps = adapter.install_for("mps", "infer").steps[0].template

    assert "openadmet-models-gpu.yaml" in cuda
    assert "openadmet-models.yaml" in cpu and "gpu" not in cpu
    assert mps == cpu


def test_install_resolution_prefers_the_most_specific_entry(run, tmp_path):
    body = FAKE_ADAPTER + '\n[install.cpu.full]\nsteps = ["echo specific"]\n'
    adapter = run.load_adapter(write_adapter(tmp_path, body))

    assert adapter.install_for("cpu", "full").steps[0].template == "echo specific"
    assert adapter.install_for("cpu", "infer").note == "fake install"


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.replace('name = "fake"', 'name = "other"'), "file stem"),
        (lambda s: s.replace('status = "ready"', 'status = "maybe"'), "status must be"),
        (lambda s: "bogus = 1\n" + s, "unknown top-level keys"),
        (
            lambda s: s.replace("echo fake-tool 1.0", "echo {nonsense}"),
            "unknown placeholder",
        ),
        (lambda s: s.replace("echo fake-tool 1.0", "TODO: later"), "TODO"),
        (lambda s: s.replace('run_prefix = "env"\n', ""), "run_prefix is required"),
        (lambda s: s.replace("schema = 1", "schema = 2"), "schema must be 1"),
        (
            lambda s: s.replace("[install.any.any]", "[install.tpu.any]"),
            "hardware must be",
        ),
        (lambda s: s.replace("{date}", "{installer}"), "unknown placeholder"),
    ],
)
def test_ready_adapter_rejects_invalid_content(run, tmp_path, edit, message):
    path = write_adapter(tmp_path, edit(FAKE_ADAPTER))

    with pytest.raises(run.AdapterError, match=message):
        run.load_adapter(path)


def test_group_alias_is_loaded_and_rejects_unknown_group(run, tmp_path):
    good = FAKE_ADAPTER + '\n[group_aliases]\nfull = "dev"\n'
    bad = FAKE_ADAPTER + '\n[group_aliases]\nnightly = "dev"\n'

    adapter = run.load_adapter(write_adapter(tmp_path, good))

    assert adapter.group_aliases == {"full": "dev"}
    with pytest.raises(run.AdapterError, match="group must be one of"):
        run.load_adapter(write_adapter(tmp_path, bad))


def test_draft_adapter_accepts_todo_placeholders(run, tmp_path):
    body = (
        'schema = 1\nname = "fake"\nstatus = "draft"\n'
        'tool_version = ["TODO: print version"]\n'
        '[install.any.any]\nsteps = ["TODO: install {env_dir}"]\n'
    )

    adapter = run.load_adapter(write_adapter(tmp_path, body))

    assert adapter.is_draft


# Placeholder expansion -----------------------------------------------------------


def test_expand_argv_keeps_paths_with_spaces_as_one_argument(run):
    context = {"env_dir": "/tmp/with space/env", "python": "3.12"}

    argv = run.expand_argv("tool create -p {env_dir} python={python}", context)

    assert argv == ["tool", "create", "-p", "/tmp/with space/env", "python=3.12"]


def test_expand_argv_rejects_unknown_placeholder(run):
    with pytest.raises(run.AdapterError, match="cannot expand"):
        run.expand_argv("tool {missing}", {"python": "3.12"})


def test_expand_argv_supports_literal_braces(run):
    assert run.expand_argv("echo '{{x}}' {python}", {"python": "3.13"}) == [
        "echo",
        "{x}",
        "3.13",
    ]


# Refusal rules ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "machine", "smi", "hardware", "refused"),
    [
        ("Darwin", "arm64", False, "cpu", False),
        ("Darwin", "arm64", False, "mps", False),
        ("Darwin", "arm64", False, "cuda", True),
        ("Darwin", "x86_64", False, "mps", True),
        ("Linux", "x86_64", False, "cpu", False),
        ("Linux", "x86_64", True, "cuda", False),
        ("Linux", "x86_64", False, "cuda", True),
        ("Linux", "x86_64", True, "mps", True),
        ("Windows", "AMD64", False, "cpu", True),
    ],
)
def test_cell_refusal_matrix(run, system, machine, smi, hardware, refused):
    host = run.HostInfo(system=system, machine=machine, has_nvidia_smi=smi)

    assert (run.cell_refusal(host, hardware) is not None) is refused


def test_cuda_on_mac_is_refused_by_the_cli(run, mac_host, tmp_path, repo_dir, capsys):
    args = cli_args(tmp_path, write_adapter(tmp_path), repo_dir, "--dry-run")
    args[args.index("cpu")] = "cuda"

    code = run.main(args)

    assert code == 2
    assert "cuda cells do not exist on macOS" in capsys.readouterr().err


def test_list_cells_marks_other_os_cells_unavailable(run, mac_host, capsys):
    assert run.main(["--list-cells"]) == 0
    out = capsys.readouterr().out

    assert "mac-cpu     cpu       yes" in out
    assert "linux-cuda  cuda      no         needs a linux host" in out


def test_unit_tests_phase_is_refused_for_infer_group(run):
    with pytest.raises(run.RefusalError, match="group 'full' only"):
        run.resolve_phases("unit_tests", "infer")


def test_draft_adapter_cannot_run_for_real(run, mac_host, tmp_path, repo_dir, capsys):
    draft = (
        'schema = 1\nname = "fake"\nstatus = "draft"\n'
        '[install.any.any]\nsteps = ["TODO: install"]\n'
    )
    args = cli_args(tmp_path, write_adapter(tmp_path, draft), repo_dir)

    assert run.main(args) == 2
    assert "draft" in capsys.readouterr().err


def test_global_cache_mode_requires_confirmation_flag(
    run, mac_host, tmp_path, repo_dir, capsys
):
    args = cli_args(
        tmp_path, write_adapter(tmp_path), repo_dir, "--cache-mode", "global"
    )

    assert run.main(args) == 2
    assert "--yes-clear-cache" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


# Filesystem safety -------------------------------------------------------------------


@pytest.fixture
def work_dir(run, tmp_path) -> Path:
    path = tmp_path / "work"
    run.ensure_work_dir(path)
    return path.resolve()


def test_safe_rmtree_removes_directory_inside_work_dir(run, work_dir):
    target = work_dir / "envs" / "one"
    target.mkdir(parents=True)
    (target / "file").write_text("x")

    assert run.safe_rmtree(target, work_dir) is True
    assert not target.exists()


@pytest.mark.parametrize("relative", ["../outside", "."])
def test_safe_rmtree_refuses_outside_path_and_work_dir_itself(run, work_dir, relative):
    outside = work_dir.parent / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")

    with pytest.raises(run.RefusalError):
        run.safe_rmtree(work_dir / relative, work_dir)

    assert (outside / "keep.txt").exists()
    assert work_dir.exists()


def test_safe_rmtree_unlinks_symlink_without_touching_its_target(
    run, work_dir, tmp_path
):
    target = tmp_path / "precious"
    target.mkdir()
    (target / "data").write_text("x")
    (work_dir / "link").symlink_to(target)

    run.safe_rmtree(work_dir / "link", work_dir)

    assert not (work_dir / "link").exists()
    assert (target / "data").read_text() == "x"


def test_safe_rmtree_refuses_symlink_whose_parent_is_outside(run, work_dir, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "link").symlink_to(work_dir)

    with pytest.raises(run.RefusalError):
        run.safe_rmtree(outside / "link", work_dir)


def test_safe_rmtree_refuses_work_dir_without_marker(run, tmp_path):
    bare = tmp_path / "bare"
    (bare / "child").mkdir(parents=True)

    with pytest.raises(run.RefusalError, match="marker"):
        run.safe_rmtree(bare / "child", bare)

    assert (bare / "child").exists()


def test_validate_work_dir_refuses_broad_overlapping_and_foreign_directories(
    run, tmp_path, repo_dir
):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "data").write_text("x")

    for bad in (Path("/"), Path.home(), tmp_path, repo_dir / "inside", foreign):
        with pytest.raises(run.RefusalError):
            run.validate_work_dir(bad, repo_dir)


# Dry run -------------------------------------------------------------------------------


def snapshot(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_dry_run_prints_plan_and_has_no_side_effects(
    run, mac_host, tmp_path, repo_dir, capsys, monkeypatch
):
    adapter = write_adapter(tmp_path)
    before = snapshot(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run must not start a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)

    code = run.main(
        cli_args(
            tmp_path,
            adapter,
            repo_dir,
            "--dry-run",
            "--phases",
            "default,resolve_asof",
            "--asof",
            "2026-04-27",
        )
    )
    out = capsys.readouterr().out

    assert code == 0
    assert snapshot(tmp_path) == before
    assert "DRY RUN" in out
    assert "DESTRUCTIVE: remove isolated cache dir" in out
    assert "DESTRUCTIVE: remove env dir" in out
    assert "asof 2026-04-27" in out
    assert f"cache dir : {(tmp_path / 'work').resolve()}" in out


def test_dry_run_global_mode_lists_clear_cache_command(
    run, mac_host, tmp_path, repo_dir, capsys
):
    args = cli_args(
        tmp_path,
        write_adapter(tmp_path),
        repo_dir,
        "--dry-run",
        "--cache-mode",
        "global",
    )

    assert run.main(args) == 0
    assert (
        "clear_cache (GLOBAL, needs --yes-clear-cache): touch"
        in capsys.readouterr().out
    )


def test_destructive_actions_refuse_targets_outside_work_dir(
    run, mac_host, tmp_path, repo_dir
):
    adapter = run.load_adapter(write_adapter(tmp_path))
    opts = run.Options(
        candidate="fake", adapter_path=adapter.path, python="3.12", hardware="cpu", group="full",
        repo=repo_dir, work_dir=(tmp_path / "work").resolve(), out_dir=tmp_path / "out",
        results=tmp_path / "r.jsonl", phases=("install_cold",),
    )  # fmt: skip
    state = run.build_state(opts, adapter, mac_host, "20260101T000000Z-abc123")
    state.env_dir = tmp_path / "somewhere-else"

    with pytest.raises(run.RefusalError, match="not inside work dir"):
        run.destructive_actions(state, "install_cold")


# Command construction ----------------------------------------------------------------------


def make_state(run, host, tmp_path, repo_dir, group="full", hardware="cpu"):
    adapter = run.load_adapter(write_adapter(tmp_path))
    opts = run.Options(
        candidate="fake", adapter_path=adapter.path, python="3.12", hardware=hardware, group=group,
        repo=repo_dir, work_dir=(tmp_path / "work").resolve(), out_dir=tmp_path / "out",
        results=tmp_path / "r.jsonl", phases=run.resolve_phases("default", group),
    )  # fmt: skip
    return run.build_state(opts, adapter, host, "20260101T000000Z-abc123")


def test_full_group_runs_pytest_for_device_and_unit_phases(
    run, mac_host, tmp_path, repo_dir
):
    state = make_state(run, mac_host, tmp_path, repo_dir, hardware="mps")

    device = run.phase_commands(state, "device_smoke")[0]
    unit = run.phase_commands(state, "unit_tests")[0]

    assert device.env == {"OPENADMET_EXPECT_DEVICE": "mps"}
    assert device.argv[:1] == ("env",) and run.DEVICE_TEST_PATH in device.argv
    assert unit.argv[-5:] == ("-n", "auto", "-q", "-p", "no:cacheprovider")
    assert unit.cwd == state.source_dir


def test_infer_group_uses_inline_device_check_and_predict_smoke(
    run, mac_host, tmp_path, repo_dir
):
    state = make_state(run, mac_host, tmp_path, repo_dir, group="infer")

    device = run.phase_commands(state, "device_smoke")[0]
    predict = run.phase_commands(state, "predict_smoke")

    assert "-c" in device.argv and "pytest" not in device.argv
    assert predict[0].argv[-2] == "build" and predict[2].argv[-2] == "check"
    assert "predict" in predict[1].argv and "--accelerator" in predict[1].argv
    assert predict[1].argv[-1] == "cpu"
    assert "unit_tests" not in run.resolve_phases("default", "infer")


def test_import_and_cli_run_outside_the_source_tree(run, mac_host, tmp_path, repo_dir):
    state = make_state(run, mac_host, tmp_path, repo_dir)

    for phase in ("import", "cli"):
        assert run.phase_commands(state, phase)[0].cwd == state.neutral_cwd
        assert state.neutral_cwd != state.source_dir


# Parsing ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "..\n12 passed, 2 skipped in 4.50s\n",
            {"passed": 12, "skipped": 2, "failed": 0},
        ),
        (
            "1 failed, 11 passed, 3 warnings in 3.10s (0:00:03)",
            {"failed": 1, "passed": 11, "warnings": 3},
        ),
        ("=== 2 errors in 0.5s ===", {"errors": 2, "passed": 0}),
        ("no tests ran in 0.01s", {"passed": 0, "failed": 0}),
    ],
)
def test_parse_pytest_summary_reads_final_line(run, text, expected):
    parsed = run.parse_pytest_summary(text)

    assert {k: parsed[k] for k in expected} == expected


def test_parse_pytest_summary_returns_none_without_summary(run):
    assert run.parse_pytest_summary("Traceback (most recent call last)") is None


def test_parse_package_list_formats(run):
    freeze = run.parse_package_list(
        "Foo_Bar==1.0\n# c\n-e /x\nbaz @ git+https://h/r@abc\n", "freeze"
    )
    export = run.parse_package_list(
        "@EXPLICIT\nnumpy=1.26.4=py312h1_0\n", "conda-export"
    )
    conda_json = run.parse_package_list(
        '[{"name": "Torch", "version": "2.5"}]', "conda-json"
    )

    assert freeze == ["baz @ git+https://h/r@abc", "foo-bar==1.0"]
    assert export == ["numpy==1.26.4"]
    assert conda_json == ["torch==2.5"]


def test_package_set_hash_ignores_order_and_detects_changes(run):
    assert run.package_set_hash(["a==1", "b==2"]) == run.package_set_hash(
        ["b==2", "a==1"]
    )
    assert run.package_set_hash(["a==1", "b==2"]) != run.package_set_hash(
        ["a==1", "b==3"]
    )


# End-to-end runs with a fake adapter ------------------------------------------------------------


def run_fake(run, tmp_path, repo_dir, *extra, body=FAKE_ADAPTER):
    args = cli_args(
        tmp_path, write_adapter(tmp_path, body), repo_dir, "--reps", "2", *extra
    )
    return run.main(args)


def read_results(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_fake_run_writes_one_complete_record(
    run, mac_host, tmp_path, repo_dir, python_on_path, monkeypatch
):
    monkeypatch.setattr(run, "IMPORT_SNIPPET", "import json")
    monkeypatch.setattr(run, "CLI_ARGS", ("true",))

    code = run_fake(
        run,
        tmp_path,
        repo_dir,
        "--group",
        "infer",
        "--phases",
        "install_cold,install_warm,import,cli",
    )
    [record] = read_results(tmp_path)

    assert code == 0
    assert (
        record["status"] == "ok"
        and record["failed_phase"] is None
        and record["failure_tail"] == []
    )
    assert record["schema_version"] == 1
    assert record["cell"] == {
        "id": "mac-cpu",
        "os": "mac",
        "hardware": "cpu",
        "python": "3.12",
        "group": "infer",
    }
    assert record["candidate"] == "fake" and record["adapter_status"] == "ready"
    assert record["tool_versions"] == {"echo fake-tool 1.0": "fake-tool 1.0"}
    assert {
        "platform",
        "machine",
        "cpu_count",
        "ram_bytes",
        "os_version",
        "gpus",
        "harness_python",
    } <= set(record["host"])
    assert set(record["git"]) == {"candidate", "harness"}
    assert record["timestamp_utc"].endswith("+00:00")
    assert record["options"]["cache_mode"] == "isolated"
    assert record["installer"]["selected"] is None

    phases = record["phases"]
    assert [
        len(phases[p]["seconds"])
        for p in ("install_cold", "install_warm", "import", "cli")
    ] == [1, 1, 2, 2]
    for phase in phases.values():
        assert phase["status"] == "ok"
        assert phase["median_s"] >= 0 and phase["min_s"] <= phase["max_s"]
        assert phase["exit_codes"] and set(phase["exit_codes"]) == {0}
    assert phases["install_cold"]["note"] == "fake install"
    assert "cache_size_bytes" in phases["install_cold"]

    env = record["env"]
    assert env["size_bytes"] > 0
    assert env["n_packages"] == 1 and env["package_list_source"] == "adapter"
    assert env["package_set_hash"] == run.package_set_hash(["foo-bar==1.0"])
    assert env["python_dist_hash"] and env["python_dist_count"] > 0
    out = tmp_path / "out"
    assert (out / env["package_list_file"]).read_text() == "foo-bar==1.0\n"
    assert (out / env["python_dist_file"]).exists()
    assert record["env_facts"]["python_version"].count(".") == 2
    assert (out / "logs" / "install_cold-r0-s0.log").exists()
    assert not Path(record["paths"]["env_dir"]).exists()
    assert (tmp_path / "work" / run.WORKDIR_MARKER).exists()


def test_failed_install_records_phase_and_log_tail_and_skips_dependents(
    run, mac_host, tmp_path, repo_dir
):
    body = FAKE_ADAPTER.replace(
        'steps = ["mkdir -p {env_dir}/bin", "touch {env_dir}/bin/marker"]',
        "steps = [\"sh -c 'echo boom-message; exit 3'\"]",
    )

    code = run_fake(
        run,
        tmp_path,
        repo_dir,
        "--phases",
        "install_cold,install_warm,import",
        body=body,
    )
    [record] = read_results(tmp_path)

    assert code == 1
    assert record["status"] == "failed" and record["failed_phase"] == "install_cold"
    assert any("boom-message" in line for line in record["failure_tail"])
    assert len(record["failure_tail"]) <= 40
    assert record["phases"]["install_cold"]["exit_codes"] == [3]
    assert record["phases"]["import"] == {
        "status": "skipped",
        "reason": "environment is not installed",
    }


def test_cell_without_install_entry_is_recorded_as_skipped(
    run, mac_host, tmp_path, repo_dir
):
    body = FAKE_ADAPTER.replace("[install.any.any]", "[install.cuda.any]")

    code = run_fake(run, tmp_path, repo_dir, "--phases", "install_cold", body=body)
    [record] = read_results(tmp_path)

    assert code == 0
    assert record["status"] == "skipped" and "no install entry" in record["reason"]


def test_global_cache_mode_runs_clear_cache_only_with_flag(
    run, mac_host, tmp_path, repo_dir
):
    marker = tmp_path / "work" / "cleared-marker"

    code = run_fake(
        run,
        tmp_path,
        repo_dir,
        "--cache-mode",
        "global",
        "--yes-clear-cache",
        "--phases",
        "install_cold",
    )
    [record] = read_results(tmp_path)

    assert code == 0 and marker.exists()
    assert record["options"]["cache_mode"] == "global"


def test_isolated_cold_run_deletes_only_the_cache_dir_inside_work_dir(
    run, mac_host, tmp_path, repo_dir
):
    cache = tmp_path / "work" / "cache" / "fake"
    run.ensure_work_dir(tmp_path / "work")
    cache.mkdir(parents=True)
    (cache / "stale").write_text("x")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")

    run_fake(run, tmp_path, repo_dir, "--phases", "install_cold")

    assert not (cache / "stale").exists()
    assert outside.read_text() == "keep"


def test_resolve_asof_records_success_and_failure(run, mac_host, tmp_path, repo_dir):
    run_fake(
        run, tmp_path, repo_dir, "--phases", "resolve_asof", "--asof", "2026-04-27"
    )
    failing = FAKE_ADAPTER.replace(
        'lock_asof = "echo asof {date}"',
        "lock_asof = \"sh -c 'echo no solution found; exit 1'\"",
    )
    run_fake(
        run,
        tmp_path,
        repo_dir,
        "--phases",
        "resolve_asof",
        "--asof",
        "2026-04-27",
        body=failing,
    )
    missing = FAKE_ADAPTER.replace('lock_asof = "echo asof {date}"\n', "")
    run_fake(
        run,
        tmp_path,
        repo_dir,
        "--phases",
        "resolve_asof",
        "--asof",
        "2026-04-27",
        body=missing,
    )
    ok, failed, skipped = read_results(tmp_path)

    assert (
        ok["phases"]["resolve_asof"]["status"] == "ok"
        and ok["phases"]["resolve_asof"]["asof"] == "2026-04-27"
    )
    assert failed["status"] == "failed" and failed["failed_phase"] == "resolve_asof"
    assert any("no solution found" in line for line in failed["failure_tail"])
    assert skipped["phases"]["resolve_asof"]["status"] == "skipped"


def test_resolve_asof_requires_a_valid_date(run, mac_host, tmp_path, repo_dir, capsys):
    assert run_fake(run, tmp_path, repo_dir, "--phases", "resolve_asof") == 2
    assert (
        run_fake(
            run, tmp_path, repo_dir, "--phases", "resolve_asof", "--asof", "27/04/2026"
        )
        == 2
    )
    assert "YYYY-MM-DD" in capsys.readouterr().err


def test_environment_phases_require_an_install_phase(
    run, mac_host, tmp_path, repo_dir, capsys
):
    assert run_fake(run, tmp_path, repo_dir, "--phases", "import") == 2
    assert "need install_cold or install_warm" in capsys.readouterr().err


def test_timeout_kills_command_and_reports_exit_124(run, tmp_path):
    log = tmp_path / "log.txt"

    code, seconds = run.run_logged(
        ["sleep", "30"], cwd=None, env=os.environ, log_path=log, timeout=0.3
    )

    assert code == 124 and seconds < 10
    assert "timeout" in log.read_text()


def test_missing_executable_reports_exit_127(run, tmp_path):
    code, _ = run.run_logged(
        ["definitely-not-a-command-xyz"],
        cwd=None,
        env=os.environ,
        log_path=tmp_path / "l",
        timeout=5,
    )

    assert code == 127


def test_expand_argv_supports_python_without_dot_and_tool_group(run):
    context = {"python_nodot": "313", "tool_group": "dev", "hardware": "cpu"}

    argv = run.expand_argv("pixi run -e {tool_group}-{hardware}-py{python_nodot}", context)

    assert argv == ["pixi", "run", "-e", "dev-cpu-py313"]
