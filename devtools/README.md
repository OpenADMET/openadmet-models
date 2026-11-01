# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests,
environment management, container builds, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

The workflows in `.github/workflows` test each supported install route in its own job: [pixi](https://pixi.sh) from `pixi.lock`
(`pixi run --locked`) and [uv](https://docs.astral.sh/uv/) from `uv.lock` (`uv sync --locked`, tests with `uv run --no-sync`) on
Python 3.12 and 3.13, pip (`pip install -e ".[test]"`, unlocked) on Python 3.12 and 3.13, and the conda file
`conda-envs/openadmet-models.yaml` on Python 3.12. The pip and conda jobs are best effort, so a failure is reported without blocking
a merge. The scheduled GPU workflow tests the pixi and uv CUDA environments on one instance.

### Environments

Dependencies are declared once in `pyproject.toml` at the repository root, and each supported install route has its own lock:

* `pyproject.toml`: dependencies, the `test` extra, the `cpu` and `cuda` torch selectors, the `[tool.uv]` settings (locked platforms, PyTorch indexes), and the `[tool.pixi]` tables (conda-forge dependencies and environments)
* `pixi.lock`: the resolved pixi environments for osx-arm64 and linux-64 on Python 3.12 and 3.13; after changing dependencies, run `pixi lock`
* `uv.lock`: the resolved uv environment for macOS arm64 and Linux x86_64 on Python 3.12 and 3.13; after changing dependencies, run `uv lock`
* `conda-envs/openadmet-models.yaml`: an unlocked conda file for Python 3.12 (best effort)

See `docs/installation.rst` for the commands.

### Scripts

* `scripts/share_libomp_macos.sh`: points scikit-learn, LightGBM, and XGBoost at the OpenMP runtime bundled in the torch wheel on macOS, so one runtime is loaded per process and Homebrew libomp is not needed; it applies to uv environments only
* `scripts/create_conda_env.py`: helper for creating a conda environment from a starter file with a Python version and environment name

### Containers

* `containers/Dockerfile`: image that installs the `cuda-py312` environment from `pixi.lock`; choose another with `--build-arg PIXI_ENV=<name>`


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Ensure that `pixi.lock` and `uv.lock` are up to date with `pyproject.toml` (`pixi lock` and `uv lock --check`)
- Push the branch to the repo (either the main or your fork) with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Versioneer Auto-version
[Versioneer](https://github.com/warner/python-versioneer) will automatically infer what version
is installed by looking at the `git` tags and how many commits ahead this version is. The format follows
[PEP 440](https://www.python.org/dev/peps/pep-0440/) and has the regular expression of:
```regexp
\d+.\d+.\d+(?\+\d+-[a-z0-9]+)
```
If the version of this commit is the same as a `git` tag, the installed version is the same as the tag,
e.g. `openadmet_models-0.1.2`, otherwise it will be appended with `+X` where `X` is the number of commits
ahead from the last tag, and then `-YYYYYY` where the `Y`'s are replaced with the `git` commit hash.
