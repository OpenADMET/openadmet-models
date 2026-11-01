# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests,
conda installation, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

The workflows in `.github/workflows` test each supported install route in its own job: [pixi](https://pixi.sh) from `pixi.lock`
(`pixi run --locked`) and [uv](https://docs.astral.sh/uv/) from `uv.lock` (`uv sync --locked`, tests with `uv run --no-sync`) on
Python 3.12 and 3.13, pip (`pip install -e ".[test]"`, unlocked) on Python 3.12 and 3.13, and the conda file
`conda-envs/openadmet-models.yaml` on Python 3.12. The pip and conda jobs are best effort, so a failure is reported without blocking
a merge. The scheduled GPU workflow tests the pixi and uv CUDA environments on one instance.

### Conda Environment:

This directory contains the files to setup the Conda environment for testing purposes

* `conda-envs`: directory containing the YAML file(s) which fully describe Conda Environments, their dependencies, and those dependency provenance's
  * `openadmet-models.yaml`: unlocked environment file for Python 3.12 (best effort)
  * `openadmet-models-gpu.yaml`: the same environment with the GPU build of PyTorch

### Lock files

Dependencies are declared once in `pyproject.toml` at the repository root, and the pixi and uv routes each have a lock file:

* `pyproject.toml`: dependencies, the `test` extra, the `cpu` and `cuda` torch selectors, the `[tool.uv]` settings (locked platforms, PyTorch indexes), and the `[tool.pixi]` tables (conda-forge dependencies and environments)
* `pixi.lock`: the resolved pixi environments for osx-arm64 and linux-64 on Python 3.12 and 3.13; after changing dependencies, run `pixi lock`
* `uv.lock`: the resolved uv environment for macOS arm64 and Linux x86_64 on Python 3.12 and 3.13; after changing dependencies, run `uv lock`

See `docs/installation.rst` for the commands.

### Additional Scripts:

This directory contains helper scripts which don't fall in any of the previous categories
* `scripts`
  * `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options
  * `share_libomp_macos.sh`: macOS only; points scikit-learn, LightGBM, and XGBoost at the OpenMP runtime bundled in the torch wheel, so one runtime is loaded per process and Homebrew libomp is not needed. Run it in uv and pip environments, not in pixi or conda

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
