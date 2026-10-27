# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests,
container builds, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

The workflows in `.github/workflows` install the package with [uv](https://docs.astral.sh/uv/) from the committed `uv.lock`
(`uv sync --locked`) and run the tests with `uv run --no-sync`. `CI.yaml` also checks that the inference-only install
(core dependencies plus the `infer` extra) can run `openadmet --help` and import `openadmet.models.inference.inference`.
Each workflow defines `OPENADMET_EXPECT_DEVICE` (`cpu` or `cuda`) so the device smoke test fails instead of skipping when the
runner cannot use the expected accelerator.

### Environment

The environment is defined in the repository root, not in this directory:

* `pyproject.toml`: dependencies, extras (`infer`, `models`, `train`, `report`, `all`, `cpu`, `cuda`), the `test` and `dev` groups,
  and the `[tool.uv]` settings (locked platforms, PyTorch indexes)
* `uv.lock`: the resolved environment for macOS arm64 and Linux x86_64 on Python 3.12 and 3.13

After editing dependencies, run `uv lock` and commit `uv.lock` with `pyproject.toml`. `uv lock --check` verifies that they agree.

### Containers

* `containers/Dockerfile`: builds an image from `uv.lock` (`all` and `cuda` extras by default; `--build-arg TORCH_EXTRA=cpu` for a CPU image)


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Ensure that `uv.lock` is up to date with `pyproject.toml` (`uv lock --check`)
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
