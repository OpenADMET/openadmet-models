# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests,
conda installation, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

### Conda environments

The conda environments are locked with [conda-lock](https://conda.github.io/conda-lock/) so every install resolves to the same packages. See `docs/installation.rst` for how to install them.

* `conda-envs`: the specs that the lock files are solved from
  * `infer.yaml`, `full.yaml`, `dev.yaml`: conda-forge packages of the dependency groups, one lock category each (`main`, `full`, `dev`)
  * `pip-infer.txt`, `pip-full.txt`: pinned git dependencies and packages that conda-forge does not provide for every supported Python, installed with `pip install --no-deps` after the lock file
  * `python-3.12.yaml`, `python-3.13.yaml`: Python version fragments
  * `hardware-cpu.yaml`, `hardware-mps.yaml`, `hardware-cuda.yaml`: PyTorch build fragments
  * `virtual-packages-cpu.yaml`, `virtual-packages-cuda.yaml`: the driver and OS versions the lock files assume
* `locks`: generated lock files named `py<version>-<hardware>-lock.yml`; do not edit them by hand

### Containers

* `containers/Dockerfile`: image built by `.github/workflows/docker_deploy.yaml` from a CUDA lock file

### Additional Scripts:

This directory contains OS agnostic helper scripts which don't fall in any of the previous categories
* `scripts`
  * `lock_envs.py`: regenerates the lock files and checks that the specs, `pyproject.toml`, and the lock files agree
  * `install_env.py`: creates an environment from a lock file for a Python version, hardware flavor, and dependency group


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Make `pyproject.toml`, the specs in `conda-envs`, and the lock files in `locks` agree: edit the specs, run `python devtools/scripts/lock_envs.py lock`, and commit the lock files
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
