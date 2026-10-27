# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests,
environment management, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

### Environments

Python environments are defined in the `[tool.pixi]` tables of `pyproject.toml` and locked in `pixi.lock` at the repository root. See `docs/installation.rst` for the environment names and commands. After changing dependencies, run `pixi lock` and commit `pixi.lock`.

### Containers

* `containers/Dockerfile`: image that installs the `full-cuda-py312` environment from `pixi.lock`; choose another with `--build-arg PIXI_ENV=<name>`
