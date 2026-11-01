`OpenADMET Models`
==================
[//]: # (Badges)
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)
[![CI](https://github.com/OpenADMET/openadmet-models/actions/workflows/CI.yaml/badge.svg)](https://github.com/OpenADMET/openadmet-models/actions/workflows/CI.yaml)
[![Documentation Status](https://readthedocs.org/projects/openadmet-models/badge/?version=latest)](https://openadmet-models.readthedocs.io/en/latest/?badge=latest)
[![Demos](https://img.shields.io/badge/GitHub-Demos-blue?logo=github)](https://github.com/OpenADMET/openadmet-demos)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://try.openadmet.org)
<!--[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OpenADMET/openadmet-demos/HEAD?urlpath=%2Fdoc%2Ftree%2Fshowcase%2FOpenADMET_Models_Showcase.ipynb)-->


`openadmet-models` contains implementations of machine learning architectures and training routines for use in the [OpenADMET project](https://openadmet.org). Our goal is to provide a consistent framework for rapid development, experimentation, and prototyping of ML models for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction tasks.

The library includes traditional machine learning methods, deep learning models, and active learning workflows. It is designed for general-purpose use and is not intended to implement every state-of-the-art architectures, but rather to provide a practical, flexible foundation for ADMET modeling.

Read the documentation [here](https://docs.openadmet.org) to learn more. There is also a set of [demonstration tutorials](https://demos.openadmet.org) for a deeper dive into and a showcase example you can try live on [Google Colab](https://try.openadmet.org)!

## License

This library is made available under the [MIT](https://opensource.org/licenses/MIT) open source license.


## Install

### Development version

The development version of `openadmet-models` is installed from a clone of this repository:

```bash
git clone git@github.com:OpenADMET/openadmet-models.git
cd openadmet-models
```

Four install routes are supported. [pixi](https://pixi.sh) is recommended; [uv](https://docs.astral.sh/uv/) is the fastest.

| Route | Status | Python | Lock file |
|---|---|---|---|
| pixi | Recommended; tested in CI | 3.12, 3.13 | `pixi.lock` |
| uv | Supported; tested in CI | 3.12, 3.13 | `uv.lock` |
| pip | Best effort; not locked | 3.12, 3.13 | none |
| conda | Best effort | 3.12 only | none |

#### pixi

```bash
pixi install
pixi run openadmet --help
```

`pixi install` creates the `default` environment: CPU with Python 3.12 and the test tools, with the package installed in editable mode. Other environments are called `<hardware>-py<version>`:

| Hardware | Platform | Torch build |
|---|---|---|
| `cpu` | linux-64, osx-arm64 | CPU build on Linux; the macOS build on Apple silicon, which includes the Metal Performance Shaders (MPS) backend |
| `cuda` | linux-64 | CUDA 12 |

The version suffix is `py312` or `py313`. For example, install the environment for a CUDA machine with:

```bash
pixi install -e cuda-py312
pixi run -e cuda-py312 test-unit
```

#### uv

Pick the command for your hardware. It installs the package in editable mode with the test tools.

| Hardware | Command |
|---|---|
| macOS, Apple silicon (uses MPS) | `uv sync --extra test` |
| Linux, CPU only | `uv sync --extra test --extra cpu` |
| Linux, NVIDIA GPU (CUDA 12.6 wheels) | `uv sync --extra test --extra cuda` |

On macOS, run `sh devtools/scripts/share_libomp_macos.sh` after every `uv sync` (see Known issues below). On Linux, always pass `--extra cpu` or `--extra cuda`; without one, PyPI supplies the CUDA 13 build of torch, which downloads about 3.4 GB. Add `--python 3.13` to use Python 3.13. Run commands inside the environment with `uv run --no-sync`, for example `uv run --no-sync openadmet --help`, because a plain `uv run` syncs again and on Linux replaces the torch build you selected.

#### pip and conda

Inside a virtual environment, `pip install -e ".[test]"` resolves the dependencies from `pyproject.toml` without a lock file, and torch comes from PyPI. On Python 3.13, add `--ignore-requires-python`, because the molfeat fork's `requires-python` bound rejects 3.13.x releases. On macOS, run the OpenMP script described under Known issues with the path of the virtual environment, for example `sh devtools/scripts/share_libomp_macos.sh .venv`.

For conda, `mamba env create -f devtools/conda-envs/openadmet-models.yaml` followed by `pip install -e . --no-deps` works on Python 3.12 only, because chemprop 2.1.1 or newer is not on conda-forge for Python 3.13.

#### Known issues

* **uv on macOS:** the torch and scikit-learn wheels bundle an OpenMP runtime, while the LightGBM and XGBoost wheels search for a Homebrew one. A process that loads several runtimes crashes or stalls inside LightGBM and XGBoost. After `uv sync`, run `sh devtools/scripts/share_libomp_macos.sh` to point all four libraries at the runtime in the torch wheel (no Homebrew `libomp` is needed; the script uses `install_name_tool` and `codesign` from the Xcode command line tools), and repeat it whenever `uv sync` reinstalls torch, scikit-learn, LightGBM, or XGBoost. pip installs show the same problem; pixi and conda are not affected.
* **Linux:** the LightGBM wheel needs the system `libgomp` (package `libgomp1` on Debian and Ubuntu).
* **Python 3.13 with conda files:** not supported; use pixi or uv.

See the [installation guide](docs/installation.rst) for details.

## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
