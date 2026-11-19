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

The development version of `openadmet-models` can be installed directly from the `main` branch of this repository.

First clone the repository:

```bash
git clone git@github.com:OpenADMET/openadmet-models.git
cd openadmet-models
```

Four install routes are supported. [pixi](https://pixi.sh) is recommended; [uv](https://docs.astral.sh/uv/) is the fastest on Linux. On macOS, use pixi or conda.

| Route | Status | Python | Lock file |
|---|---|---|---|
| pixi | Recommended; tested in CI | 3.12, 3.13 | `pixi.lock` |
| uv | Supported; tested in CI | 3.12, 3.13 | `uv.lock` |
| pip | Tested in CI | 3.12, 3.13 | none |
| conda | Tested in CI | 3.12, 3.13 | none |

To choose between the CPU and CUDA commands below, run `nvidia-smi`. If it prints a `CUDA Version` of 12.0 or newer, use CUDA. Otherwise, or if the command is not found, use CPU.

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
| Linux, CPU only | `uv sync --extra test --extra cpu` |
| Linux, NVIDIA GPU (CUDA 12.6 wheels) | `uv sync --extra test --extra cuda` |

Always pass `--extra cpu` or `--extra cuda`; without one, uv installs the CUDA 13 build of torch from PyPI (about 3.4 GB). Add `--python 3.13` to use Python 3.13. Run commands inside the environment with `uv run --no-sync`, for example `uv run --no-sync openadmet --help`, because a plain `uv run` syncs again and replaces the torch build you selected.

#### pip

Inside a virtual environment, install torch for your hardware first, then the package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu126    # NVIDIA GPU
pip install -e ".[test]"
```

Without the first step, pip installs the CUDA 13 build of torch from PyPI (about 3.4 GB).

#### conda

```bash
mamba env create -f devtools/conda-envs/openadmet-models.yaml
mamba activate openadmet-models
pip install -e . --no-deps
```

For NVIDIA GPUs, use `devtools/conda-envs/openadmet-models-gpu.yaml`.

#### Known issues

* **uv and pip on macOS:** not supported, because the OpenMP runtimes in the torch, scikit-learn, LightGBM, and XGBoost wheels conflict. Use pixi or conda. Tracked in [#625](https://github.com/OpenADMET/openadmet-models/issues/625).
* **uv and pip on Linux:** if importing LightGBM fails with an error about `libgomp.so.1`, run `sudo apt install libgomp1` (Debian and Ubuntu).
* **Python 3.13 with pip or conda:** the molfeat fork's `requires-python` bound rejects 3.13.x. Add `--ignore-requires-python` to the pip install, or set `PIP_IGNORE_REQUIRES_PYTHON=1` for `mamba env create`.

See the [installation guide](docs/installation.rst) for details.

## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
