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

The development version of `openadmet-models` is installed from a clone of this repository with [uv](https://docs.astral.sh/uv/). The repository ships a `uv.lock` for macOS (Apple silicon) and Linux (x86_64) on Python 3.12 and 3.13, and uv downloads the Python interpreter when none is installed.

```bash
git clone https://github.com/OpenADMET/openadmet-models.git
cd openadmet-models
```

Pick the command for your hardware. It installs the package in editable mode with the training, model, and reporting dependencies plus the test tools.

| Hardware | Command |
|---|---|
| macOS, Apple silicon (uses MPS) | `uv sync --extra all` |
| Linux, CPU only | `uv sync --extra all --extra cpu` |
| Linux, NVIDIA GPU (CUDA 12.6 wheels) | `uv sync --extra all --extra cuda` |

On macOS, the LightGBM and XGBoost wheels from PyPI link a system OpenMP runtime, so install it with Homebrew before using the `models` extra: `brew install libomp`. On Linux, the LightGBM wheel needs the system `libgomp` (package `libgomp1` on Debian and Ubuntu).

On Linux, always pass `--extra cpu` or `--extra cuda`. Without one, PyPI supplies the CUDA 13 build of torch, which downloads about 3.4 GB of wheels including the NVIDIA libraries. Add `--python 3.13` to use Python 3.13 (the default is the first interpreter that satisfies `requires-python`).

Run commands inside the environment with `uv run --no-sync`, for example `uv run --no-sync openadmet --help`, or activate it with `source .venv/bin/activate`. A plain `uv run` syncs again using only the flags passed to it, and on Linux that replaces the torch build you selected. Check that torch can use your accelerator with:

```bash
OPENADMET_EXPECT_DEVICE=mps uv run --no-sync pytest openadmet/models/tests/unit/test_devices.py  # or cpu, cuda
```

### Inference only

To load a trained model directory and run `openadmet predict`, install the core dependencies plus the `infer` extra and skip the test tools:

```bash
uv sync --no-dev --extra infer              # macOS
uv sync --no-dev --extra infer --extra cpu  # Linux, CPU only
```

Models trained with the CatBoost, LightGBM, XGBoost, TabPFN, TabICL, or NePaRe architectures, the monroe featurizer, or the mordred descriptors import that backend when loaded. Add `--extra models` to install them.

| Extra | Contents |
|---|---|
| `infer` | Modules imported by the CLI and when a model directory is parsed (plotting, reporting, uncertainty, and tracking libraries) |
| `models` | Model backends and the monroe and mordred featurizers |
| `train` | Anvil training dependencies (splitting, zarr output); includes `infer` |
| `report` | S3 support for comparison reports; includes `infer` |
| `all` | `infer`, `models`, `train`, and `report` |
| `cpu`, `cuda` | Torch build selectors for Linux (see above) |

Without uv, `pip install -e ".[all]"` resolves the extras from `pyproject.toml` without the lockfile, and torch comes from PyPI.


## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
