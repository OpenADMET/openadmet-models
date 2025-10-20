`OpenADMET Models`
==================
[//]: # (Badges)
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)
[![GitHub Actions Build Status](https://github.com/OpenADMET/openadmet_models/workflows/CI/badge.svg)](https://github.com/OpenADMET/openadmet_models/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main)
[![Documentation Status](https://readthedocs.org/projects/openadmet-models/badge/?version=latest)](https://openadmet-models.readthedocs.io/en/latest/?badge=latest)


`openadmet-models` contains implementations of machine learning architectures and training routines for use in the [OpenADMET project](https://openadmet.org). Our goal is to provide a consistent framework for rapid development, experimentation, and prototyping of ML models for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction tasks.

The library includes traditional machine learning methods, deep learning models, and active learning workflows. It is designed for general-purpose use and is not intended to implement every state-of-the-art architectures, but rather to provide a practical, flexible foundation for ADMET modeling.


>[!NOTE]
> This repo is under very active development, we make no guarantees about stability or correctness. Check out the documentation (WIP) here: https://openadmet-models.readthedocs.io/en/latest/

## License

This library is made available under the [MIT](https://opensource.org/licenses/MIT) open source license.


## Install

### Development version

The development version of `openadmet-models` can be installed directly from the `main` branch of this repository.

First install the package dependencies using `mamba`:

```bash
mamba env create -f devtools/conda-envs/openadmet-models.yaml
# or if you want a GPU compatible version devtools/conda-envs/openadmet-models-gpu.yaml
```

The `openadmet-models` library can then be installed via:

```
python -m pip install -e --no-deps .
```

## Examples

This repo contains a set of workflows called `Anvil` (inspired by conda forge) used to produce models from human readable model recipe specifications.

This allows you to train a new model with in a simple command line call:

```bash
openadmet anvil --recipe-path my_model_recipe.yaml --output-dir my_model
```

You can also do inference easily with a model trained with `anvil`

```
openadmet predict --input-path query-mols.sdf --model-dir my_model --output-path predictions.csv
```

Additionally, we ship the ability to robustly compare multiple models trained with `anvil`. We take a lot of inspiration from [this preprint](https://chemrxiv.org/engage/chemrxiv/article-details/672a91bd7be152b1d01a926b) about how to compare models

```bash
openadmet compare --model-stats my_model-1/cross_validation_metrics.json --taskname cyp3a4_pchembl --model-stats my_model-2/cross_validation_metrics.json --taskname cyp3a4_pchembl --output-dir compare_plots --report report.pdf
```

Some examples of `Anvil` specifications are provided in the Tutorials and Documentation.


## Getting started

>[!NOTE]
> We are currently in development mode and are working on what our first release and documentation will look like, we hope to have something stable for people to try out in Q4 2025!



## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
