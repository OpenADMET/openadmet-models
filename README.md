openadmet-models
==============================
[//]: # (Badges)
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)
[![GitHub Actions Build Status](https://github.com/OpenADMET/openadmet_models/workflows/CI/badge.svg)](https://github.com/OpenADMET/openadmet_models/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main)
[![Documentation Status](https://readthedocs.org/projects/openadmet-models/badge/?version=latest)](https://openadmet-models.readthedocs.io/en/latest/?badge=latest)


Implementations of machine learning architectures used in the OpenADMET project within a consistent framework for rapid development and prototyping of ML models for ADMET


## Examples
This repo contains a set of workflows called `anvil` (inspired by conda forge) used to produce models from human readable model recipe specifications.

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



>[!NOTE]
> This repo is under very active development, we make no guarantees about stability or correctness.

## Getting started

>[!NOTE]
> We are currently in development mode and are working on what our first release and documentation will look like, we hope to have something stable for people to try out in Q4 2025!

> When it is available, you will be able to read the documentation here: https://openadmet-models.readthedocs.io/en/latest/, watch this space!


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
