`OpenADMET Models`
==================
[//]: # (Badges)
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)
[![CI](https://github.com/OpenADMET/openadmet-models/actions/workflows/CI.yaml/badge.svg)](https://github.com/OpenADMET/openadmet-models/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenADMET/openadmet_models/branch/main)
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

First install the package dependencies using `mamba`:

```bash
mamba env create -f devtools/conda-envs/openadmet-models.yaml
# or if you want a GPU compatible version devtools/conda-envs/openadmet-models-gpu.yaml
```

The `openadmet-models` library can then be installed via:

```
python -m pip install -e --no-deps .
```


## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
