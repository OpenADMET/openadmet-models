Getting Started
===============

Installation
------------

You can install **openadmet-models** via our GitHub package. If you want the latest development version, clone the repository and install in editable mode:

.. code-block:: bash

    git clone git@github.com:OpenADMET/openadmet-models.git

Conda Environment Setup
----------------------

You can set up an environment using the provided files in `devtools/conda-envs`. For example:

.. code-block:: bash

    conda env create -f devtools/conda-envs/openadmet-models.yaml
    conda activate openadmet-models
    pip install -e .

If you want to use GPU acceleration, ensure you have the appropriate CUDA toolkit installed and use the `openadmet-models-cuda.yaml` file instead:

.. code-block:: bash

    conda env create -f devtools/conda-envs/openadmet-models-gpu.yaml
    conda activate openadmet-models
    pip install -e .

For more details, see the full documentation or the README.
