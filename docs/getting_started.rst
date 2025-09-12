Getting Started
===============

Installation
------------

You can install **openadmet-models** using pip. We recommend using a virtual environment (such as `venv` or `conda`) for isolation.

.. code-block:: bash

    pip install openadmet-models

If you want the latest development version, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/your-org/openadmet-models.git
    cd openadmet-models
    pip install -e .

Conda Environment Setup
----------------------

If you prefer to use `conda`, you can set up an environment using the provided files in `devtools/conda-envs`. For example:

.. code-block:: bash

    # Create a new environment from the YAML file
    conda env create -f devtools/conda-envs/openadmet-models.yaml

    # Activate the environment
    conda activate openadmet-models

    # (Optional) Install the package in editable/development mode
    pip install -e .

You can find additional environment files for different configurations in the `devtools/conda-envs` directory.

For more details, see the full documentation or the README.
