Getting Started
===============

Installation
------------

You can install **openadmet-models** via our GitHub package. If you want the latest development version, clone the repository and install in editable mode:

.. code-block:: bash

    git clone git@github.com:OpenADMET/openadmet-models.git

Environment setup
-----------------

The repository contains conda-lock files in ``devtools/locks``. A lock file lists every package of an environment with its version, build, and checksum, so each install produces the same environment. Pick the lock file by Python version and hardware flavor:

.. list-table::
   :header-rows: 1

   * - Hardware flavor
     - Platforms
     - Lock files
     - Notes
   * - ``cpu``
     - ``osx-arm64``, ``linux-64``
     - ``py312-cpu-lock.yml``, ``py313-cpu-lock.yml``
     - CPU-only PyTorch
   * - ``mps``
     - ``osx-arm64``
     - ``py312-mps-lock.yml``, ``py313-mps-lock.yml``
     - Apple silicon GPU through the Metal Performance Shaders (MPS) backend; requires macOS 14.5 or newer
   * - ``cuda``
     - ``linux-64``
     - ``py312-cuda-lock.yml``, ``py313-cuda-lock.yml``
     - CUDA 12 build of PyTorch for NVIDIA GPUs such as the T4; the lock installs CUDA 12.9 libraries, which run on a driver that supports CUDA 12.4 or newer

The macOS requirement comes from ``mlx``, which TabPFN requires on Apple silicon and which conda-forge builds only for macOS 14.5 and newer.

Each lock file serves three dependency groups, which are lock categories:

.. list-table::
   :header-rows: 1

   * - Group
     - Lock categories
     - Contents
   * - ``infer``
     - ``main``
     - Load a trained model directory and run ``openadmet predict``
   * - ``full``
     - ``main``, ``full``
     - ``infer`` plus the ``models``, ``train``, and ``report`` extras
   * - ``dev``
     - ``main``, ``full``, ``dev``
     - ``full`` plus the test tools

The ``infer`` group is the core dependencies plus the ``infer`` extra of ``pyproject.toml`` (boto3, matplotlib, reportlab, seaborn, statsmodels, tabulate, uncertainty-toolbox, wandb), and the requirements of ChemProp and ``molfeat``. The ``openadmet`` command imports these packages when it starts. Model directories that use CatBoost, LightGBM, XGBoost, TabPFN, TabICL, NEPARE, or the Monroe featurizer need the ``full`` group because those libraries are not in ``infer``. The ``infer`` group covers ChemProp, random forest, and support vector models, based on the imports of ``openadmet/models/architecture``.

Install with micromamba
~~~~~~~~~~~~~~~~~~~~~~~

Install `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_ (or mamba 2.0 or newer), then create the environment and install the pip layer. The pip layer contains the git dependencies and the packages that conda-forge does not provide for every supported Python, such as ChemProp:

.. code-block:: bash

    micromamba create --name openadmet-models --file devtools/locks/py312-cpu-lock.yml \
        --category main --category full
    micromamba run --name openadmet-models python -m pip install --no-deps \
        -r devtools/conda-envs/pip-infer.txt -r devtools/conda-envs/pip-full.txt
    micromamba run --name openadmet-models python -m pip install --no-deps -e .

Without ``--category`` flags micromamba installs only the ``main`` category, which is the ``infer`` group. For the ``dev`` group add ``--category dev``. For the ``infer`` group install only ``pip-infer.txt``.

The helper script runs the same steps and checks the result with ``pip check``:

.. code-block:: bash

    python devtools/scripts/install_env.py --python 3.12 --hardware cpu --group full \
        --name openadmet-models --package editable

Install with conda
~~~~~~~~~~~~~~~~~~

Conda cannot read lock files directly. Use `conda-lock <https://conda.github.io/conda-lock/>`_ and then install the pip layer as above:

.. code-block:: bash

    conda-lock install --name openadmet-models -e full --no-dev devtools/locks/py312-cpu-lock.yml
    conda activate openadmet-models
    python -m pip install --no-deps -r devtools/conda-envs/pip-infer.txt -r devtools/conda-envs/pip-full.txt
    python -m pip install --no-deps -e .

Check the accelerator
~~~~~~~~~~~~~~~~~~~~~

The device smoke test skips an accelerator that is unavailable. Define ``OPENADMET_EXPECT_DEVICE`` as ``cpu``, ``mps``, or ``cuda`` to make the test fail when torch cannot use that device:

.. code-block:: bash

    OPENADMET_EXPECT_DEVICE=cuda pytest openadmet/models/tests/unit/test_devices.py

Update the lock files
~~~~~~~~~~~~~~~~~~~~~

Developers regenerate the lock files after changing dependencies, see the :doc:`developer_guide`.

For more details, see the full documentation or the README.
