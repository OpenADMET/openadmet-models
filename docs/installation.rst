Getting Started
===============

Installation
------------

Environments are managed with `pixi <https://pixi.sh>`_. The dependencies are declared in ``pyproject.toml`` and every environment is locked in ``pixi.lock``, so a clone of the repository installs the same packages on every machine. Install pixi first (see the pixi documentation), then clone the repository and install an environment:

.. code-block:: bash

    git clone git@github.com:OpenADMET/openadmet-models.git
    cd openadmet-models
    pixi install
    pixi run openadmet --help

``pixi install`` creates the ``default`` environment (``full-cpu-py312``). The package is installed in editable mode, so source changes apply immediately.

Choosing an environment
-----------------------

Environment names have the form ``<group>-<hardware>-py<version>``. Pass the name with ``-e`` to any pixi command:

.. code-block:: bash

    pixi install -e dev-mps-py313
    pixi run -e dev-mps-py313 test-unit
    pixi shell -e full-cuda-py312

Groups select the dependencies:

.. list-table::
   :header-rows: 1

   * - Group
     - Contents
   * - ``infer``
     - Core dependencies from ``[project.dependencies]`` plus the ``infer`` extra. Enough to load a trained model directory and run ``openadmet predict``.
   * - ``full``
     - ``infer`` plus the ``models``, ``train``, and ``report`` extras (``report`` adds s3fs). Required to train models and to predict with the CatBoost, LightGBM, XGBoost, TabICL, TabPFN, and neural pairwise regression backends.
   * - ``dev``
     - ``full`` plus the ``test`` extra (pytest and plugins).

The ``infer`` extra covers the packages that the ``openadmet`` command imports at startup (the comparison module) and while parsing a trained model directory (the evaluator registry).

Hardware selects the torch build:

.. list-table::
   :header-rows: 1

   * - Hardware
     - Platform
     - Torch build
   * - ``cpu``
     - linux-64, osx-arm64
     - CPU build on Linux, standard macOS build on Apple silicon
   * - ``mps``
     - osx-arm64
     - Standard macOS build, which includes the Metal Performance Shaders (MPS) backend
   * - ``cuda``
     - linux-64
     - CUDA 12 build (``pytorch-gpu`` from conda-forge)

The ``cpu`` and ``mps`` environments contain the same packages on Apple silicon; the separate names let tests assert which device is expected. The ``cuda`` environments target the ``linux-64-cuda-12`` platform, which declares a CUDA 12 virtual package, so they are meant for machines with an NVIDIA driver that supports CUDA 12. Export ``CONDA_OVERRIDE_CUDA=12`` to install one on a machine without a GPU, as the Dockerfile does.

The Python version is ``py312`` or ``py313``. The three groups, three hardware targets, and two Python versions give 18 environments; only the combinations needed for a task have to be installed.

Packages come from conda-forge unless it does not provide them. The pinned git dependencies (molfeat, neural-pairwise-regression, TabPFN, monroe, useful_rdkit_utils), tabpfn-extensions, and their dependencies come from PyPI. On Python 3.13, chemprop comes from PyPI because conda-forge builds of chemprop support Python below 3.13 only. It is pinned to 2.2.4: chemprop 2.3 imports the cuik_molmaker wheel, which needs the libraries bundled in the PyPI rdkit wheel and fails to load next to conda-forge rdkit. conda-forge wandb below 0.26 requires protobuf below 7, so ``pyproject.toml`` bounds protobuf and the environments use torch 2.12.

Checking the device
-------------------

The device smoke tests run a tensor operation and a one-epoch Lightning fit on each accelerator. Export ``OPENADMET_EXPECT_DEVICE`` as ``cpu``, ``mps``, or ``cuda`` to make the test fail, instead of skip, when that device is unusable:

.. code-block:: bash

    OPENADMET_EXPECT_DEVICE=cuda pixi run -e dev-cuda-py312 pytest openadmet/models/tests/unit/test_devices.py

Changing dependencies
---------------------

Add a dependency to ``[project.dependencies]`` or an extra in ``pyproject.toml``. If conda-forge provides the package, add it with the same version bounds to the matching ``[tool.pixi.dependencies]`` or ``[tool.pixi.feature.<extra>.dependencies]`` table; otherwise pixi installs it from PyPI. Then run ``pixi lock`` and commit ``pixi.lock``. CI installs with ``--locked``, so a stale lock file fails the build. For git dependencies, ``pixi lock`` installs the conda part of one environment under ``.pixi/`` to build package metadata.

Container image
---------------

``devtools/containers/Dockerfile`` installs ``full-cuda-py312`` from ``pixi.lock``. Choose another environment with ``--build-arg PIXI_ENV=<name>``.
