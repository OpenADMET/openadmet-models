Getting Started
===============

Installation
------------

**openadmet-models** is installed from a clone of the GitHub repository. Four routes are supported:

.. list-table::
   :header-rows: 1

   * - Route
     - Status
     - Platforms
     - Python
     - Lock file
   * - `pixi <https://pixi.sh>`_
     - Recommended; tested in CI
     - Linux x86_64, macOS Apple silicon
     - 3.12, 3.13
     - ``pixi.lock``
   * - `uv <https://docs.astral.sh/uv/>`_
     - Supported; tested in CI
     - Linux x86_64, macOS Apple silicon
     - 3.12, 3.13
     - ``uv.lock``
   * - pip
     - Best effort; not locked
     - Any platform with the dependency wheels
     - 3.12, 3.13
     - none
   * - conda
     - Best effort; Python 3.12 only
     - Linux x86_64, macOS Apple silicon
     - 3.12
     - none

Pixi installs everything from conda-forge (with PyPI for the few packages that conda-forge does not provide), so it needs no
extra steps on any platform. uv installs from PyPI, is the fastest route, and needs one extra step on macOS (see
`Known issues`_). The lock files make a clone of the repository install the same packages on every machine.

.. code-block:: bash

    git clone git@github.com:OpenADMET/openadmet-models.git
    cd openadmet-models

Hardware
--------

Every route offers two hardware targets, which differ in the torch build:

.. list-table::
   :header-rows: 1

   * - Hardware
     - Platform
     - Torch build
   * - CPU
     - Linux x86_64, macOS Apple silicon
     - CPU build on Linux; standard macOS build on Apple silicon, which includes the Metal Performance Shaders (MPS) backend and uses it automatically
   * - CUDA
     - Linux x86_64 with an NVIDIA driver that supports CUDA 12
     - CUDA 12 build (CUDA 12.6 wheels with uv)

ROCm (AMD) is not tested.

Installing
----------

Choose a route. The commands for each hardware target are in the tab.

.. tab-set::

   .. tab-item:: pixi (recommended)

      Install pixi first (see the pixi documentation), then install an environment:

      .. code-block:: bash

          pixi install
          pixi run openadmet --help

      ``pixi install`` creates the ``default`` environment (``cpu-py312``), which includes the test tools. The package is installed in
      editable mode, so source changes apply immediately. Other environments are identified as ``<hardware>-py<version>``, with hardware
      ``cpu`` or ``cuda`` and Python version ``py312`` or ``py313``. Pass the identifier with ``-e`` to any pixi command:

      .. code-block:: bash

          pixi install -e cpu-py313
          pixi run -e cpu-py313 test-unit
          pixi shell -e cuda-py312

      On Apple silicon, use the ``cpu`` environments: they install the macOS build of torch, which includes MPS. The ``cuda`` environments target the ``linux-64-cuda-12`` platform, which declares a CUDA 12 virtual
      package. Export ``CONDA_OVERRIDE_CUDA=12`` to install one on a machine without a GPU, as the Dockerfile does.

      Packages come from conda-forge unless it does not provide them. The pinned git dependencies (molfeat, neural-pairwise-regression,
      TabPFN, monroe, useful_rdkit_utils), tabpfn-extensions, and their dependencies come from PyPI.

   .. tab-item:: uv

      uv downloads the Python interpreter when none is installed. Pick the command for your hardware. Each one installs the package in
      editable mode with the test tools:

      .. code-block:: bash

          # macOS, Apple silicon (uses MPS)
          uv sync --extra test

          # Linux, CPU only
          uv sync --extra test --extra cpu

          # Linux, NVIDIA GPU (CUDA 12.6 wheels)
          uv sync --extra test --extra cuda

      On macOS, run ``sh devtools/scripts/share_libomp_macos.sh`` after every ``uv sync`` (see `Known issues`_). On Linux, always pass
      ``--extra cpu`` or ``--extra cuda``. Add ``--python 3.13`` to use Python 3.13.

      Run commands inside the environment with ``uv run --no-sync``, or activate it with ``source .venv/bin/activate``.
      A plain ``uv run`` syncs again using only the flags passed to it, and on Linux that replaces the torch build you selected.

      .. code-block:: bash

          uv run --no-sync openadmet --help

   .. tab-item:: pip

      Create a virtual environment and install from ``pyproject.toml``. This route does not use a lock file, so the
      resolved versions can differ from the tested ones (a fresh install resolved a newer torch than ``uv.lock`` pins):

      .. code-block:: bash

          python -m venv .venv
          source .venv/bin/activate
          pip install -e ".[test]"          # Python 3.13: add --ignore-requires-python (see Known issues)

      Torch comes from PyPI, so a Linux install downloads the CUDA build (see `Known issues`_). On macOS, run
      ``sh devtools/scripts/share_libomp_macos.sh .venv`` after the install (see `Known issues`_).

   .. tab-item:: conda

      The conda files are unlocked and are tested on Python 3.12 only (the GPU file has not been solved on macOS):

      .. code-block:: bash

          mamba env create -f devtools/conda-envs/openadmet-models.yaml
          mamba activate openadmet-models
          pip install -e . --no-deps

      ``devtools/conda-envs/openadmet-models-gpu.yaml`` is the variant for NVIDIA GPUs.

Known issues
------------

**uv on macOS: several OpenMP runtimes.** The torch and scikit-learn wheels bundle an OpenMP runtime (``libomp``), while the
LightGBM and XGBoost wheels search for a Homebrew one. A process that loads several runtimes crashes or stalls inside LightGBM and
XGBoost. After ``uv sync``, run ``sh devtools/scripts/share_libomp_macos.sh`` to point all four libraries at the runtime in the torch
wheel. No Homebrew ``libomp`` is needed; the script uses ``install_name_tool`` and ``codesign`` from the Xcode command line tools.
Repeat it whenever ``uv sync`` reinstalls torch, scikit-learn, LightGBM, or XGBoost, and do not use ``UV_LINK_MODE=symlink``, which
makes the script edit uv's shared cache.
Pixi and conda use one shared runtime and need no step. pip installs the same wheels as uv and shows the same problem, so run the
script there too, passing the path of the virtual environment as an argument.

**uv and pip on Linux: torch build.** Without ``--extra cpu`` or ``--extra cuda``, PyPI supplies the CUDA 13 build of torch,
which downloads about 3.4 GB of wheels including the NVIDIA libraries. The CUDA 12.6 wheels need an NVIDIA driver that supports
CUDA 12.x (525 or newer, following NVIDIA's minor-version compatibility rule). The LightGBM wheel needs the system ``libgomp``
(package ``libgomp1`` on Debian and Ubuntu).

**pip on Python 3.13: molfeat.** The molfeat fork declares ``requires-python "<=3.13"``, which pip rejects on every 3.13.x patch
release. Add ``--ignore-requires-python`` to the ``pip install -e ".[test]"`` command. uv does not enforce the bound, and pixi and conda
do not install the package this way.

**conda files: Python 3.12 only.** chemprop 2.1.1 or newer is not available on conda-forge for Python 3.13, so the conda files
cannot be solved there. Use pixi or uv for Python 3.13.

**pixi on Python 3.13.** chemprop comes from PyPI because conda-forge builds of chemprop support Python below 3.13 only. It is pinned
to 2.2.4: chemprop 2.3 imports the cuik_molmaker wheel, which needs the libraries bundled in the PyPI rdkit wheel and fails to load
next to conda-forge rdkit. conda-forge wandb below 0.26 requires protobuf below 7, so ``pyproject.toml`` bounds protobuf and the
pixi environments use torch 2.12.

Changing dependencies
---------------------

Add a dependency to ``[project.dependencies]`` in ``pyproject.toml``, then update both lock files and commit them with
``pyproject.toml``. CI installs with ``--locked``, so a stale lock file fails the build.

* uv: run ``uv lock`` (``uv lock --check`` verifies that the lock file agrees with ``pyproject.toml``).
* pixi: if conda-forge provides the package, add it with the same version bounds to the matching
  ``[tool.pixi.dependencies]`` table; otherwise pixi installs it from PyPI. Then run
  ``pixi lock``. For git dependencies, ``pixi lock`` installs the conda part of one environment under ``.pixi/`` to build package
  metadata.

Container image
---------------

``devtools/containers/Dockerfile`` installs ``cuda-py312`` from ``pixi.lock``. Choose another environment with
``--build-arg PIXI_ENV=<name>``.
