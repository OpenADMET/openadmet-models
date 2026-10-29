Getting Started
===============

Installation
------------

**openadmet-models** is installed from a clone of the GitHub repository with `uv <https://docs.astral.sh/uv/>`_.
The repository ships a ``uv.lock`` for macOS (Apple silicon) and Linux (x86_64) on Python 3.12 and 3.13.
uv downloads the Python interpreter when none is installed.

.. code-block:: bash

    git clone git@github.com:OpenADMET/openadmet-models.git
    cd openadmet-models

Hardware selection
------------------

Pick the command for your hardware. Each one installs the package in editable mode with the model, training, and reporting
dependencies plus the test tools.

.. code-block:: bash

    # macOS, Apple silicon (uses MPS)
    uv sync --extra all

    # Linux, CPU only
    uv sync --extra all --extra cpu

    # Linux, NVIDIA GPU (CUDA 12.6 wheels)
    uv sync --extra all --extra cuda

On macOS, the LightGBM and XGBoost wheels from PyPI link the Homebrew OpenMP runtime, while the torch and scikit-learn
wheels bundle their own copy. A process that loads several runtimes crashes or stalls inside LightGBM and XGBoost.
After ``brew install libomp`` and ``uv sync``, run ``sh devtools/scripts/share_libomp_macos.sh`` to link the bundled copies
to the Homebrew one, and repeat it whenever ``uv sync`` reinstalls torch or scikit-learn. This applies to the ``models``
extra (and ``all``). On Linux, the LightGBM wheel needs the system ``libgomp`` (package ``libgomp1`` on Debian and Ubuntu).

On Linux, always pass ``--extra cpu`` or ``--extra cuda``. Without one, PyPI supplies the CUDA 13 build of torch,
which downloads about 3.4 GB of wheels including the NVIDIA libraries. The CUDA 12.6 wheels need an NVIDIA driver that
supports CUDA 12.x (525 or newer, following NVIDIA's minor-version compatibility rule). Add ``--python 3.13`` to use Python 3.13.

Run commands inside the environment with ``uv run --no-sync``, or activate it with ``source .venv/bin/activate``.
A plain ``uv run`` syncs again using only the flags passed to it, and on Linux that replaces the torch build you selected.

.. code-block:: bash

    uv run --no-sync openadmet --help

    # Check that torch can use the accelerator (use cpu, mps, or cuda)
    OPENADMET_EXPECT_DEVICE=mps uv run --no-sync pytest openadmet/models/tests/unit/test_devices.py

Inference only
--------------

To load a trained model directory and run ``openadmet predict``, install the core dependencies plus the ``infer`` extra and
skip the test tools:

.. code-block:: bash

    uv sync --no-dev --extra infer              # macOS
    uv sync --no-dev --extra infer --extra cpu  # Linux, CPU only

The ``infer`` extra adds the plotting, reporting, uncertainty, and tracking libraries that the CLI and the evaluator
registry import. Models trained with the CatBoost, LightGBM, XGBoost, TabPFN, TabICL, or NePaRe architectures, the monroe
featurizer, or the mordred descriptors import that backend when loaded. Add ``--extra models`` to install them.

Extras
------

.. list-table::
   :header-rows: 1

   * - Extra
     - Contents
   * - ``infer``
     - Modules imported by the CLI and when a model directory is parsed
   * - ``models``
     - Model backends and the monroe and mordred featurizers
   * - ``train``
     - Anvil training dependencies (splitting, zarr output); includes ``infer``
   * - ``report``
     - S3 support for comparison reports; includes ``infer``
   * - ``all``
     - ``infer``, ``models``, ``train``, and ``report``
   * - ``cpu``, ``cuda``
     - Torch build selectors for Linux

Without uv, ``pip install -e ".[all]"`` resolves the extras from ``pyproject.toml`` without the lockfile, and torch comes from PyPI.

Container image
---------------

``devtools/containers/Dockerfile`` builds an image from ``uv.lock`` with the ``all`` and ``cuda`` extras.
Build a CPU image with ``--build-arg TORCH_EXTRA=cpu``.

For more details, see the full documentation or the README.
