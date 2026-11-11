Developer Guide
===============

Welcome to the developer documentation for OpenADMET Models!

Contributing
------------

OpenADMET Models is an open-source project, and we welcome contributions from the community. Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated!
We also welcome feedback and suggestions for improving the package. Please feel free to open issues on our GitHub repository or join our community discussions to share ideas and ask questions.

We require that all contributions adhere to our coding standards and pass our test suite.
Additionally, we ask that you follow our `Code of Conduct <https://omsf.io/resources/conduct/>`_ to ensure a welcoming and inclusive environment for all contributors.

We also require all contributors to agree to a Developer Certificate of Origin (DCO). This is a simple statement that you have the right to submit the code you are contributing and that you agree to have it included in the project under the project's license.
You can indicate your agreement by ticking the DCO box when submitting a pull request on GitHub.
Learn more here: `Developer Certificate of Origin <https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin>`_.

License
-------

OpenADMET Models is distributed under the **MIT License** — see the
`LICENSE <https://github.com/OpenADMET/openadmet_models/blob/main/LICENSE>`_ for full details.

Installation
------------

Follow the steps in the :doc:`installation` guide to set up your development environment, with pixi (for example
``pixi install -e cpu-py312``) or with uv (``uv sync --extra test --extra cpu`` on Linux, ``uv sync --extra test`` on macOS, where you
also run the OpenMP script described under Known issues). Both install the package in editable mode with the test tools.
This ensures that changes to the source code are reflected immediately.

Dependencies are declared in ``pyproject.toml`` (in ``[project.dependencies]`` for uv and pip, and in the ``[tool.pixi]`` tables
for pixi) and resolved in two lock files, ``uv.lock`` and ``pixi.lock``. Commit both lock files together with any change to the
dependencies; CI installs with ``uv sync --locked`` and ``pixi install --locked`` and fails when a lock file is out of date.

* uv: add a dependency with ``uv add <package>``, or ``uv add --optional test <package>`` for a test tool; both update ``uv.lock``
  (``uv lock --check`` detects drift).
* pixi: edit the ``[tool.pixi.dependencies]`` table (or ``[tool.pixi.feature.test.dependencies]`` for a test tool) with the same
  version bounds as ``[project.dependencies]``, then run ``pixi lock``.

The torch build varies by hardware. For uv, the ``cpu`` and ``cuda`` extras route it to the PyTorch package
indexes on Linux (see ``[tool.uv.sources]`` in ``pyproject.toml``), and ``[tool.uv] conflicts`` prevents installing the ``cpu`` and ``cuda`` extras together.
To move to another CUDA version, change the URL and index name of ``pytorch-cu126`` (in the index table and in ``[tool.uv.sources]``), then run ``uv lock``,
and update the CUDA versions in the pixi ``cuda`` feature and in :doc:`installation`.
For pixi, the ``cpu`` and ``cuda`` features select the torch package.

Testing
-------

We require that new features and bug fixes include appropriate tests. Tests are located in the ``openadmet/models/tests/`` directory.
We use `pytest` as our testing framework.

Tests are separated into **unit** and **integration** tests:
- Unit tests focus on individual components.
- Integration tests ensure that different parts of the package work together as expected.

In particular, the `Anvil` workflows are tested extensively in the integration test suite.

Tests are automatically run on each pull request using GitHub Actions.
You can also run the tests locally:

.. code-block:: bash

    # From the root of the repository
    # With uv, use "uv run --no-sync" so that uv does not change the torch build selected at install time

    # Run integration tests (pixi runs the cpu tests)
    pixi run -e cpu-py312 test-integration-cpu
    uv run --no-sync pytest openadmet/models/tests/integration

    # Run unit tests
    pixi run -e cpu-py312 test-unit
    uv run --no-sync pytest openadmet/models/tests/unit

To run integration tests using a GPU (if available and properly configured):

.. code-block:: bash

    pixi run -e cuda-py312 test-integration-gpu
    uv run --no-sync pytest -v -m gpu openadmet/models/tests/integration

Documentation
-------------

The documentation is built using Sphinx and is located in the ``docs/`` directory.

To build the documentation locally:

1. Create a new conda environment using the dependencies listed in ``docs/requirements.yaml``.
2. Activate the environment and build the HTML docs.

.. code-block:: bash

    mamba env create -f docs/requirements.yaml
    mamba activate docs_openadmet_models
    cd docs
    make html

You can then view the documentation by opening ``_build/html/index.html`` in your web browser.

Code Style
----------

We use `pre-commit` hooks to enforce code style and quality. These should run automatically when you submit a pull request to the repository.

Tips and Tricks
---------------

- Use descriptive commit messages to clarify the history of changes.
- Break large changes into smaller, manageable commits to simplify reviews.
- Keep your branches up to date with the main branch to avoid merge conflicts.
- Always write tests for new features and bug fixes to ensure code quality and prevent regressions.

Getting Help
------------

We’re a friendly bunch! If you have questions or need assistance, don’t hesitate to reach out.
You can open an issue on our GitHub repository or join the discussion on `GitHub Discussions <https://github.com/orgs/OpenADMET/discussions>`_.

We look forward to your contributions and hope you enjoy working with OpenADMET Models!
