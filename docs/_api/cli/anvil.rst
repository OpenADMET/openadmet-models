

Anvil CLI Guide
===============

The ``anvil`` command-line interface (CLI) runs Anvil workflows for model building
from a recipe file. It provides options for specifying input files, debug mode,
output directories, and optional tagging of models.

Usage
-----

.. code-block:: bash

   openadmetanvil --recipe-path PATH [--output-dir DIR] [--tag TAG] [--debug]

Options
-------

.. option:: --recipe-path PATH

   **Required.**
   Path to the recipe YAML file that defines the workflow.
   Must be an existing file.

   Example:

   .. code-block:: bash

      openadmet anvil --recipe-path ./recipes/my_model.yaml

.. option:: --output-dir DIR

   Directory where output artifacts (such as logs, trained models,
   or reports) will be saved.
   Defaults to ``anvil_training`` if not provided.
   The directory must be writable.
   If the directory already exists, a hash will be appended to avoid overwriting.

   Example:

   .. code-block:: bash

      openadmet anvil --recipe-path ./recipes/my_model.yaml --output-dir ./results

.. option:: --tag TAG

   An optional user-defined string to tag and identify the model.
   Useful for differentiating between multiple runs or experiments.

   Example:

   .. code-block:: bash

      openadmet anvil --recipe-path ./recipes/my_model.yaml --tag experiment_v1

.. option:: --debug

   Enable debug mode for more verbose logging.
   Useful for troubleshooting workflows.

   Example:

   .. code-block:: bash

      openadmet anvil --recipe-path ./recipes/my_model.yaml --debug

Description
-----------

The ``openadmet anvil`` CLI reads a recipe YAML file, initializes a workflow using
``AnvilSpecification``, and executes it. A workflow run typically includes
data preparation, model training, and report generation.

The command proceeds in the following steps:

1. Load the recipe specification from ``--recipe-path``.
2. Convert the recipe into a workflow instance.
3. Run the workflow with the specified ``--tag``, ``--debug``, and ``--output-dir`` options.
4. Print status messages before and after execution.

Example Workflow Run
--------------------

.. code-block:: bash

   openadmet anvil \
       --recipe-path ./recipes/lgbm.yaml \
       --output-dir ./output \
       --debug

Expected output:

.. code-block:: text

   Workflow initialized successfully with recipe: ./recipes/lgbm.yaml
   ... # lots of logging output
   Workflow completed successfully

Exit Codes
----------

- ``0``: Workflow completed successfully.
- Non-zero: Workflow encountered an error (details will be shown in debug mode or logs).

Notes
-----

- Ensure that the recipe YAML file is valid and contains all required fields.
- The ``--output-dir`` will be created if it does not exist.
- If the output directory already exists, a hash will be appended to avoid overwriting.
- Debug mode can produce extensive logging; use it when diagnosing failures.
