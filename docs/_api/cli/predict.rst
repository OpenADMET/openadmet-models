Predict CLI Guide
=================

The ``predict`` command-line interface (CLI) generates predictions using
trained Anvil models. It supports inference from CSV or SDF input files,
hardware accelerator configuration, and optional active learning acquisition
functions.

Usage
-----

.. code-block:: bash

   predict --input-path PATH --model-dir MODEL_DIR [OPTIONS]

Options
-------

.. option:: --input-path PATH

   **Required.**
   Path to the input file containing molecular structures.
   Supported formats: CSV or SDF.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv --model-dir ./models/my_model

.. option:: --input-col NAME

   Column name in the CSV file that contains the molecular structures
   (SMILES strings).
   Defaults to ``OPENADMET_SMILES`` if not specified.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv --model-dir ./models/my_model --input-col smiles

.. option:: --model-dir PATH

   **Required.**
   Path to one or more trained model directories produced by ``openadmet anvil``.
   Can be specified multiple times to run predictions with multiple models.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/model_a \
              --model-dir ./models/model_b

.. option:: --output-csv FILE

   Path to the output CSV file where predictions will be written.
   Defaults to ``predictions.csv``.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --output-csv ./results/preds.csv

.. option:: --accelerator {cpu,gpu,tpu,ipu,mps,auto}

   Hardware accelerator to use for inference.
   Defaults to ``gpu`` if available.

   Choices:

   - ``cpu`` – Run inference on the CPU.
   - ``gpu`` – Run inference on the GPU (default).
   - ``tpu`` – Use TPU hardware.
   - ``ipu`` – Use IPU hardware.
   - ``mps`` – Use Apple MPS backend.
   - ``auto`` – Automatically select available hardware.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --accelerator cpu

.. option:: --aq-fxn {ucb,ei,pi}

   Acquisition function(s) for **active learning**.
   Can be specified multiple times to combine different functions.
   Supported values:

   - ``ucb`` – Upper Confidence Bound (requires ``--beta``).
   - ``ei`` – Expected Improvement (requires ``--best-y`` and ``--xi``).
   - ``pi`` – Probability of Improvement (requires ``--best-y`` and ``--xi``).

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --aq-fxn ucb --beta 0.5

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --aq-fxn ei --best-y 1.0 --xi 0.1

.. option:: --beta VALUE

   Parameter for the ``ucb`` acquisition function.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --aq-fxn ucb --beta 2.0

.. option:: --best-y VALUE

   Parameter for the ``ei`` and ``pi`` acquisition functions.
   Must be specified once per acquisition function.

.. option:: --xi VALUE

   Exploration parameter for ``ei`` and ``pi`` acquisition functions.
   Must be specified once per acquisition function.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv \
              --model-dir ./models/my_model \
              --aq-fxn ei --best-y 0.85 --xi 0.01

.. option:: --debug

   Enable verbose debug logging.
   Useful for diagnosing errors or inspecting execution details.

   Example:

   .. code-block:: bash

      predict --input-path ./data/molecules.csv --model-dir ./models/my_model --debug

Description
-----------

The ``predict`` CLI:

1. Reads molecular input data from CSV or SDF files.
2. Loads one or more trained Anvil models from ``--model-dir``.
3. Runs inference on the specified hardware accelerator.
4. Optionally applies active learning acquisition functions (UCB, EI, PI).
5. Writes predictions to the output CSV file.

Example Workflow Run
--------------------

.. code-block:: bash

   predict \
       --input-path ./data/test_set.csv \
       --input-col smiles \
       --model-dir ./models/final_model \
       --output-csv ./results/predictions.csv \
       --accelerator gpu \
       --aq-fxn ei --best-y 0.9 --xi 0.05 \
       --debug

Expected output:

.. code-block:: text

   Predictions written to ./results/predictions.csv


Example: Predict from an SDF File
---------------------------------

Suppose you have an input file ``molecules.sdf`` containing a set of molecular
structures. You can run inference with a trained model directory as follows:

.. code-block:: bash

   predict \
       --input-path ./data/molecules.sdf \
       --model-dir ./models/final_model \
       --output-csv ./results/predictions_from_sdf.csv \
       --accelerator gpu

Notes:

- The ``--input-col`` option is **not required** when using SDF input.
- Predictions will be saved in ``./results/predictions_from_sdf.csv``.
- If metadata fields (e.g., ``<ID>``) are present in the SDF, they will be
  included in the output CSV alongside predictions.
- Hardware can be selected with ``--accelerator`` (e.g., ``cpu``, ``gpu``).

Expected output:

.. code-block:: text

   Predictions written to ./results/predictions_from_sdf.csv


Exit Codes
----------

- ``0``: Prediction completed successfully.
- Non-zero: Prediction encountered an error (see logs or use ``--debug``).

Notes
-----

- Multiple models can be specified with ``--model-dir`` to perform ensemble predictions.
- Acquisition functions must be configured with their required parameters, otherwise execution will fail.
- Debug mode provides detailed logging for troubleshooting.
