Compare CLI Guide
=================

The ``compare`` command-line interface (CLI) is used to compare the performance
of two or more trained models based on their summary statistics based on sampling from the performance distribution with cross-validation.
It supports multiple tasks, optional tagging of models, and report generation.
It is based on [this paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609) by Ash *et al* which details a systematic workflow for model comparison based on cross validation statistics.


Usage
-----

.. code-block:: bash

   compare --model-stats FILE [--model-stats FILE ...] \
           --model-tag TAG [--model-tag TAG ...] \
           --task-name TASK [--task-name TASK ...] \
           [--output-dir DIR] [--report BOOL]

Options
-------

.. option:: --model-stats FILE

   **Required.**
   Path(s) to JSON files containing model statistics (most likely produced by the ``openadmet anvil`` command with cross-validation).
   Can be specified multiple times to compare multiple models.
   Must be specified **once per model**.
   The order of ``--model-stats`` arguments must match the order of ``--model-tag`` arguments and ``--task-name`` arguments.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json \
              --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json \
              --task-name cyp3a4_ic50 \
              --task-name cyp3a4_ic50


.. option:: --model-tag TAG

   User-defined names to label and identify different models in the comparison.
   Should be specified in the same order as ``--model-stats``.
   Optional but highly recommended for clarity.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --model-tag lgbm_model \
                --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --model-tag chemprop_model \
                --task-name cyp3a4_ic50 \
                --task-name cyp3a4_ic50

.. option:: --task-name TASK

   **Required.**
   One or more task names to compare across models.
   These must exactly match the task names as they appear in the model statistics JSON files.
   and must be specified **once per model**. AN example is shown below, where the task names differ by model.

   Example:

   .. code-block:: bash

      openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --model-tag lgbm_model \
              --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --model-tag chemprop_model \
              --task-name cyp3a4_ic50_v0 --task-name cyp3a4_ic50_chembl_v1



.. option:: --output-dir DIR

   Path to a directory where comparison results (tables, plots, or reports) will be saved.
   If not provided, results will be shown in the console only.
   The directory must already exist.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --model-tag lgbm_model \
                --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --model-tag chemprop_model \
                --task-name cyp3a4_ic50 \
                --task-name cyp3a4_ic50 \
                --output-dir ./comparison_results

.. option:: --report BOOL

   Whether to generate a summary PDF report in the ``--output-dir``.
   Defaults to ``False``.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --model-tag lgbm_model \
                --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --model-tag chemprop_model \
                --task-name cyp3a4_ic50 \
                --task-name cyp3a4_ic50 \
                --output-dir ./comparison_results \
                --report True

Description
-----------

The ``compare`` CLI:

1. Loads one or more JSON files containing model summary statistics.
2. Matches each statistics file to its corresponding ``--model-tag``.
3. Compares model performance across a specified task.
4. Optionally writes results and a PDF report to the ``--output-dir``.

Example Workflow Run
--------------------

.. code-block:: bash

        openadmet compare --model-stats ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --model-tag lgbm_model \
                --model-stats ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --model-tag chemprop_model \
                --model-stats ./cyp3a4_chembl_rf/anvil_training/cross_validation.json --model-tag rf_model \
                --output-dir ./comparison_results \
                --report True \
                --task-name cyp3a4_ic50 \
                --task-name cyp3a4_ic50 \
                --task-name cyp3a4_ic50

Expected output:

.. code-block:: text

   Comparison complete. Results written to ./comparison_results
   PDF report generated.

Exit Codes
----------

- ``0``: Comparison completed successfully.
- Non-zero: Comparison encountered an error (see logs for details).

Notes
-----

- Ensure that the JSON files passed via ``--model-stats`` contain valid summary statistics.
- The number of ``--model-tag`` values must match the number of ``--model-stats`` files.
- The number of ``--task-name`` values must match the number of ``--model-stats`` files.
- Task names must exactly match those found in the JSON files.
- Report generation requires ``--output-dir`` to be specified.
