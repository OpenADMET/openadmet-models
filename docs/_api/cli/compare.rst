Compare CLI Guide
=================

The ``compare`` command-line interface (CLI) is used to compare the performance
of two or more trained models based on their summary statistics based on sampling from the performance distribution with cross-validation.
It supports multiple tasks, optional tagging of models, and report generation.
It is based on [this paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01609) by Ash *et al* which details a systematic workflow for model comparison based on cross validation statistics.


Usage
-----

Comparsion can be performed in two ways, either by specifying the path to the directory containing the model training outputs from which the statistics files will be located automatically
or by providing the model statistics files and their corresponding tags directly via the CLI. Option 1 is generally recommended for simplicity and to avoid errors.

Option 1: Specify the model directory

.. code-block:: bash

   compare --model-dirs DIR [--model-dirs DIR ...] \
           --label-types TAG [--label-types TAG ...] \
           [--mt-id TAG] [--output-dir DIR] [--report BOOL]

Option 2 : Specify cross validation statistics files and tags directly

.. code-block:: bash

   compare --model-stats-fns FILE [--model-stats-fns FILE ...] \
           --labels TAG [--labels TAG ...] \
           --task-names TASK [--task-names TASK ...] \
           [--mt-id TAG] [--output-dir DIR] [--report BOOL]

Options
-------

.. option:: --model-dirs DIR

   Path(s) to directories containing model training outputs (most likely produced by the ``openadmet anvil`` command).
   The command will automatically locate the relevant JSON files containing model statistics within these directories.
   Must be specified along with ``--label-types``.
   Can be specified multiple times to compare models in different directories. All models within parent directory will be used.
   Directories will be identified by any folder containing an ``anvil_recipe.yaml`` file and a ``cross_validation_metrics.json`` file.

   Example:

   .. code-block:: bash

        openadmet compare --model-dirs ./cyp3a4_chembl_lgbm/anvil_training \
              --model-dirs ./cyp3a4_chembl_chemprop/anvil_training \
              --label-types biotarget \
              --label-types model \
              --report True \
              --output-dir ./comparison_results \

.. option:: --label-types TAG

   Types of labels to use for the generated comparison plots.
   These labels will be automatically generated from the input anvil yaml file.
   Must be specified if ``--model-dirs`` is used.
   Must be one of:
   - biotarget : which will label by the specified biotarget from ``anvil["metadata"]["biotargets"]``
   - model : which will label from the model type from ``anvil["procedure"]["model"]["type"]``
   - feat : which will label from the featurizer type from ``anvil["procedure"]["feat"]["type"]``
   - tasks : which will label whether the model was trained as multi or single task from the LENGTH of ``anvil["data"]["target_cols"]``

   Example:

   .. code-block:: bash

        openadmet compare --model-dirs ./cyp3a4_chembl_lgbm/anvil_training \
                --label-types biotarget \
                --label-types model \
                --report True \
                --output-dir ./comparison_results \

.. option:: --model-stats-fns FILE

   Path(s) to JSON files containing model statistics (most likely produced by the ``openadmet anvil`` command with cross-validation).
   Must be specified along with ``--labels`` and ``--task-names``.
   Can be specified multiple times to compare multiple models.
   Must be specified **once per model**.
   The order of ``--model-stats-fns`` arguments must match the order of ``--labels`` arguments and ``--task-names`` arguments.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json \
              --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json \
              --labels lgbm_model \
              --labels chemprop_model \
              --task-names cyp3a4_ic50 \
              --task-names cyp3a4_ic50


.. option:: --labels TAG

   User-defined names to label and identify different models in the comparison.
   Should be specified in the same order as ``--model-stats-fns``.
   Optional but highly recommended for clarity.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --labels lgbm_model \
                --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --labels chemprop_model \
                --task-names cyp3a4_ic50 \
                --task-names cyp3a4_ic50

.. option:: --task-names TASK

   One or more task names to compare across models, these must exactly match the task names as they appear in the model statistics JSON files, and must be specified **once per model**.
   An example is shown below, where the task names differ by model.

   Example:

   .. code-block:: bash

      openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --labels lgbm_model \
              --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --labels chemprop_model \
              --task-names cyp3a4_ic50_v0 --task-names cyp3a4_ic50_chembl_v1


.. option:: --mt-id TAG

   Optional tag to identify multi-task models in the comparison.
   Must match some part of the string associated with the multi-task model in the statistics JSON file.
   If provided, this will trigger special handling of multi-task models in the comparison.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats-fns ./chembl_chemprop/anvil_training/cross_validation.json \
                --model-stats-fns ./chembl_chemprop/anvil_training/cross_validation.json \
                --labels chemprop_model \
                --labels chemprop_model \
                --task-names cyp3a4_ic50 \
                --task-names cyp2c9_ic50 \
                --mt-id 3a4 \
                --mt-id 2c9 \
                --output-dir ./comparison_results

.. option:: --output-dir DIR

   Path to a directory where comparison results (tables, plots, or reports) will be saved.
   If not provided, results will be shown in the console only.
   The directory must already exist.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --labels lgbm_model \
                --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --labels chemprop_model \
                --task-names cyp3a4_ic50 \
                --task-names cyp3a4_ic50 \
                --output-dir ./comparison_results

.. option:: --report BOOL

   Whether to generate a summary PDF report in the ``--output-dir``.
   Defaults to ``False``.

   Example:

   .. code-block:: bash

        openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --labels lgbm_model \
                --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --labels chemprop_model \
                --task-names cyp3a4_ic50 \
                --task-names cyp3a4_ic50 \
                --output-dir ./comparison_results \
                --report True

Description
-----------

The ``compare`` CLI:

1. Loads one or more JSON files containing model summary statistics.
2. Compares model performance across a specified task or label-type.
3. Optionally writes results and a PDF report to the ``--output-dir``.

Example Workflow Run
--------------------

.. code-block:: bash

        openadmet compare --model-stats-fns ./cyp3a4_chembl_lgbm/anvil_training/cross_validation.json --labels lgbm_model \
                --model-stats-fns ./cyp3a4_chembl_chemprop/anvil_training/cross_validation.json --labels chemprop_model \
                --model-stats-fns ./cyp3a4_chembl_rf/anvil_training/cross_validation.json --labels rf_model \
                --output-dir ./comparison_results \
                --report True \
                --task-names cyp3a4_ic50 \
                --task-names cyp3a4_ic50 \
                --task-names cyp3a4_ic50

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

- Ensure that the directories passed via ``--model-dirs`` contain valid cross validation metrics and anvil YAML files for option 1.
- Ensure that the JSON files passed via ``--model-stats-fns`` contain valid summary statistics for option 2.
- The number of ``--labels`` values must match the number of ``--model-stats-fns`` files for option 2.
- The number of ``--task-names`` values must match the number of ``--model-stats-fns`` files for option 2.
- Task names must exactly match those found in the JSON files.
- Report generation requires ``--output-dir`` to be specified.
