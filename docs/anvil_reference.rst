Anvil YAML Reference
====================

To initiate the ``anvil`` workflow, a recipe yaml file must be provided. 
There are many configuration options available..
Each workflow consists of four main sections: ``data``, ``metadata``,
``procedure``, and ``report``.

This guide should help you navigate the ``anvil`` workflow and understand the parameters 
you can set, their types, and how they interact across models and trainers.

.. contents::
   :local:
   :depth: 2

Metadata 
---------

Metadata specification available to ensure organized workflow.

.. code-block:: yaml

   metadata:
     authors: Author Name
     email: author@email.org
     biotargets: [CYP3A4, CYP2D6]
     build_number: 0
     description: description of run
     driver: driver_name
     name: workflow_name
     tag: chemprop
     tags: [openadmet, chemprop]
     version: v1

**Parameters:**

- ``authors`` (str | list[str]) — Author(s) of the workflow.
- ``email`` (str) — Contact email.
- ``biotargets`` (list[str]) — List of biotargets associated with the workflow.
- ``build_number`` (int) — Iteration number of the workflow.
- ``description`` (str) — Short description of the workflow.
- ``driver`` (str) — Backend framework for the workflow (ex. ``pytorch`` or ``sklearn``).
- ``name`` (str) — Workflow name.
- ``tag`` (str) — Main tag for the workflow.
- ``tags`` (list[str]) — Additional tags associated with the workflow description.
- ``version`` (str) — Version of the metadata schema. 

Data 
-----

Data specification for the workflow. 
###Need to add optional values for specs

.. code-block:: yaml

   data:
     type: intake
     resource: PATH_TO_DATASET.filetype
     input_col: COLUMN_NAME
     target_cols:
     - target_column_name1
     - target_column_name2
     dropna: false

**Parameters:**

- ``resource`` (str)  
  Path to dataset file. Allowed filetypes are ####. check with featurizer  

- ``type`` (str, default: ``intake``)  
  Loader type. Must be set to ``intake``. #Explain why 

- ``input_col`` (str)  
  Column containing model input. Allowed format is SMILES, ####  check with featurizer (maybe jsut has to be str)

- ``target_cols`` (list[str])  
  One or more target columns for prediction.  ####Modify explanation to be more robust

- ``dropna`` (bool, default: ``false``)  
  Whether to drop rows with NaN targets. #### Modify explanation to include situations where you might want Nan


Procedure 
----------

The ``procedure`` section defines featurization, model, splitting, and training.
# Split these sections into subsections? 

Featurization
~~~~~~~~~~~~~

- ``ChemPropFeaturizer`` — SMILES graph featurizer for ChemProp  
- ``GATGraphFeaturizer`` — Graph attention featurizer for GAT models
- ``MTENNFeaturizer`` - Masked PDB featurizer for MTENN models
- ``DescriptorFeaturizer`` — Uses RDKit descriptors (e.g. ``desc2d``).  
- ``FingerprintFeaturizer`` — Generates fingerprints (e.g. ``ecfp:4``).  
- ``FeatureConcatenator`` — Combines multiple featurizers 

.. code-block:: yaml

   feat:
     type: FeatureConcatenator
     params:
       featurizers:
         DescriptorFeaturizer:
           descr_type: desc2d
         FingerprintFeaturizer:
           fp_type: ecfp:4

Models
~~~~~~

#Go into more detail of each model? or provide a source for the models code or paper? 

# Got key params from canonical recipes. will visit code to ensure all options are listed

Supported model types:

- ``ChemPropModel`` — Description of chemprop  Message Passing NeuralNet
  - Key params: ``depth``, ``ffn_hidden_dim``, ``message_hidden_dim``, ``dropout``, ``batch_norm``, ``n_tasks``, ``from_chemeleon``.  
- ``GATv2Model`` — Graph Attention Network  
  - Key params: ``input_dim``, ``edge_dim``, ``hidden_dim``, ``num_layers``, ``num_heads``, ``gat_dropout``, ``pooling``, ``output_dim``.  
- ``CatBoostRegressorModel`` — Gradient boosting on decision trees .  
  - Key params: ``n_estimators``.  
- ``LGBMRegressorModel`` — LightGBM regressor 
  - Key params: ``n_estimators``, ``learning_rate``, ``alpha``.  
- ``XGBRegressorModel`` — XGBoost regressor  
  - Key params: ``n_estimators``, ``learning_rate``, ``alpha``.  
- ``RFRegressorModel`` — Random Forest regressor  
  - Key params: ``n_estimators``, ``max_depth``.  
- ``TabPFNRegressorModel`` — Transformer-based probabilistic forest  
  - Key params: ``ignore_pretraining_limits``, ``device``.  

Splits
~~~~~~

.. code-block:: yaml

   split:
     type: ShuffleSplitter
     params:
       train_size: 0.7
       val_size: 0.1
       test_size: 0.2
       random_state: 42

**Splitter options:**
- ``ShuffleSplitter`` — Random shuffling into train/val/test.  
- ``ScaffoldSplitter`` — Scaffold-based chemical splits.  

Training
~~~~~~~~

## Need to be more thorough; how far of explanations? 

Trainer types:

- ``LightningTrainer`` — For PyTorch models (Chemprop, Chemeleon, GAT, MTENN) 
- ``SKLearnBasicTrainer`` — For sklearn-compatible models (RF, LGBM, CatBoost, XGBoost, TabPFN)   
- ``SKLearnGridSearchTrainer`` — Hyperparameter tuning with grid search.  

Training parameters include:

- ``accelerator`` (``cpu`` | ``gpu``)  
- ``early_stopping`` (bool), ``early_stopping_patience`` (int), ``early_stopping_mode`` (``min``/``max``), ``early_stopping_min_delta`` (float)  
- ``max_epochs`` (int)  
- ``monitor_metric`` (str)  
- ``use_wandb`` (bool), ``wandb_project`` (str)  

Report Section
--------------

##This should be its own section. I need to comb through the code a bit more and be more thorough

Evaluations are defined as a list of tasks:

- ``RegressionMetrics`` — Computes regression statistics.  
- ``RegressionPlots`` — Generates plots of predicted vs true  
- ``SKLearnRepeatedKFoldCrossValidation`` — Cross-validation for sklearn models  
- ``PytorchLightningRepeatedKFoldCrossValidation`` — Cross-validation for Lightning models  

Each evaluation has a ``params`` dict. Common options:  

- ``axes_labels`` (list[str]) — Labels for plots.  
- ``n_splits`` (int) — Number of CV splits.  
- ``n_repeats`` (int) — Number of CV repeats.  
- ``random_state`` (int) — Seed for reproducibility.  
- ``title`` (str) — Title for plot or report.  
- ``max_val`` / ``min_val`` (float) — Plot value ranges.  
- ``pXC50`` (bool) — Whether to plot on pAC50 scale.  

----

This page should be updated as new models, featurizers, and trainers are added.

