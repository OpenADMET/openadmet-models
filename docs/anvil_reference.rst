Anvil Reference
================

Anvil is a workflow engine integrated into openadmet_models that allows users to define a human-readable model
specification to reproducibly train and evaluate machine learning models. This is to facilitate large-scale reproducible training and comparisons
across different datasets, models, and featurizations. In addition, anvil also allows for the training of model ensembles, that can be easily used in downstream
applications such as active learning.

Anvil is built around the concept of a "recipe" - a YAML file that specifies all the components of a machine learning workflow, including data loading,
featurization, model architecture, training parameters, and evaluation metrics. By defining a recipe, users can easily reproduce experiments, share workflows with others, and systematically explore different modeling approaches.
Anvil also makes our (OpenADMET Team) lives easier by handling the boilerplate code associated with setting up and running machine learning experiments, saving us from a twisted jungle of scripts and configuration files.


A full list of available models, featurizers, trainers, and evaluators can be found in the OpenADMET API documentation. Additionally, we maintain a list of canonical recipes we use in production at https://github.com/OpenADMET/optimus-prime


How to Use Anvil
=================

To initiate the ``anvil`` workflow, a recipe yaml file must be provided.
There are many configuration options available.
Each workflow consists of four main sections: ``metadata``, ``data``,
``procedure``, and ``report``.

This guide should help you navigate the ``anvil`` workflow and understand the parameters
you can set, their types, and how they interact across models and trainers.

.. contents::
   :local:
   :depth: 2

Metadata
---------

The ``metadata`` section provides essential information about the workflow, such as authorship,
versioning, and descriptive tags. This section ensures that workflows are well-documented
and easily identifiable. Many of these fields are purely descriptive and do not affect the workflow's execution.

Workflows are divided into "drivers" which specify the backend framework to be used.
Currently supported drivers are ``pytorch``, ``sklearn``, and ``pytorch_ensemble``.
The ``driver`` field must be set accordingly to ensure compatibility with the chosen featurizer, model and
trainer.  In short ``pytorch`` is used for deep learning models, ``sklearn`` for traditional machine learning models (and their ensembles),
and ``pytorch_ensemble`` for training ensembles of deep learning models.


.. code-block:: yaml

   metadata:
     authors: Author Name
     email: author@email.org
     biotargets: [target_1, target_2]
     build_number: 0
     description: description of run
     driver: driver_name
     name: workflow_name
     tag: main_tag
     tags: [sub_tag_1, sub_tag_2]
     version: v1

**Parameters**

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Name
     - Type
     - Description
   * - authors
     - str | list[str]
     - Author(s) of the workflow.
   * - email
     - str
     - Contact email.
   * - biotargets
     - list[str]
     - List of biological targets associated with the workflow.
   * - build_number
     - int
     - Iteration number of the workflow.
   * - description
     - str
     - Short description of the workflow.
   * - driver
     - str
     - Backend framework for the workflow (e.g., ``pytorch`` or ``sklearn`` or ``pytorch_ensemble``).
   * - name
     - str
     - Workflow name.
   * - tag
     - str
     - Main tag for the workflow.
   * - tags
     - list[str]
     - Additional tags associated with the workflow description.
   * - version
     - str
     - Version of the metadata schema. (Currently must be set to ``v1``).

Data
-----

The ``data`` section defines how input data is loaded and which columns are
used for modeling. You must specify the dataset location, input column, target columns,
and optional preprocessing steps. The data loader can read from remote locations as well as local files


Reading from a local file requires specifying the path to the dataset file in the ``resource`` field.
Supported file types include CSV, and Parquet.

.. code-block:: yaml

   data:
     type: intake
     resource: PATH_TO_DATASET.parquet
     input_col: COLUMN_NAME
     target_cols:
     - target_column_name1
     - target_column_name2
     dropna: false


A more advanced option is to use an Intake catalog to manage datasets. This is done by specifying a YAML file in the ``resource`` field
and the catalog entry in the ``cat_entry`` field. This allows for more flexible dataset management, especially when dealing with multiple datasets or complex data sources.

.. code-block:: yaml

   data:
     type: intake
     resource: PATH_TO_CATALOG.yaml
     cat_entry: catalog_entry_name
     input_col: COLUMN_NAME
     target_cols:
     - target_column_name1
     - target_column_name2
     dropna: false


Pulling data from a remote location is also possible by specifying a URL in the ``resource`` field.

.. code-block:: yaml

   data:
     type: intake
     resource: https://path_to_your_data/dataset.parquet
     input_col: COLUMN_NAME
     target_cols:
     - target_column_name1
     - target_column_name2
     dropna: false

**Parameters**

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Name
     - Type
     - Description
   * - resource
     - str
     - Path to dataset file. Allowed filetypes: YAML, CSV, parquet.
   * - type
     - str, default: ``intake``
     - Loader type. Must be ``intake``. Uses the `Intake`_ data catalog
       system to read datasets.
   * - input_col
     - str
     - Column name containing molecular input.
   * - target_cols
     - Union[str, list[str]]
     - Name(s) of the target column(s) for the model to predict.
   * - dropna
     - Optional[bool]
     - Whether to drop rows with missing values (``NaN``) in the input or
       target columns.
   * - cat_entry
     - Optional[str]
     - Used when ``resource`` is a YAML file, to specify which
       catalog entry to load.
   * - anvil_dir
     - Optional[str]
     - Allows for ``resource`` to point to a directory path.
       Useful for flexible dataset locations.

.. _Intake: https://intake.readthedocs.io/

Procedure
----------

The ``procedure`` section is the core of the workflow, where the data is transformed, models are defined,
data splits are configured, and training parameters are set. Each subsection provides
details on the available options and their configurations:

- **Featurization**: Defines how molecular data is transformed into numerical representations
  using various available featurizers.
- **Models**: Specifies the model to be used.
- **Splits**: Configures how the dataset is divided into training, validation, and test sets
  using assigned splitter.
- **Training**: Sets up the training process, including the trainer type
  and training parameters.

Each subsection provides examples and parameter descriptions to help you configure the workflow
according to your requirements.

Featurization
~~~~~~~~~~~~~
The ``features`` module provides a variety of featurizers which map
molecular data into suitable input formats for the specified model.
Below are the available options. Each featurizer has its own set of parameters
which can be found in the linked OpenADMET API documentation.

In general we follow the design pattern that all deep learning featurizers return
a ``PyTorch DataLoader`` as input, while traditional machine learning models return a
a ``2D NumPy array`` or pandas ``DataFrame``.

.. list-table::
  :header-rows: 1
  :widths: 20 80

  * - Featurizer
    - Description
  * - :doc:`ChemPropFeaturizer </_api/api/featurization/chemprop>`
    - Converts SMILES strings into a ChemProp compatible PyTorch DataLoader.
  * - :doc:`GATGraphFeaturizer </_api/api/featurization/GAT>`
    - Converts SMILES strings into graph Data objects for GAT-like models, uses a PyTorch DataLoader.
  * - :doc:`MTENNFeaturizer </_api/api/featurization/mtenn>`
    - Creates masked PDB features suitable for downstream use in MTENN models, uses a PyTorch DataLoader.
  * - :doc:`DescriptorFeaturizer </_api/api/featurization/descriptors>`
    - Uses the  `molfeat <https://github.com/datamol-io/molfeat>`_ library to compute molecular descriptors.
  * - :doc:`FingerprintFeaturizer </_api/api/featurization/fingerprints>`
    - Uses the `molfeat <https://github.com/datamol-io/molfeat>`_ library to compute molecular fingerprints.
  * - :doc:`FeatureConcatenator </_api/api/featurization/feature_combiner>`
    - Combines multiple featurizers into a single feature array.



Featurization for a traditional machine learning model using fingerprints is easily done
by specifying the ``FingerprintFeaturizer``


.. code-block:: yaml

   feat:
     type: FingerprintFeaturizer
     params:
       fp_type: ecfp:4
       radius: 2


You can also combine multiple traditional ML featurizers using the ``FeatureConcatenator``. Here we combine  RDKit 2D descriptors and ECFP4 fingerprints.

.. code-block:: yaml

   feat:
     type: FeatureConcatenator
     params:
       featurizers:
         DescriptorFeaturizer:
           descr_type: desc2d
         FingerprintFeaturizer:
           fp_type: ecfp:4
           radius: 2
           n_bits: 2048


For deep learning models, the ``ChemPropFeaturizer`` is a common choice. Below is an example configuration.

.. code-block:: yaml

  feat:
    type: ChemPropFeaturizer
    params: {}


Models
~~~~~~

The ``models`` section specifies the model to be used in the workflow.
It allows you to define the type of model, its parameters, and any additional configurations
required for training and evaluation. Each model type has its own set of options, enabling
customization to suit specific tasks and datasets. Refer to the linked OpenADMET API documentation for detailed information
on each model's implementation and usage.

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Model Type
    - Description
  * - :doc:`ChemPropModel </_api/api/model_architectures/chemprop>`
    - `ChemProp <https://github.com/chemprop/chemprop>`_ Message Passing Neural Network. Also, used when implementing `Chemeleon <https://github.com/JacksonBurns/chemeleon>`_.
  * - :doc:`GATv2Model </_api/api/model_architectures/GAT>`
    - Graph Attention Network v2 (`GATv2 <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv>`_) model implementation.
  * - :doc:`CatBoostClassifierModel </_api/api/model_architectures/catboost>`
    - Gradient boosting on decision trees for classification using `CatBoost <https://catboost.ai/docs/en/>`_.
  * - :doc:`CatBoostRegressorModel </_api/api/model_architectures/catboost>`
    - Gradient boosting on decision trees for regression using `CatBoost <https://catboost.ai/docs/en/>`_.
  * - :doc:`LGBMClassifierModel </_api/api/model_architectures/lgbm>`
    - `LightGBM <https://lightgbm.readthedocs.io/en/stable/>`_ classifier.
  * - :doc:`LGBMRegressorModel </_api/api/model_architectures/lgbm>`
    - `LightGBM <https://lightgbm.readthedocs.io/en/stable/>`_ regressor.
  * - :doc:`XGBClassifierModel </_api/api/model_architectures/xgboost>`
    - `XGBoost <https://xgboost.readthedocs.io/en/latest/python/>`_ classifier model implementation
  * - :doc:`XGBRegressorModel </_api/api/model_architectures/xgboost>`
    - `XGBoost <https://xgboost.readthedocs.io/en/latest/python/>`_ regressor model implementation
  * - :doc:`RFClassifierModel </_api/api/model_architectures/random_forest>`
    - scikit-learn `Random Forest classifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ .
  * - :doc:`RFRegressorModel </_api/api/model_architectures/random_forest>`
    - scikit-learn `Random Forest regressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ .
  * - :doc:`TabPFNClassifierModel </_api/api/model_architectures/tabpfn>`
    - TabPFN classification model using the basic `tabpfn <https://github.com/PriorLabs/TabPFN>`_ implementation.
  * - :doc:`TabPFNRegressorModel </_api/api/model_architectures/tabpfn>`
    - TabPFN regression model using the basic `tabpfn <https://github.com/PriorLabs/TabPFN>`_ implementation.
  * - :doc:`TabPFNPostHocClassifierModel </_api/api/model_architectures/tabpfn>`
    - TabPFN classification model using `tabpfn-extensions <https://github.com/priorlabs/tabpfn-extensions>`_ with posthoc ensembling.
  * - :doc:`TabPFNPostHocRegressorModel </_api/api/model_architectures/tabpfn>`
    - TabPFN regression model using `tabpfn-extensions <https://github.com/priorlabs/tabpfn-extensions>`_ with posthoc ensembling.
  * - :doc:`MTENNSchNetModel </_api/api/model_architectures/mtenn>`
    - Modular Training and Evaluation of Neural Networks (`MTENN <https://github.com/choderalab/mtenn>`_) `SchNet <https://github.com/atomistic-machine-learning/SchNet>`_ implementation.
  * - :doc:`DummyClassifierModel </_api/api/model_architectures/dummy>`
    - scikit-learn `Dummy Classifier <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier>`_ for baseline comparisons.
  * - :doc:`DummyRegressorModel </_api/api/model_architectures/dummy>`
    - scikit-learn  `Dummy Regressor <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor>`_  for baseline comparisons.
  * - :doc:`SVMClassifierModel </_api/api/model_architectures/svm>`
    - scikit-learn Support Vector Machine classifier `(SVC) <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_ .
  * - :doc:`SVMRegressorModel </_api/api/model_architectures/svm>`
    - scikit-learn Support Vector Machine regressor `(SVR) <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_ .

Example
^^^^^^^
.. code-block:: yaml

  model:
    type: ChemPropModel
    params:
      depth: 4
      ffn_hidden_dim: 1024
      ffn_hidden_num_layers: 4
      message_hidden_dim: 2048
      dropout: 0.2
      batch_norm: True
      messages: bond
      n_tasks: 1
      from_chemeleon: False


Split
~~~~~~

The ``split`` section defines how the dataset is divided into training, validation, and test sets.
You can choose from different splitter types, each with its own parameters to control the splitting behavior.

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Splitter
    - Description
  * - :doc:`ShuffleSplitter </_api/api/splitting/sklearn>`
    - Randomly shuffles and splits the dataset into training, validation, and test sets based on specified proportions.
  * - :doc:`ScaffoldSplitter </_api/api/splitting/data_driven>`
    - Splits the dataset based on molecular scaffolds to ensure that similar compounds are grouped together in the same set.
  * - :doc:`MaxDissimilaritySplitter </_api/api/splitting/data_driven>`
    - Splits the dataset based on maximum dissimilarity between training, validation, and test sets, promoting diversity in each set.
  * - :doc:`PerimeterSplitter </_api/api/splitting/data_driven>`
    - Splits the dataset by selecting compounds at the periphery of the chemical space, ensuring that edge cases are included in the training set.

Example
^^^^^^^

.. code-block:: yaml

   split:
     type: ShuffleSplitter
     params:
       train_size: 0.7
       val_size: 0.1
       test_size: 0.2
       random_state: 42


Training
~~~~~~~~

The ``training`` section configures the training process for the selected model.
It allows you to specify the trainer type and various training parameters to control the training workflow.

.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Trainer
    - Description
  * - :doc:`LightningTrainer </_api/api/training/lightning>`
    - Trainer for deep learning models using `PyTorch Lightning <https://www.pytorchlightning.ai/>`_.
  * - :doc:`SKLearnBasicTrainer </_api/api/training/sklearn>`
    - Basic trainer for sklearn models.
  * - :doc:`SKLearnGridSearchTrainer </_api/api/training/sklearn>`
    - Trainer that performs hyperparameter tuning using specifically grid search for sklearn models (`GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_).
  * - :doc:`SKLearnSearchTrainer </_api/api/training/sklearn>`
    - Trainer that performs hyperparameter tuning using specified search object for sklearn models.


Example
^^^^^^^
.. code-block:: yaml

  train:
    type: LightningTrainer
    params:
      accelerator: gpu
      early_stopping: true
      early_stopping_patience: 10
      early_stopping_mode: min
      early_stopping_min_delta: 0.001
      max_epochs: 50
      monitor_metric: val_loss
      use_wandb: false
      wandb_project: demos

Ensemble
~~~~~~~~
There is also an optional ``ensemble`` section that allows you to specify if you want to train an ensemble of models.
You can define the number of models in the ensemble and the calibration method to be used.
Currently we only offer a :doc:`CommitteeRegressor </_api/api/active_learning/committee>` to measure disagreement among the models in the ensemble.

Models can also be calibrated after training using a scaling factor method to improve uncertainty estimates. This functionality is provided by the `uncertainty_toolbox <https://github.com/uncertainty-toolbox/uncertainty-toolbox>`_ package.
See :doc:`UncertaintyMetrics </_api/api/model_evaluation/uncertainty>` for more details.

Example
^^^^^^^
.. code-block:: yaml

  ensemble:
    type: CommitteeRegressor
    n_models: 10
    calibration_method: scaling-factor

Report
--------------

The ``report`` section specifies the evaluations to be performed after training the model.
You can choose from various evaluation types, each with its own parameters to customize the output. Regression models are only compatible with RegressionMetrics and similarly classification models only with ClassificationMetrics.

Importantly, the ``report`` section also allows for cross-validation to be performed as part of the evaluation to evaluate the robustness of the model. Note that cross-validation can be computationally expensive, especially for deep learning models.


.. list-table::
  :header-rows: 1
  :widths: 30 70

  * - Evaluation
    - Description
  * - :doc:`RegressionMetrics </_api/api/model_evaluation/regression>`
    - Computes regression statistics.
  * - :doc:`RegressionPlots </_api/api/model_evaluation/regression>`
    - Generates plots of predicted vs true values for regression tasks.
  * - :doc:`ClassificationMetrics </_api/api/model_evaluation/classification>`
    - Computes classification statistics.
  * - :doc:`ClassificationPlots </_api/api/model_evaluation/classification>`
    - Generates plots such as ROC and precision-recall curves for classification tasks.
  * - :doc:`SKLearnRepeatedKFoldCrossValidation </_api/api/model_evaluation/cross_validation>`
    - Performs repeated K-Fold cross-validation for sklearn models. It should be noted that performing cross-validation can be computationally expensive.
  * - :doc:`PytorchLightningRepeatedKFoldCrossValidation </_api/api/model_evaluation/cross_validation>`
    - Performs repeated K-Fold cross-validation for PyTorch Lightning models. It should be noted that performing cross-validation can be computationally expensive.
  * - :doc:`PosthocBinaryMetrics </_api/api/model_evaluation/binary>`
    - Compute posthoc binary metrics. Intended to be used for regression-based models to calculate precision and recall metrics for user-input cutoffs. Not intended for binary models.
  * - :doc:`UncertaintyMetrics </_api/api/model_evaluation/uncertainty>`
    - Evaluate uncertainty metrics using `uncertainty_toolbox <https://github.com/uncertainty-toolbox/uncertainty-toolbox>`_.
  * - :doc:`UncertaintyPlots </_api/api/model_evaluation/uncertainty>`
    - Generates uncertainty plots.

Example
^^^^^^^
.. code-block:: yaml

  report:
    eval:
    - type: RegressionMetrics
      params: {}
    - type: PytorchLightningRepeatedKFoldCrossValidation
      params:
        axes_labels:
        - True pIC50
        - Predicted pIC50
        n_repeats: 5
        n_splits: 2
        random_state: 42
        pXC50: true
        title: True vs Predicted pIC50 on test set
