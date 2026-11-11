Feature Transforms
==================

Transforms applied to featurizer output after featurization and before
model training. Transforms are fitted on the train partition only and applied to
val, test, and inference features, so learned statistics never see held-out data.
The fitted transforms are saved next to the model and re-applied at inference time.

.. toctree::
   :maxdepth: 1

   transform_base
   impute
   pca
