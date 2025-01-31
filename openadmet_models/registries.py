# IMPORTANT: order matters here, make sure to import all the registered classes first
# before importing the registry classes

from openadmet_models.eval.eval_base import evaluators  # noqa: F401

# evaluators
from openadmet_models.eval.regression import *  # noqa: F401
from openadmet_models.features.feature_base import featurizers  # noqa: F401

# featurizers
from openadmet_models.features.molfeat_fingerprint import *  # noqa: F401
from openadmet_models.features.molfeat_properties import *  # noqa: F401

# models
from openadmet_models.models.gradient_boosting.lgbm import *  # noqa: F401
from openadmet_models.models.model_base import models  # noqa: F401
from openadmet_models.split.split_base import splitters  # noqa: F401

# splitters
from openadmet_models.split.vanilla import *

# trainers
from openadmet_models.trainer.sklearn import *
from openadmet_models.trainer.trainer_base import trainers  # noqa: F401
from openadmet_models.util.log import logger  # noqa: F401
