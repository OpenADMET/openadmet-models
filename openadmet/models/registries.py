# IMPORTANT: order matters here, make sure to import all the registered classes first
# before importing the registry classes
# there must be a better way to do this, but for now this works

# active learning
from openadmet.models.active_learning.committee import *  # noqa: F401 F403
from openadmet.models.active_learning.ensemble_base import ensemblers  # noqa: F401 F403

# models
from openadmet.models.architecture.catboost import *  # noqa: F401 F403
from openadmet.models.architecture.chemprop import *  # noqa: F401 F403  # noqa: F401 F403
from openadmet.models.architecture.gat import *  # noqa: F401 F403
from openadmet.models.architecture.dummy import *  # noqa: F401 F403
from openadmet.models.architecture.lgbm import *  # noqa: F401 F403  # noqa: F401 F403
from openadmet.models.architecture.mtenn import *  # noqa: F401 F403
from openadmet.models.architecture.rf import *  # noqa: F401 F403
from openadmet.models.architecture.svm import *  # noqa: F401 F403
from openadmet.models.architecture.tabpfn import *  # noqa: F401 F403
from openadmet.models.architecture.xgboost import *  # noqa: F401 F403
from openadmet.models.architecture.model_base import models  # noqa: F401  F403

# evaluators
from openadmet.models.eval.classification import *  # noqa: F401 F403
from openadmet.models.eval.cross_validation import *  # noqa: F401 F403
from openadmet.models.eval.regression import *  # noqa: F401 F403
from openadmet.models.eval.uncertainty import *  # noqa: F401 F403
from openadmet.models.eval.eval_base import evaluators  # noqa: F401 F403

# featurizers
from openadmet.models.features.chemprop import *  # noqa: F401 F403
from openadmet.models.features.combine import *  # noqa: F401 F403
from openadmet.models.features.gat_featurizer import *  # noqa: F401 F403
from openadmet.models.features.molfeat_fingerprint import *  # noqa: F401 F403
from openadmet.models.features.molfeat_properties import *  # noqa: F401 F403
from openadmet.models.features.mtenn import *  # noqa: F401 F403
from openadmet.models.features.feature_base import featurizers  # noqa: F401 F403

# util
from openadmet.models.log import logger  # noqa: F401 F403

# splitters
from openadmet.models.split.scaffold import *  # noqa: F401 F403
from openadmet.models.split.sklearn import *  # noqa: F401 F403
from openadmet.models.split.split_base import splitters  # noqa: F401 F403

# trainers
from openadmet.models.trainer.lightning import *  # noqa: F401 F403
from openadmet.models.trainer.sklearn import *  # noqa: F401 F403
from openadmet.models.trainer.trainer_base import trainers  # noqa: F401 F403

# transforms
from openadmet.models.transforms.impute import *  # noqa: F401 F403
from openadmet.models.transforms.transform_base import *  # noqa: F401 F403
