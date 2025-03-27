# IMPORTANT: order matters here, make sure to import all the registered classes first
# before importing the registry classes

from openadmet.models.architecture.chemprop import *  # noqa: F401 F403

# models
from openadmet.models.architecture.lgbm import *  # noqa: F401 F403
from openadmet.models.architecture.model_base import models  # noqa: F401  F403
from openadmet.models.eval.classification import *  # noqa: F401 F403
from openadmet.models.eval.cross_validation import *  # noqa: F401 F403
from openadmet.models.eval.eval_base import evaluators  # noqa: F401 F403

# evaluators
from openadmet.models.eval.regression import *  # noqa: F401 F403
from openadmet.models.features.chemprop import *  # noqa: F401 F403
from openadmet.models.features.combine import *  # noqa: F401 F403
from openadmet.models.features.feature_base import featurizers  # noqa: F401 F403

# featurizers
from openadmet.models.features.molfeat_fingerprint import *  # noqa: F401 F403
from openadmet.models.features.molfeat_properties import *  # noqa: F401 F403

# util
from openadmet.models.log import logger  # noqa: F401 F403
from openadmet.models.split.scaffold import *  # noqa: F401 F403

# splitters
from openadmet.models.split.sklearn import *  # noqa: F401 F403
from openadmet.models.split.split_base import splitters  # noqa: F401 F403
from openadmet.models.trainer.lightning import *  # noqa: F401 F403

# trainers
from openadmet.models.trainer.sklearn import *  # noqa: F401 F403
from openadmet.models.trainer.trainer_base import trainers  # noqa: F401 F403
