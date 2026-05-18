"""Registry objects and lazy loader for all model components.

Importing this module is intentionally cheap.  Concrete classes are registered
only when the relevant group is first accessed.  To eagerly load everything
(e.g. for CLI tools or the Anvil workflow), call ``load_all()``:

    from openadmet.models.registries import load_all
    load_all()
"""

from openadmet.models._registry_loader import load_all  # noqa: F401
from openadmet.models.active_learning.ensemble_base import ensemblers  # noqa: F401
from openadmet.models.architecture.model_base import models  # noqa: F401
from openadmet.models.eval.eval_base import evaluators  # noqa: F401
from openadmet.models.features.feature_base import featurizers  # noqa: F401
from openadmet.models.log import logger  # noqa: F401
from openadmet.models.split.split_base import splitters  # noqa: F401
from openadmet.models.trainer.trainer_base import trainers  # noqa: F401
