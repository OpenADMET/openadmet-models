"""
Lazy registry loader — zero heavy imports at module level.

Call ``load_group(name)`` to load a specific registry group, or ``load_all()``
to populate every registry at once.  Both are idempotent.
"""

import importlib

_MODELS = [
    "openadmet.models.architecture.catboost",
    "openadmet.models.architecture.chemprop",
    "openadmet.models.architecture.dummy",
    "openadmet.models.architecture.lgbm",
    "openadmet.models.architecture.nepare",
    "openadmet.models.architecture.rf",
    "openadmet.models.architecture.svm",
    "openadmet.models.architecture.tabpfn",
    "openadmet.models.architecture.xgboost",
]

_EVALUATORS = [
    "openadmet.models.eval.classification",
    "openadmet.models.eval.cross_validation",
    "openadmet.models.eval.regression",
    "openadmet.models.eval.uncertainty",
]

_FEATURIZERS = [
    "openadmet.models.features.chemprop",
    "openadmet.models.features.combine",
    "openadmet.models.features.molfeat_fingerprint",
    "openadmet.models.features.molfeat_properties",
    "openadmet.models.features.null_featurizer",
]

_SPLITTERS = [
    "openadmet.models.split.scaffold",
    "openadmet.models.split.sklearn",
    "openadmet.models.split.cluster",
]

_TRAINERS = [
    "openadmet.models.trainer.lightning",
    "openadmet.models.trainer.sklearn",
]

_TRANSFORMS = [
    "openadmet.models.transforms.impute",
    "openadmet.models.transforms.pca",
    "openadmet.models.transforms.transform_base",
]

_ACTIVE_LEARNING = [
    "openadmet.models.active_learning.committee",
]

_GROUPS: dict[str, list[str]] = {
    "models": _MODELS,
    "evaluators": _EVALUATORS,
    "featurizers": _FEATURIZERS,
    "splitters": _SPLITTERS,
    "trainers": _TRAINERS,
    "transforms": _TRANSFORMS,
    "active_learning": _ACTIVE_LEARNING,
}

_loaded: set[str] = set()


def load_group(name: str) -> None:
    """
    Import all modules in the named registry group (idempotent).

    Parameters
    ----------
    name : str
        Registry group key.  Must be one of: ``"models"``, ``"evaluators"``,
        ``"featurizers"``, ``"splitters"``, ``"trainers"``, ``"transforms"``,
        ``"active_learning"``.

    """
    if name in _loaded:
        return
    for mod in _GROUPS[name]:
        importlib.import_module(mod)
    _loaded.add(name)


def load_all() -> None:
    """Import all registry groups, making every registered class available (idempotent)."""
    for name in _GROUPS:
        load_group(name)
