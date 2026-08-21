"""Base class for transforms, allows for arbitrary transformation of input data."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

transforms = ClassRegistry(unique=True)


def get_transform_class(trans_type):
    """
    Retrieve a transform class from the registry by type.

    Parameters
    ----------
    trans_type : str
        The type of transform to retrieve.

    Returns
    -------
    TransformBase
        The transform class corresponding to the given type.

    Raises
    ------
    ValueError
        If ``trans_type`` is not found in the transform registry.

    """
    from openadmet.models._registry_loader import load_group

    load_group("transforms")
    try:
        transf_class = transforms.get_class(trans_type)
    except RegistryKeyError:
        raise ValueError(
            f"Transform type {trans_type} not found in transform catalogue"
        )
    return transf_class


class TransformBase(BaseModel, ABC):
    """Base class for transforms, allows for arbitrary transformation of feature data."""

    # Whether transform accepts the feature_blocks kwarg at fit time; the workflow
    # forwards feature_blocks only to transforms that declare this as True
    accepts_feature_blocks: ClassVar[bool] = False

    def required_block_keys(self) -> set[str] | None:
        """
        Return the block keys this transform is configured against.

        Lets the workflow check a per-block configuration against the
        featurizer's block layout before any data is featurized. The default
        is None, meaning the transform is not configured per block and imposes
        no constraint on the layout.

        Returns
        -------
        set of str or None
            The required block keys, or None if the transform is layout-agnostic.

        """
        return None

    @abstractmethod
    def transform(self, X: np.ndarray, *args, **kwargs):
        """
        Transform the input data X, returns transformed data in an appropriate format.

        Parameters
        ----------
        X : np.ndarray
            Input data to be transformed.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Any
            Transformed data in an appropriate format for the model (e.g., numpy arrays, dataloaders, etc.)
            and optional processing info.

        """


def to_transform_list(
    transform: TransformBase | Sequence[TransformBase],
) -> list[TransformBase]:
    """
    Normalize a single transform or a sequence of transforms to a list.

    Parameters
    ----------
    transform : TransformBase or sequence of TransformBase
        A fitted or unfitted transform, or an ordered sequence of them.

    Returns
    -------
    list
        The transforms as a list, in application order.

    """
    # A bare transform is itself the whole sequence
    if isinstance(transform, TransformBase):
        return [transform]

    return list(transform)


def fit_transforms(
    transform: TransformBase | Sequence[TransformBase],
    X: np.ndarray,
    feature_blocks: list[tuple[str, int]] | None = None,
) -> np.ndarray:
    """
    Fit a transform sequence on X and return the transformed result.

    Each transform must implement fit; stateless transforms define a no-op.
    Elements are fit in order, each on the previous element's output, so
    statistics are computed on the train data only. ``feature_blocks`` is
    forwarded only to elements that declare ``accepts_feature_blocks``.

    Parameters
    ----------
    transform : TransformBase or sequence of TransformBase
        The transform or ordered transform sequence to fit.
    X : np.ndarray
        Train feature matrix.
    feature_blocks : list of tuple, optional
        Pairs of (block key, block width) in column order, forwarded to
        transforms that accept it.

    Returns
    -------
    np.ndarray
        The train features after the full sequence.

    """
    current = np.asarray(X)

    for step in to_transform_list(transform):
        # Only block-aware transforms take the layout; passing it to the rest
        # would break their fit signature
        if getattr(step, "accepts_feature_blocks", False):
            step.fit(current, feature_blocks=feature_blocks)
        else:
            step.fit(current)

        # Feed this element's output to the next one, so each fits on what it
        # will actually see rather than on the raw features
        current = step.transform(current)

    return current


def transform_features(
    transform: TransformBase | Sequence[TransformBase],
    X: np.ndarray,
) -> np.ndarray:
    """
    Apply a fitted transform sequence to X in order.

    Used on the inference path where no fitting happens; every transform must
    already be fitted or it raises.

    Parameters
    ----------
    transform : TransformBase or sequence of TransformBase
        The fitted transform or ordered transform sequence.
    X : np.ndarray
        Feature matrix to transform.

    Returns
    -------
    np.ndarray
        The features after the full sequence.

    """
    current = np.asarray(X)
    for step in to_transform_list(transform):
        current = step.transform(current)
    return current
