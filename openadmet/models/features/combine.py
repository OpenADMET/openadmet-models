"""Combine features from multiple featurizers into a single feature array."""

from functools import reduce
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, PrivateAttr, field_validator

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)


# See #595: this function only exists to normalize the single-key-dict shape
# to the {type, params} wrapper; if that shape is ever dropped, this collapses
# to `return item["type"], item.get("params") or {}` inline at the call site
def _parse_featurizer_entry(item: dict) -> tuple[str, dict]:
    """
    Normalize a list-form featurizer entry to (type, params).

    Accepts either the AnvilSection-style wrapper (``{"type": ..., "params":
    {...}}``) or a single entry mapping the registry type directly to its
    params (the shape used by each entry in the dict form).

    """
    if "type" in item:
        return item["type"], item.get("params") or {}
    if len(item) != 1:
        raise ValueError(
            "Featurizer list entries must be {type: ..., params: ...} "
            f"wrappers or single-type entries, got keys: {list(item.keys())}."
        )
    ((feat_type, params),) = item.items()
    return feat_type, params or {}


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    """
    Concatenate features from multiple featurizers into a single feature array.

    Attributes
    ----------
    featurizers : list of FeaturizerBase
        List of featurizer instances to concatenate. Same-class featurizers
        are rejected because per-key transforms key blocks by featurizer name
        and cannot disambiguate same-class blocks.

    """

    provides_feature_blocks: ClassVar[bool] = True

    featurizers: list[FeaturizerBase] = Field(
        ..., description="List of featurizers to concatenate"
    )
    _cached_feature_blocks: list[tuple[str, int]] | None = PrivateAttr(default=None)

    @field_validator("featurizers", mode="before")
    @classmethod
    def validate_featurizers(cls, value):
        """
        Validate and construct the list of featurizers.

        Accepts a dict of {type: params} entries, a list of featurizer
        instances, or a list mixing instances with dicts. Dict entries in a
        list are either {type: ..., params: ...} wrappers (the AnvilSection
        style, params optional) or single entries mapping the registry type
        to its params, as in the dict form.

        Parameters
        ----------
        value : dict or list
            Dictionary of featurizer types and parameters, or a list of
            featurizer instances and type/parameter entries.

        Returns
        -------
        list
            Sorted list of featurizer instances.

        """
        processed_featurizers = []
        if isinstance(value, dict):
            # dict form: keys are registry types, values are parameter dicts
            # see #595: this whole branch goes away once the dict form is
            # deprecated in favor of the {type, params} list form used elsewhere
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                processed_featurizers.append(feat_class(**feat_params))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, FeaturizerBase):
                    processed_featurizers.append(item)
                elif isinstance(item, dict):
                    feat_type, feat_params = _parse_featurizer_entry(item)
                    feat_class = get_featurizer_class(feat_type)
                    processed_featurizers.append(feat_class(**feat_params))
                else:
                    raise TypeError(
                        "Featurizer list entries must be featurizer instances or "
                        f"dicts of type/params, got {type(item)}."
                    )
        else:
            # Or raise an error if the type is unexpected
            return value

        # Reject same-class duplicates up front: per-key transforms (e.g.
        # per-block PCA) key blocks by featurizer name and cannot tell
        # same-class blocks apart
        names = [feat.__class__.__name__ for feat in processed_featurizers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "FeatureConcatenator cannot combine multiple featurizers of the same "
                f"class: {duplicates}. Per-key transforms cannot disambiguate same-class blocks."
            )

        # Sort the featurizers by class name so the block order is deterministic
        return sorted(processed_featurizers, key=lambda f: f.__class__.__name__)

    def feature_blocks(self) -> list[tuple[str, int]]:
        """
        Return the feature blocks recorded by the most recent ``featurize`` call.

        Returns
        -------
        list of tuple
            Pairs of (block key, feature width) covering the concatenated
            matrix in column order; nested concatenators are flattened.

        Raises
        ------
        RuntimeError
            If ``featurize`` has not been called yet.

        """
        if self._cached_feature_blocks is None:
            raise RuntimeError(
                "feature_blocks() requires featurize() to have been called first."
            )
        return self._cached_feature_blocks

    def featurize(self, smiles: list[str]) -> np.ndarray:
        """
        Featurize a list of SMILES strings using all featurizers and concatenate the results.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to featurize.

        Returns
        -------
        np.ndarray
            Concatenated feature array for all SMILES.

        """
        features = []
        indices = []
        blocks: list[tuple[str, int]] = []
        for feat in self.featurizers:
            feat_res, idx = feat.featurize(smiles)
            features.append(feat_res)
            indices.append(idx)

            # A nested concatenator already flattened its own children's blocks;
            # reuse them instead of collapsing them into one block for this key
            if isinstance(feat, FeatureConcatenator):
                blocks.extend(feat.feature_blocks())
            else:
                key = getattr(feat, "type", type(feat).__name__)
                blocks.append((key, feat_res.shape[1]))
        self._cached_feature_blocks = blocks

        return self.concatenate(features, indices)

    @staticmethod
    def concatenate(feats: list[ArrayLike], indices: list[np.ndarray]) -> np.ndarray:
        """
        Concatenate a list of feature arrays, keeping only features present in all datasets.

        Parameters
        ----------
        feats : list of array-like
            List of feature arrays to concatenate.
        indices : list of np.ndarray
            List of index arrays indicating valid entries for each feature array.

        Returns
        -------
        tuple
            Tuple of (concatenated feature array, common indices).

        """
        # If the input arrays are 1d, make them 2d
        feats = [
            feat.reshape(1, -1) if len(feat.shape) == 1 else feat for feat in feats
        ]

        # Use indices to mask out the features that are not present in all datasets
        common_indices = reduce(np.intersect1d, indices)

        # Filter features to only include common indices
        filtered_feats = []
        for feat, idx in zip(feats, indices):
            # Find where common_indices are in idx
            mask = np.isin(idx, common_indices)
            filtered_feats.append(feat[mask])

        # Handle 1d features from single input by making them 2, concatenate column wise
        concat_feats = np.concatenate(filtered_feats, axis=1)
        return (
            concat_feats,
            common_indices,
        )
