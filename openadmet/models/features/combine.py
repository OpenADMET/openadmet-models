"""Combine features from multiple featurizers into a single feature array."""

from collections.abc import Iterable
from functools import reduce

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field, field_validator

from openadmet.models.features.feature_base import (
    FeaturizerBase,
    featurizers,
    get_featurizer_class,
)


def _parse_featurizer_entry(item: dict) -> tuple[str, dict]:
    """
    Parse a featurizer entry dict into a (type, params) pair.

    Both accepted shapes: the AnvilSection-style wrapper {"type": ..., "params": {...}}
    and the single entry mapping the registry type directly to its params, as used
    by the dict form.

    Parameters
    ----------
    item : dict
        A single featurizer entry from a list-form featurizers value.

    Returns
    -------
    tuple
        Pair of (registry type, parameter dict).

    Raises
    ------
    ValueError
        If the entry matches neither shape.

    """
    if "type" in item:
        extra = set(item) - {"type", "params"}
        if extra:
            raise ValueError(
                f"Unsupported keys in featurizer entry: {sorted(extra)}. "
                "Entries may only use 'type' and 'params'."
            )
        return item["type"], item.get("params", {})
    if len(item) == 1:
        return next(iter(item.items()))
    raise ValueError(
        "Featurizer list entries must be {type: ..., params: ...} wrappers or "
        f"single-type entries, got keys: {list(item.keys())}."
    )


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

    featurizers: list[FeaturizerBase] = Field(
        ..., description="List of featurizers to concatenate"
    )

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

    def feature_blocks(self, probe: Iterable[str]) -> list[tuple[str, int]]:
        """
        Flatten the feature blocks of all child featurizers in concatenation order.

        Parameters
        ----------
        probe : Iterable[str]
            Input rows (e.g., SMILES) passed through to each child's probe.

        Returns
        -------
        list of tuple
            Pairs of (block key, feature width) covering the concatenated
            matrix in column order; nested concatenators are flattened.

        """
        blocks: list[tuple[str, int]] = []
        for feat in self.featurizers:
            blocks.extend(feat.feature_blocks(probe))
        return blocks

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
        for feat in self.featurizers:
            feat_res, idx = feat.featurize(smiles)
            features.append(feat_res)
            indices.append(idx)

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
