"""Combine features from multiple featurizers into a single feature array."""

import warnings
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


def _block_key(feat: FeaturizerBase) -> str:
    """
    Return the feature block key for a single featurizer.

    Parameters
    ----------
    feat : FeaturizerBase
        The featurizer whose block key is wanted.

    Returns
    -------
    str
        The featurizer's registry type, falling back to its class name.

    """
    return getattr(feat, "type", type(feat).__name__)


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    """
    Concatenate features from multiple featurizers into a single feature array.

    Attributes
    ----------
    featurizers : list of FeaturizerBase
        At least two featurizer instances to concatenate; concatenating fewer
        is a no-op, so use the featurizer directly instead. Same-class
        featurizers are rejected because per-key transforms key blocks by
        featurizer name and cannot disambiguate same-class blocks.

    """

    provides_feature_blocks: ClassVar[bool] = True

    featurizers: list[FeaturizerBase] = Field(
        ...,
        min_length=2,
        description="List of at least two featurizers to concatenate",
    )
    _cached_feature_blocks: list[tuple[str, int]] | None = PrivateAttr(default=None)

    @field_validator("featurizers", mode="before")
    @classmethod
    def validate_featurizers(cls, value):
        """
        Construct featurizer instances from the accepted input shapes.

        Accepts a list of featurizer instances, ``{type: ..., params: ...}``
        entries (the AnvilSection wrapper form, params optional), or a mix of
        the two. A whole-field dict mapping registry types to their params is
        also accepted but deprecated.

        Parameters
        ----------
        value : dict or list
            List of featurizer instances and type/parameter entries, or the
            deprecated dictionary of featurizer types and parameters.

        Returns
        -------
        list
            List of featurizer instances, in the order given.

        """
        # Reject a bare featurizer before anything else: pydantic models define
        # __iter__, so one passed instead of a list would otherwise be coerced
        # to an empty list and accepted as a concatenator over nothing
        if isinstance(value, FeaturizerBase):
            raise ValueError(
                "`featurizers` takes a list of featurizers, not a single "
                f"featurizer; wrap it as [{type(value).__name__}(...)]."
            )

        # Container for live featurizers
        processed_featurizers = []

        # Deprecated dict path, still read because saved recipe YAMLs use it  (see #595)
        if isinstance(value, dict):
            # Deprecation warning
            warnings.warn(
                "The whole-field dict form for `featurizers` is deprecated; use a "
                "list of {type: ..., params: ...} entries instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Instantiate each featurizer from the dict; a bare `TypeName:` with
            # no params parses as None, so treat it as an empty mapping (`or {}`)
            for feat_type, feat_params in value.items():
                feat_class = get_featurizer_class(feat_type)
                processed_featurizers.append(feat_class(**(feat_params or {})))

        # List path
        elif isinstance(value, list):
            for item in value:
                # Code path: featurizer instance, just append it
                if isinstance(item, FeaturizerBase):
                    processed_featurizers.append(item)

                # YAML path: dict entry
                # Instantiate the featurizer from the type/params dict
                elif isinstance(item, dict):
                    # Without `type` there is no registry key to resolve against
                    if "type" not in item:
                        raise ValueError(
                            "Featurizer list entries must be {type: ..., params: ...} "
                            f"wrappers, got keys: {list(item.keys())}."
                        )

                    # Get the class from the registry
                    feat_class = get_featurizer_class(item["type"])

                    # Instantiate and append, same bare `TypeName:` with no params
                    # applies, so treat it as an empty mapping (`or {}`)
                    processed_featurizers.append(
                        feat_class(**(item.get("params") or {}))
                    )

                # Invalid type path
                else:
                    raise ValueError(
                        "Featurizer list entries must be featurizer instances or "
                        f"dicts of type/params, got {type(item)}."
                    )
        else:
            # Not a shape this validator builds from; hand it back for pydantic
            # to check against the declared list[FeaturizerBase]
            return value

        return processed_featurizers

    @field_validator("featurizers", mode="after")
    @classmethod
    def reject_duplicates_and_sort(cls, value):
        """
        Reject same-class featurizers and fix the block order.

        Runs on the validated list, so it holds for every input shape,
        including sequences pydantic coerced to a list on its own.

        Parameters
        ----------
        value : list of FeaturizerBase
            The constructed featurizers, in the order given.

        Returns
        -------
        list
            The featurizers sorted by class name.

        Raises
        ------
        ValueError
            If two featurizers share a class.

        """
        # Per-key transforms (e.g. per-block PCA) key blocks by featurizer name
        # and cannot tell same-class blocks apart
        names = [feat.__class__.__name__ for feat in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "FeatureConcatenator cannot combine multiple featurizers of the same "
                f"class: {duplicates}. Per-key transforms cannot disambiguate same-class blocks."
            )

        # Sort by class name so the block order is deterministic
        return sorted(value, key=lambda f: f.__class__.__name__)

    def feature_block_keys(self) -> list[str]:
        """
        Return the feature block keys without featurizing.

        Block widths are only knowable once ``featurize`` has run, but the keys
        come from the featurizer types alone, so they are available at
        construction time and can be checked against a transform's per-block
        configuration before any data is loaded.

        Returns
        -------
        list of str
            Block keys in the order ``feature_blocks`` will report them;
            nested concatenators are flattened the same way.

        """
        keys: list[str] = []
        for feat in self.featurizers:
            # A nested concatenator contributes its children's keys, not one key
            # for itself, matching how featurize flattens the block list
            if isinstance(feat, FeatureConcatenator):
                keys.extend(feat.feature_block_keys())
            else:
                keys.append(_block_key(feat))
        return keys

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
                blocks.append((_block_key(feat), feat_res.shape[1]))
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
