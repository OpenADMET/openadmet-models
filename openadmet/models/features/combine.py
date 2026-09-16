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
        The featurizer's alias if it has one, otherwise its registry type,
        falling back to its class name.

    """
    # Read both through getattr: a featurizer unpickled from a model saved
    # before aliases existed carries no alias attribute at all
    return getattr(feat, "alias", None) or getattr(feat, "type", type(feat).__name__)


def _flatten_block_keys(feats: list[FeaturizerBase]) -> list[str]:
    """
    Return the block keys a list of featurizers emits, flattening nested concatenators.

    Parameters
    ----------
    feats : list of FeaturizerBase
        The featurizers whose block keys are wanted, in column order.

    Returns
    -------
    list of str
        One key per emitted block; a nested concatenator contributes its
        children's keys rather than one key for itself, matching how
        ``featurize`` flattens the block list.

    """
    keys: list[str] = []
    for feat in feats:
        if isinstance(feat, FeatureConcatenator):
            keys.extend(_flatten_block_keys(feat.featurizers))
        else:
            keys.append(_block_key(feat))
    return keys


@featurizers.register("FeatureConcatenator")
class FeatureConcatenator(FeaturizerBase):
    """
    Concatenate features from multiple featurizers into a single feature array.

    Attributes
    ----------
    featurizers : list of FeaturizerBase
        At least two featurizer instances to concatenate; concatenating fewer
        is a no-op, so use the featurizer directly instead. Two featurizers
        emitting the same block key are rejected because per-key transforms
        address blocks by that key and could not tell the two apart; give
        same-class featurizers distinct aliases to combine them.

    """

    provides_feature_blocks: ClassVar[bool] = True

    featurizers: list[FeaturizerBase] = Field(
        ...,
        min_length=2,
        description="List of at least two featurizers to concatenate",
    )
    _cached_feature_blocks: list[tuple[str, int]] | None = PrivateAttr(default=None)

    @staticmethod
    def _resolve_entry_alias(item: dict, params: dict) -> str:
        """
        Return the alias for one list entry, rejecting two conflicting spellings.

        An alias may be written beside ``type`` or inside ``params``; giving
        both with different values leaves no honest way to pick one.
        """
        entry_alias = item["alias"]
        param_alias = params.get("alias")

        if param_alias is not None and param_alias != entry_alias:
            raise ValueError(
                f"Featurizer entry for {item['type']} sets alias {entry_alias!r} "
                f"beside `type` and {param_alias!r} inside `params`; give it once."
            )

        return entry_alias

    @field_validator("featurizers", mode="before")
    @classmethod
    def validate_featurizers(cls, value):
        """
        Construct featurizer instances from the accepted input shapes.

        Accepts a list of featurizer instances, ``{type: ..., params: ...}``
        entries (the AnvilSection wrapper form, params optional), or a mix of
        the two. An entry may carry an ``alias`` beside its ``type``, which is
        folded into the params it is constructed with. A whole-field dict
        mapping registry types to their params is also accepted but deprecated.

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
        # A bare featurizer is iterable, so it would coerce to an empty list
        if isinstance(value, FeaturizerBase):
            raise ValueError(
                "`featurizers` takes a list of featurizers, not a single "
                f"featurizer; wrap it as [{type(value).__name__}(...)]."
            )

        # Container for live featurizers
        processed_featurizers = []

        # Deprecation warning for the dict path, still read by saved recipe YAMLs (see #595)
        if isinstance(value, dict):
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

                    # A bare `TypeName:` with no params parses as None, so treat
                    # it as an empty mapping (`or {}`)
                    params = dict(item.get("params") or {})

                    # `alias` reads naturally beside `type` in a recipe, but it is
                    # a field on the featurizer, so fold it into the params
                    if "alias" in item:
                        params["alias"] = cls._resolve_entry_alias(item, params)

                    processed_featurizers.append(feat_class(**params))

                # Invalid type path
                else:
                    raise ValueError(
                        "Featurizer list entries must be featurizer instances or "
                        f"dicts of type/params, got {type(item)}."
                    )
        # Pass validation through to pydantic
        else:
            return value

        return processed_featurizers

    @field_validator("alias")
    @classmethod
    def reject_concatenator_alias(cls, value):
        """
        Reject an alias on the concatenator itself.

        A concatenator emits its children's blocks rather than one block of its
        own, so an alias here would name no block while still reordering the
        children it holds.

        Parameters
        ----------
        value : str or None
            The configured alias.

        Returns
        -------
        None
            The unset alias.

        Raises
        ------
        ValueError
            If an alias is given.

        """
        if value is not None:
            raise ValueError(
                f"FeatureConcatenator takes no alias (got {value!r}); its feature "
                "blocks are named by the featurizers it contains, so put the alias "
                "on those instead."
            )

        return value

    @field_validator("featurizers", mode="after")
    @classmethod
    def reject_duplicates_and_sort(cls, value):
        """
        Reject featurizers sharing a block key and fix the block order.

        Blocks are sorted by key so that featurization and transformation stay
        consistent across recipes naming the same featurizers. Featurizers
        sharing a key are rejected because per-block transforms address blocks
        by that key and could not tell two apart. The check covers the flattened
        key list, so a nested concatenator's child cannot collide unnoticed with
        an outer sibling.

        Parameters
        ----------
        value : list of FeaturizerBase
            The constructed featurizers, in the order given.

        Returns
        -------
        list
            The featurizers sorted by block key.

        Raises
        ------
        ValueError
            If two featurizers emit the same block key.

        """
        # Per-key transforms (e.g. per-block PCA) key blocks by featurizer name
        # and cannot tell two blocks sharing a name apart
        keys = _flatten_block_keys(value)
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                "FeatureConcatenator cannot combine featurizers that emit the same "
                f"feature block key: {duplicates}. Per-key transforms cannot "
                "disambiguate them, so give each one a distinct `alias`."
            )

        # Sort by block key
        return sorted(value, key=_block_key)

    def feature_block_keys(self) -> list[str]:
        """
        Return the feature block keys without featurizing.

        Block widths are only knowable once ``featurize`` has run, but the keys
        come from the featurizer aliases and types alone, so they are available
        at construction time and can be checked against a transform's per-block
        configuration before any data is loaded.

        Returns
        -------
        list of str
            Block keys in the order ``feature_blocks`` will report them;
            nested concatenators are flattened the same way.

        """
        return _flatten_block_keys(self.featurizers)

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
