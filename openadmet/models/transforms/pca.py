"""Principal component analysis transforms for reducing feature dimensionality."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, PrivateAttr, field_validator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from openadmet.models.transforms.transform_base import (
    TransformBase,
    check_block_keys,
    transforms,
)


@transforms.register("PCATransform")
class PCATransform(TransformBase):
    """
    Apply principal component analysis to one or more feature blocks.

    One PCA over a concatenated matrix spends its components on whichever
    block carries the most raw variance, so a wide fingerprint block crowds a
    narrow descriptor block out of the retained dimensions. Fitting one PCA
    per block gives each featurizer its own component budget.

    ``n_components`` selects the mode:

    - Global: an int fits a single PCA over the entire feature matrix.
    - Per-block: a dict maps block keys to component counts and fits one PCA
      per block, concatenating the projected blocks in ``feature_blocks``
      order, typically one block per featurizer in a FeatureConcatenator
      output.
    - Passthrough: within the dict form, a count of None leaves that block
      unreduced, which suits blocks that are already narrow and carry meaning
      column by column, such as the predictions of a pretrained model.

    The transform is fitted on the train features only and applied to
    validation, test, and inference features, so PCA loadings never see
    held-out data. Missing values are handled by an optional imputation step
    ahead of the PCA within each block, since PCA itself cannot see NaN.

    Attributes
    ----------
    n_components : int or dict
        Number of PCA components. An int applies one PCA over the whole
        matrix; a dict maps featurizer block keys to per-block component
        counts and requires ``feature_blocks`` at fit time. A per-block value
        of None leaves that block unreduced.
    impute_strategy : str
        Optional imputation applied to each block before its PCA. Options are
        'none' (default), 'mean', or 'median'.
    random_seed : int or None
        Random seed for the PCA solvers, by default 42, matching the
        procedure-level default a workflow would otherwise fill in. Threaded to
        ``random_state``, which matters because sklearn picks the randomized
        solver for wide matrices with a large reduction, exactly the per-block
        fingerprint case. Set to None only to opt into an unseeded solver.

    """

    n_components: int | dict[str, int | None] = Field(
        ...,
        description=(
            "Number of PCA components, globally or per feature block; a "
            "per-block None passes that block through unreduced"
        ),
    )
    impute_strategy: Literal["none", "mean", "median"] = (
        "none"  # PCA cannot see NaN, so imputation runs ahead of it per block
    )
    random_seed: int | None = 42

    # Store PCA blocks: (block key, start col, end col, pipeline)
    _pca_blocks: list | None = PrivateAttr(default=None)

    @field_validator("n_components")
    @classmethod
    def validate_n_components(cls, value):
        """Validate that component counts are positive ints or per-block nulls."""
        # Int form: one PCA over the whole matrix
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"n_components must be >= 1, got {value}.")

            return value

        # Dict form: one PCA per block, so every entry needs its own count
        if isinstance(value, dict):
            if not value:
                raise ValueError("n_components dict must have at least one entry.")

            for key, dims in value.items():
                # None is the passthrough marker, so it skips the count check
                if dims is None:
                    continue

                if not isinstance(dims, int) or dims < 1:
                    raise ValueError(
                        f"n_components for block '{key}' must be an int >= 1 "
                        f"or null to pass the block through, got {dims}."
                    )

            return value

        raise TypeError(
            f"n_components must be an int or a dict of block keys to ints, got {type(value)}."
        )

    def required_block_keys(self) -> set[str] | None:
        """
        Return the block keys the per-block form is configured against.

        Returns
        -------
        set of str or None
            The ``n_components`` keys for the dict form, or None for the int
            form, which fits one PCA over the whole matrix and so needs no
            particular block layout.

        """
        if isinstance(self.n_components, dict):
            return set(self.n_components)
        return None

    def _build_pipeline(self, dims: int) -> Pipeline:
        """Build the (optional imputer plus) PCA pipeline for one block."""
        steps = []

        # Imputation runs inside the pipeline so it is fitted on train rows only,
        # the same as the PCA that follows it
        if self.impute_strategy != "none":
            steps.append(("impute", SimpleImputer(strategy=self.impute_strategy)))

        steps.append(("pca", PCA(n_components=dims, random_state=self.random_seed)))

        return Pipeline(steps)

    def _check_2d(self, X: np.ndarray) -> None:
        """
        Raise if X is not a 2D feature matrix.

        Featurizers return a 1D array for single-row input, and block slicing
        (``X[:, start:stop]``) raises a bare IndexError on one. Checking up
        front names the offending shape instead.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}.")

    def _validate_blocks(
        self,
        feature_blocks: list[tuple[str, int]] | None,
        n_features: int,
    ) -> list[tuple[str, int]]:
        """
        Validate the feature block layout for per-block PCA.

        Checks that blocks are present, uniquely keyed, cover the input width
        exactly, and match the n_components keys in both directions.

        Parameters
        ----------
        feature_blocks : list of tuple, optional
            Pairs of (block key, block width) in column order.
        n_features : int
            Number of columns in the input matrix.

        Returns
        -------
        list of tuple
            The validated block list.

        Raises
        ------
        ValueError
            If blocks are missing, duplicated, misaligned, or mismatched with
            the n_components keys.

        """
        # Without a layout there is no way to know where one block ends and the
        # next begins, and guessing would silently misalign the columns
        if feature_blocks is None:
            raise ValueError(
                "Per-block n_components requires feature_blocks describing the "
                "feature layout. Supply feature_blocks from the featurizer, keep "
                "PCATransform first among width-changing transforms, or use an "
                "int n_components for a single PCA over the whole matrix."
            )

        # Blocks are addressed by key, so a repeated key is ambiguous
        keys = [key for key, _ in feature_blocks]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate feature block keys: {duplicates}. Featurizer blocks "
                "must be uniquely keyed for per-block PCA."
            )

        # The blocks must tile the matrix exactly; a short or long sum means the
        # layout describes a different matrix than the one being fitted
        total = sum(width for _, width in feature_blocks)
        if total != n_features:
            raise ValueError(
                f"Feature block widths sum to {total} but the input has "
                f"{n_features} columns. feature_blocks describe the raw featurizer "
                "layout, so a width-changing transform earlier in the sequence "
                "requires an int n_components here."
            )

        # Every block needs a component count and every count needs a block
        check_block_keys(self.required_block_keys(), set(keys), "n_components")

        return list(feature_blocks)

    def fit(
        self,
        X: np.ndarray,
        feature_blocks: list[tuple[str, int]] | None = None,
        *args,
        **kwargs,
    ) -> PCATransform:
        """
        Fit the PCA pipeline(s) on X.

        Parameters
        ----------
        X : np.ndarray
            Train feature matrix of shape (n_samples, n_features).
        feature_blocks : list of tuple, optional
            Pairs of (block key, block width) in column order, required when
            ``n_components`` is a dict. Ignored by the int form.
        *args
            Additional positional arguments (not used).
        **kwargs
            Additional keyword arguments (not used).

        Returns
        -------
        PCATransform
            The fitted transform instance.

        """
        X = np.asarray(X)
        self._check_2d(X)
        n_samples, n_features = X.shape

        # Per-block form: walk the blocks in column order, fitting one PCA each
        if isinstance(self.n_components, dict):
            blocks = self._validate_blocks(feature_blocks, n_features)
            fitted_blocks = []
            cursor = 0

            for key, width in blocks:
                dims = self.n_components[key]

                # Block passes through without transformation
                if dims is None:
                    fitted_blocks.append((key, cursor, cursor + width, None))
                    cursor += width
                    continue

                # A block cannot yield more components than its own rank
                if dims > min(n_samples, width):
                    raise ValueError(
                        f"n_components for block '{key}' is {dims}; it must be "
                        f"at most min(train rows, block width) = "
                        f"min({n_samples}, {width})."
                    )

                # Record the column span alongside the pipeline so transform can
                # slice the same way without recomputing the layout
                pipeline = self._build_pipeline(dims).fit(X[:, cursor : cursor + width])
                fitted_blocks.append((key, cursor, cursor + width, pipeline))
                cursor += width

        # Int form: a single PCA spanning every column, recorded as one block
        else:
            dims = self.n_components
            if dims > min(n_samples, n_features):
                raise ValueError(
                    f"n_components is {dims}; it must be at most "
                    f"min(train rows, feature width) = min({n_samples}, {n_features})."
                )
            pipeline = self._build_pipeline(dims).fit(X)
            fitted_blocks = [(None, 0, n_features, pipeline)]

        # A None key marks the whole-matrix case, which no block key can address
        self._pca_blocks = fitted_blocks

        return self

    def transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Project X onto the fitted PCA components.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        *args
            Additional positional arguments (not used).
        **kwargs
            Additional keyword arguments (not used).

        Returns
        -------
        np.ndarray
            Transformed matrix of shape (n_samples, total output components).

        Raises
        ------
        RuntimeError
            If the transform has not been fitted yet.

        """
        if self._pca_blocks is None:
            raise RuntimeError(
                "The PCA transform has not been fitted yet. "
                "Fit it on the train partition first."
            )
        X = np.asarray(X)
        self._check_2d(X)

        # Ensure widths and block boundaries align
        expected_width = self._pca_blocks[-1][2]
        if X.shape[1] != expected_width:
            raise ValueError(
                f"Input has {X.shape[1]} columns but the fitted transform "
                f"expects {expected_width}. The transform assumes the same "
                "column layout at transform time as at fit time."
            )
        # Transform each block in order
        outputs = [
            X[:, start:stop]
            if pipeline is None
            else pipeline.transform(X[:, start:stop])
            for (_, start, stop, pipeline) in self._pca_blocks
        ]

        return np.concatenate(outputs, axis=1)
