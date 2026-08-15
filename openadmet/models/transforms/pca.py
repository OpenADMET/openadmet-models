"""Principal component analysis transforms for reducing feature dimensionality."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, PrivateAttr, field_validator
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from openadmet.models.transforms.transform_base import TransformBase, transforms


@transforms.register("PCATransform")
class PCATransform(TransformBase):
    """
    Apply principal component analysis to one or more feature blocks.

    With an int ``n_components``, a single PCA is fitted over the entire
    feature matrix. With a dict mapping block keys to component counts, one
    PCA is fitted per block and the projected blocks are concatenated in the
    block order from ``feature_blocks``, typically one block per featurizer in
    a FeatureConcatenator output.

    The transform is fitted on the train features only and applied to
    validation, test, and inference features, so PCA loadings never see
    held-out data. Missing values are handled by an optional imputation step
    ahead of the PCA within each block, since PCA itself cannot see NaN.

    Per-block PCA slices matrices by train-time block boundaries, so every
    matrix it applies to must carry the same column layout. Featurizers that
    emit a batch-dependent column set (e.g. Mordred, whose columns are the
    union of descriptors that compute on the given rows) can violate this
    across partitions; prefer featurizers with fixed-width output such as
    fingerprints. The fit-time width check fails loudly when a mismatch
    occurs instead of silently misaligning columns.

    Attributes
    ----------
    n_components : int or dict
        Number of PCA components. An int applies one PCA over the whole
        matrix; a dict maps featurizer block keys to per-block component
        counts and requires ``feature_blocks`` at fit time.
    impute_strategy : str
        Optional imputation applied to each block before its PCA. Options are
        'none' (default), 'mean', or 'median'.
    random_seed : int or None
        Random seed for the PCA solvers, by default None. Threaded to
        ``random_state`` so stochastic solvers stay reproducible.

    """

    n_components: int | dict[str, int] = Field(
        ..., description="Number of PCA components, globally or per feature block"
    )
    impute_strategy: Literal["none", "mean", "median"] = (
        "none"  # PCA cannot see NaN, so imputation runs ahead of it per block
    )
    random_seed: int | None = None
    # The workflow forwards feature_blocks only to transforms that declare this
    accepts_feature_blocks: ClassVar[bool] = True

    # Fitted state: list of (block key, start col, end col, fitted pipeline) in
    # column order; the key is None for the single-PCA case over the whole matrix
    _pca_blocks: list | None = PrivateAttr(default=None)

    @field_validator("n_components")
    @classmethod
    def validate_n_components(cls, value):
        """Validate that component counts are positive ints, globally or per block."""
        if isinstance(value, bool) or (
            isinstance(value, dict) and any(isinstance(v, bool) for v in value.values())
        ):
            raise TypeError("n_components values must be ints, not booleans.")
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"n_components must be >= 1, got {value}.")
            return value
        if isinstance(value, dict):
            if not value:
                raise ValueError("n_components dict must have at least one entry.")
            for key, dims in value.items():
                if not isinstance(dims, int) or dims < 1:
                    raise ValueError(
                        f"n_components for block '{key}' must be an int >= 1, got {dims}."
                    )
            return value
        raise TypeError(
            f"n_components must be an int or a dict of block keys to ints, got {type(value)}."
        )

    def _build_pipeline(self, dims: int) -> Pipeline:
        """Build the (optional imputer plus) PCA pipeline for one block."""
        steps = []
        if self.impute_strategy != "none":
            steps.append(
                ("impute", SimpleImputer(strategy=self.impute_strategy))
            )
        steps.append(("pca", PCA(n_components=dims, random_state=self.random_seed)))
        return Pipeline(steps)

    def _check_2d(self, X: np.ndarray) -> None:
        """Raise if X is not a 2D feature matrix."""
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
        if feature_blocks is None:
            raise ValueError(
                "Per-block n_components requires feature_blocks describing the "
                "feature layout. Supply feature_blocks from the featurizer, keep "
                "PCATransform first among width-changing transforms, or use an "
                "int n_components for a single PCA over the whole matrix."
            )
        keys = [key for key, _ in feature_blocks]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate feature block keys: {duplicates}. Featurizer blocks "
                "must be uniquely keyed for per-block PCA."
            )
        total = sum(width for _, width in feature_blocks)
        if total != n_features:
            raise ValueError(
                f"Feature block widths sum to {total} but the input has "
                f"{n_features} columns. feature_blocks describe the raw featurizer "
                "layout, so a width-changing transform earlier in the sequence "
                "requires an int n_components here."
            )
        requested = set(self.n_components)
        available = set(keys)
        missing = sorted(requested - available)
        unexpected = sorted(available - requested)
        if missing or unexpected:
            raise ValueError(
                "n_components keys must exactly match the feature block keys. "
                f"blocks: {sorted(available)}; n_components: {sorted(requested)}; "
                f"missing: {missing}; unexpected: {unexpected}."
            )
        return list(feature_blocks)

    def fit(
        self,
        X: np.ndarray,
        feature_blocks: list[tuple[str, int]] | None = None,
        *args,
        **kwargs,
    ) -> "PCATransform":
        """
        Fit the PCA pipeline(s) on X.

        Parameters
        ----------
        X : np.ndarray
            Train feature matrix of shape (n_samples, n_features).
        feature_blocks : list of tuple, optional
            Pairs of (block key, block width) in column order, required when
            ``n_components`` is a dict. Ignored by the int form.

        Returns
        -------
        PCATransform
            The fitted transform instance.

        """
        X = np.asarray(X)
        self._check_2d(X)
        n_samples, n_features = X.shape

        if isinstance(self.n_components, dict):
            blocks = self._validate_blocks(feature_blocks, n_features)
            fitted_blocks = []
            cursor = 0
            for key, width in blocks:
                dims = self.n_components[key]
                if dims >= min(n_samples, width):
                    raise ValueError(
                        f"n_components for block '{key}' is {dims}; it must be "
                        f"smaller than min(train rows, block width) = "
                        f"min({n_samples}, {width})."
                    )
                pipeline = self._build_pipeline(dims).fit(X[:, cursor : cursor + width])
                fitted_blocks.append((key, cursor, cursor + width, pipeline))
                cursor += width
        else:
            dims = self.n_components
            if dims >= min(n_samples, n_features):
                raise ValueError(
                    f"n_components is {dims}; it must be smaller than "
                    f"min(train rows, feature width) = min({n_samples}, {n_features})."
                )
            pipeline = self._build_pipeline(dims).fit(X)
            fitted_blocks = [(None, 0, n_features, pipeline)]

        self._pca_blocks = fitted_blocks
        return self

    def transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Project X onto the fitted PCA components.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

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
        # Fit-time and transform-time matrices must share the exact column
        # layout; a mismatch means silently truncated or misaligned blocks
        expected_width = self._pca_blocks[-1][2]
        if X.shape[1] != expected_width:
            raise ValueError(
                f"Input has {X.shape[1]} columns but the fitted transform "
                f"expects {expected_width}. The transform assumes the same "
                "column layout at transform time as at fit time."
            )
        outputs = [
            pipeline.transform(X[:, start:stop])
            for (_, start, stop, pipeline) in self._pca_blocks
        ]
        return np.concatenate(outputs, axis=1)
