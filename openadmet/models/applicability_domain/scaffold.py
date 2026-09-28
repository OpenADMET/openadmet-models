"""
Scaffold-based applicability domain.

Maps compounds to Bemis-Murcko scaffolds and assigns each query an empirical
error bound. Compounds matching a well-represented training scaffold get the
per-scaffold error percentile; compounds whose scaffold is absent or pooled
into the low-frequency miscellaneous bin get a conservative global bound.
Implements the post-hoc variant of the proposal in
https://github.com/OpenADMET/openadmet-models/issues/502 where error profiles
come from a finished model's errors rather than an in-CV training loop.
"""

from os import PathLike
from typing import Any

import joblib
import numpy as np
from loguru import logger


class ScaffoldApplicabilityDomain:
    """
    Per-compound error bounds keyed by Bemis-Murcko scaffold.

    Parameters
    ----------
    min_count : int
        Minimum number of training compounds sharing a scaffold for that
        scaffold to get its own bound. Scaffolds below this threshold are
        pooled into the miscellaneous bin.
    error_percentile : float
        Percentile of absolute errors used for both per-scaffold bounds and
        the global out-of-domain bound.

    Attributes
    ----------
    primary_bounds : dict
        Maps canonical scaffold SMILES to that scaffold's error bound.
    misc_scaffolds : set
        Canonical scaffold SMILES of the pooled low-frequency compounds.
    global_bound : float
        Out-of-domain error bound assigned to miscellaneous and unseen
        scaffolds.

    """

    def __init__(self, min_count: int = 5, error_percentile: float = 95.0):
        """Initialize the domain with binning and percentile parameters."""
        self.min_count = min_count
        self.error_percentile = error_percentile
        self.primary_bounds: dict[str, float] = {}
        self.misc_scaffolds: set[str] = set()
        self.global_bound: float | None = None

    @staticmethod
    def scaffold_smiles(smiles: str) -> str | None:
        """
        Return the canonical Bemis-Murcko scaffold SMILES for a compound.

        Returns None when the SMILES cannot be parsed or has no scaffold
        (e.g. acyclic molecules).
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffoldSmiles(mol=mol)
        return scaffold if scaffold else None

    def fit(
        self, smiles: Any, abs_errors: Any, ood_errors: Any = None
    ) -> "ScaffoldApplicabilityDomain":
        """
        Fit scaffold error profiles from a model's absolute errors.

        Parameters
        ----------
        smiles : array-like of str
            SMILES strings for the compounds whose errors were measured,
            typically the training set.
        abs_errors : array-like of float
            Absolute errors for each compound, same order as ``smiles``.
        ood_errors : array-like of float, optional
            Errors measured on compounds whose scaffolds were held out
            entirely, e.g. the held-out folds of scaffold-grouped CV. When
            provided, this pool defines the global OOD bound instead of the
            miscellaneous-bin errors, matching the two-pool design in #502.

        Returns
        -------
        ScaffoldApplicabilityDomain
            The fitted instance.

        """
        smiles = np.asarray(smiles)
        abs_errors = np.asarray(abs_errors, dtype=float)
        if smiles.shape[0] != abs_errors.shape[0]:
            raise ValueError(
                f"smiles and abs_errors must have equal length, got "
                f"{smiles.shape[0]} and {abs_errors.shape[0]}"
            )

        # drop rows where every task's error is non-finite so missing targets
        # do not poison the percentiles
        if abs_errors.ndim > 1:
            finite = np.isfinite(abs_errors).any(axis=1)
        else:
            finite = np.isfinite(abs_errors)
        smiles, abs_errors = smiles[finite], abs_errors[finite]

        scaffolds = [self.scaffold_smiles(s) for s in smiles]
        n_failed = sum(s is None for s in scaffolds)
        if n_failed:
            logger.warning(
                f"{n_failed} compound(s) had no extractable scaffold; "
                "they join the miscellaneous bin"
            )

        # Count scaffold frequencies; unscaffoldable compounds share a key
        keys = np.array([s if s is not None else "" for s in scaffolds])
        unique, counts = np.unique(keys, return_counts=True)
        primary = set(unique[counts >= self.min_count]) - {""}

        # Per-scaffold bounds for primary scaffolds
        self.primary_bounds = {
            scaffold: float(
                np.nanpercentile(abs_errors[keys == scaffold], self.error_percentile)
            )
            for scaffold in primary
        }
        self.misc_scaffolds = set(unique) - primary

        # Global OOD bound: an explicit extrapolation pool when provided,
        # otherwise the misc-bin errors approximate held-out compounds
        if ood_errors is not None:
            pool = np.asarray(ood_errors, dtype=float)
        else:
            misc_mask = np.isin(keys, list(self.misc_scaffolds))
            pool = abs_errors[misc_mask] if misc_mask.any() else abs_errors
        self.global_bound = float(np.nanpercentile(pool, self.error_percentile))

        return self

    @property
    def fitted(self) -> bool:
        """Whether the domain has been fit."""
        return self.global_bound is not None

    def bound(self, smiles: Any) -> np.ndarray:
        """
        Assign an error bound to each query compound.

        Parameters
        ----------
        smiles : array-like of str
            Query SMILES strings.

        Returns
        -------
        np.ndarray
            Error bound per compound: the per-scaffold percentile for
            primary-scaffold matches, the global bound otherwise.

        """
        if not self.fitted:
            raise ValueError("ScaffoldApplicabilityDomain is not fit yet.")

        return np.array(
            [
                self.primary_bounds.get(
                    self.scaffold_smiles(s) or "", self.global_bound
                )
                for s in smiles
            ]
        )

    def is_in_domain(self, smiles: Any) -> np.ndarray:
        """
        Whether each query compound maps to a primary (in-domain) scaffold.

        Parameters
        ----------
        smiles : array-like of str
            Query SMILES strings.

        Returns
        -------
        np.ndarray of bool
            True where the compound's scaffold is a primary scaffold.

        """
        if not self.fitted:
            raise ValueError("ScaffoldApplicabilityDomain is not fit yet.")

        return np.array(
            [(self.scaffold_smiles(s) or "") in self.primary_bounds for s in smiles]
        )

    @classmethod
    def from_predictions(
        cls,
        smiles: Any,
        y_true: Any,
        y_pred: Any,
        min_count: int = 5,
        error_percentile: float = 95.0,
    ) -> "ScaffoldApplicabilityDomain":
        """
        Fit a domain directly from a trained model's predictions.

        Convenience wrapper for the post-hoc workflow: after training, compute
        absolute errors on the training compounds and fit the domain.

        Parameters
        ----------
        smiles : array-like of str
            SMILES strings for the compounds with predictions, typically the
            training set.
        y_true : array-like of float
            Ground truth values.
        y_pred : array-like of float
            Predicted values, same order as ``smiles``.
        min_count : int
            Minimum scaffold frequency for a per-scaffold bound.
        error_percentile : float
            Percentile of absolute errors used for bounds.

        Returns
        -------
        ScaffoldApplicabilityDomain
            The fitted instance.

        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have equal shape, got "
                f"{y_true.shape} and {y_pred.shape}"
            )
        return cls(min_count=min_count, error_percentile=error_percentile).fit(
            smiles, np.abs(y_true - y_pred)
        )

    def save(self, path: PathLike = "applicability_domain.pkl"):
        """Serialize the fitted domain with joblib."""
        if not self.fitted:
            raise ValueError("Cannot save an unfit ScaffoldApplicabilityDomain.")
        joblib.dump(
            {
                "min_count": self.min_count,
                "error_percentile": self.error_percentile,
                "primary_bounds": self.primary_bounds,
                "misc_scaffolds": self.misc_scaffolds,
                "global_bound": self.global_bound,
            },
            path,
        )

    @classmethod
    def load(cls, path: PathLike) -> "ScaffoldApplicabilityDomain":
        """Load a serialized domain."""
        state = joblib.load(path)
        instance = cls(
            min_count=state["min_count"], error_percentile=state["error_percentile"]
        )
        instance.primary_bounds = state["primary_bounds"]
        instance.misc_scaffolds = state["misc_scaffolds"]
        instance.global_bound = state["global_bound"]
        return instance
