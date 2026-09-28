"""
Similarity-based applicability domain.

Alternative to the scaffold-keyed domain: a query is in-domain when its maximum
Tanimoto similarity to any training compound clears a threshold, and its bound
is the error percentile of those similar neighbors. Exists so the scaffold
approach can be evaluated against a continuous-similarity baseline, per the
discussion on https://github.com/OpenADMET/openadmet-models/issues/502.
"""

from os import PathLike
from typing import Any

import joblib
import numpy as np
from loguru import logger


class TanimotoApplicabilityDomain:
    """
    Per-compound error bounds from Tanimoto nearest neighbors.

    Parameters
    ----------
    similarity_threshold : float
        Minimum Tanimoto coefficient for a training compound to count as a
        neighbor and for a query to be in-domain.
    error_percentile : float
        Percentile of neighbor absolute errors used for the bound, and for
        the global out-of-domain bound when no neighbor clears the threshold.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Morgan fingerprint length.

    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        error_percentile: float = 95.0,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        """Initialize the domain with similarity and percentile parameters."""
        self.similarity_threshold = similarity_threshold
        self.error_percentile = error_percentile
        self.radius = radius
        self.n_bits = n_bits
        self._smiles: list[str] | None = None
        self._abs_errors: np.ndarray | None = None
        self._fps: list | None = None
        self.global_bound: float | None = None

    def _fingerprint(self, smiles: str):
        """Return a Morgan fingerprint, or None when the SMILES does not parse."""
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        ).GetFingerprint(mol)

    def fit(self, smiles: Any, abs_errors: Any) -> "TanimotoApplicabilityDomain":
        """
        Store the training fingerprints and errors the bounds are derived from.

        Parameters
        ----------
        smiles : array-like of str
            SMILES strings for the compounds whose errors were measured,
            typically the training set.
        abs_errors : array-like of float
            Absolute errors for each compound, same order as ``smiles``.

        Returns
        -------
        TanimotoApplicabilityDomain
            The fitted instance.

        """
        if isinstance(smiles, str):
            smiles = [smiles]
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
        smiles = list(smiles[finite])
        abs_errors = abs_errors[finite]

        fps, keep_smiles, keep_errors = [], [], []
        n_dropped = 0
        for s, e in zip(smiles, abs_errors):
            fp = self._fingerprint(s)
            if fp is None:
                n_dropped += 1
                continue
            fps.append(fp)
            keep_smiles.append(s)
            keep_errors.append(e)
        if n_dropped:
            logger.warning(f"{n_dropped} compound(s) with unparseable SMILES dropped")

        if not fps:
            raise ValueError("No valid SMILES to fit on.")

        self._smiles = keep_smiles
        self._abs_errors = np.asarray(keep_errors)
        self._fps = fps
        self.global_bound = float(
            np.nanpercentile(self._abs_errors, self.error_percentile)
        )
        return self

    @property
    def fitted(self) -> bool:
        """Whether the domain has been fit."""
        return self.global_bound is not None

    def _neighbor_mask(self, smiles: str) -> np.ndarray:
        """Boolean mask over training compounds above the similarity threshold."""
        from rdkit import DataStructs

        fp = self._fingerprint(smiles)
        if fp is None:
            return np.zeros(len(self._fps), dtype=bool)
        sims = DataStructs.BulkTanimotoSimilarity(fp, self._fps)
        return np.asarray(sims) >= self.similarity_threshold

    def bound(self, smiles: Any) -> np.ndarray:
        """
        Assign an error bound to each query compound.

        The bound is the error percentile over training compounds above the
        similarity threshold, or the global bound when none qualify.

        Parameters
        ----------
        smiles : array-like of str
            Query SMILES strings.

        Returns
        -------
        np.ndarray
            Error bound per compound.

        """
        if not self.fitted:
            raise ValueError("TanimotoApplicabilityDomain is not fit yet.")

        if isinstance(smiles, str):
            smiles = [smiles]

        out = []
        for s in smiles:
            mask = self._neighbor_mask(s)
            if mask.any():
                out.append(
                    float(
                        np.nanpercentile(self._abs_errors[mask], self.error_percentile)
                    )
                )
            else:
                out.append(self.global_bound)
        return np.array(out)

    def is_in_domain(self, smiles: Any) -> np.ndarray:
        """
        Whether each query has a training neighbor above the threshold.

        Parameters
        ----------
        smiles : array-like of str
            Query SMILES strings.

        Returns
        -------
        np.ndarray of bool

        """
        if not self.fitted:
            raise ValueError("TanimotoApplicabilityDomain is not fit yet.")

        if isinstance(smiles, str):
            smiles = [smiles]

        return np.array([self._neighbor_mask(s).any() for s in smiles])

    def save(self, path: PathLike = "applicability_domain_tanimoto.pkl"):
        """Serialize the fitted domain with joblib."""
        if not self.fitted:
            raise ValueError("Cannot save an unfit TanimotoApplicabilityDomain.")
        joblib.dump(
            {
                "similarity_threshold": self.similarity_threshold,
                "error_percentile": self.error_percentile,
                "radius": self.radius,
                "n_bits": self.n_bits,
                "smiles": self._smiles,
                "abs_errors": self._abs_errors,
                "global_bound": self.global_bound,
            },
            path,
        )

    @classmethod
    def load(cls, path: PathLike) -> "TanimotoApplicabilityDomain":
        """Load a serialized domain; fingerprints are recomputed on load."""
        state = joblib.load(path)
        instance = cls(
            similarity_threshold=state["similarity_threshold"],
            error_percentile=state["error_percentile"],
            radius=state["radius"],
            n_bits=state["n_bits"],
        )
        instance.fit(state["smiles"], state["abs_errors"])
        # preserve the exact stored global bound rather than recomputing
        instance.global_bound = state["global_bound"]
        return instance
