"""
UMAP-based splitter with spatial optimization for leak-free CV splits.

This module provides UMAP-based CV splits using spatial assignment directly
on the embedding space. The optimization assigns samples to folds while:
1. Penalizing spatial scatter (keeping nearby molecules in same fold)
2. Ensuring class balance (each fold has both classes)
3. Allowing flexible fold sizes

Supports sklearn CV interface for use with nested cross-validation.
"""

import logging
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from umap import UMAP

logger = logging.getLogger(__name__)


class UMAPCVSplitter(BaseCrossValidator):
    """
    UMAP-based CV splitter implementing sklearn CV interface.

    Uses UMAP to reduce features to 2D, then assigns samples to folds
    while minimizing spatial scatter (keeping nearby samples together)
    and ensuring class balance.

    This is sklearn-compatible and can be used with nested CV.

    Attributes:
        n_splits: Number of CV folds
        random_seed: Random seed for reproducibility
        n_neighbors: Number of neighbors for UMAP (default: 100)
        min_dist: Minimum distance for UMAP (default: 0.0)
        umap_embeddings: UMAP 2D embeddings for each sample
        fold_assignments: Fold assignment for each sample

    """

    # Minimum points required for stable UMAP embedding
    MIN_POINTS_FOR_UMAP: int = 20

    def __init__(
        self,
        n_splits: int = 5,
        random_seed: int = 42,
        n_neighbors: int = 100,
        min_dist: float = 0.0,
    ):
        """
        Initialize UMAP CV splitter.

        Args:
            n_splits: Number of folds
            random_seed: Random seed
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP

        """
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_embeddings: np.ndarray | None = None
        self.fold_assignments: np.ndarray | None = None

    def split(self, X, y=None, groups=None):
        """
        Generate UMAP-based train/test splits.

        Args:
            X: Feature matrix
            y: Target array (used for stratification)
            groups: Group labels (not used)

        Yields:
            train_indices, test_indices: Arrays of indices

        """
        if y is None:
            raise ValueError("UMAP splitter requires y for stratification")

        # Compute embeddings and fold assignments
        embeddings = self._compute_umap_embeddings(X)
        fold_assignments = self._create_spatial_folds(embeddings, y)
        self.fold_assignments = fold_assignments

        # Yield train/test splits for each fold
        for fold_id in range(self.n_splits):
            test_idx = np.where(fold_assignments == fold_id)[0]
            train_idx = np.where(fold_assignments != fold_id)[0]
            yield train_idx, test_idx

    def get_n_splits(  # type: ignore
        self, X=None, y=None, groups=None
    ) -> int:
        """
        Return the number of splits.

        Args:
            X: Feature matrix (not used)
            y: Target array (not used)
            groups: Group labels (not used)

        Returns:
            Number of splits

        """
        return self.n_splits

    def __getstate__(self) -> dict[str, Any]:
        """
        Get state for serialization (e.g., YAML, pickle).

        Returns only the constructor parameters, excluding runtime state
        like computed embeddings and fold assignments.

        Returns:
            Dictionary containing serializable parameters

        """
        return {
            "n_splits": self.n_splits,
            "random_seed": self.random_seed,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Set state from deserialization (e.g., YAML, pickle).

        Reconstructs the object from serialized parameters.

        Args:
            state: Dictionary containing serialized parameters

        """
        self.n_splits = state["n_splits"]
        self.random_seed = state["random_seed"]
        self.n_neighbors = state["n_neighbors"]
        self.min_dist = state["min_dist"]
        self.umap_embeddings = None
        self.fold_assignments = None

    def _deduplicate_features(self, X: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
        """
        Deduplicate identical feature vectors.

        This fixes UMAP bug where identical points get different embeddings.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (unique_features, original_to_unique_map)

        """
        unique_features = []
        feature_to_unique_idx = {}
        original_to_unique_map = {}

        for i, feature_vec in enumerate(X):
            feature_tuple = tuple(feature_vec.tolist())

            if feature_tuple not in feature_to_unique_idx:
                # New unique feature vector
                unique_idx = len(unique_features)
                feature_to_unique_idx[feature_tuple] = unique_idx
                unique_features.append(feature_vec)
                original_to_unique_map[i] = unique_idx
            else:
                # Duplicate - map to existing unique index
                original_to_unique_map[i] = feature_to_unique_idx[feature_tuple]

        unique_X = np.array(unique_features)
        logger.debug(f"Deduplication: {len(X)} -> {len(unique_X)} unique features")

        return unique_X, original_to_unique_map

    def _compute_umap_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Compute UMAP embeddings with deduplication.

        Args:
            X: Feature matrix

        Returns:
            UMAP embeddings array

        """
        # Deduplicate features
        unique_X, original_to_unique_map = self._deduplicate_features(X)

        # Check minimum points
        if len(unique_X) < self.MIN_POINTS_FOR_UMAP:
            raise ValueError(
                f"UMAP requires at least {self.MIN_POINTS_FOR_UMAP} "
                f"unique points for stable embedding, got {len(unique_X)} "
                f"unique from {len(X)} total. Dataset has too many "
                f"duplicates for UMAP."
            )

        # Apply UMAP to unique features only
        logger.debug("Computing UMAP embedding")
        n_neighbors_adj = min(self.n_neighbors, len(unique_X) - 1)
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors_adj,
            min_dist=self.min_dist,
            metric="jaccard",
            random_state=self.random_seed,
        )
        unique_embeddings: np.ndarray = reducer.fit_transform(unique_X)  # type: ignore

        # Map embeddings back to original samples
        umap_embeddings = np.zeros((len(X), 2), dtype=np.float64)
        for orig_idx in range(len(X)):
            unique_idx = original_to_unique_map[orig_idx]
            umap_embeddings[orig_idx] = unique_embeddings[unique_idx]

        self.umap_embeddings = umap_embeddings
        return umap_embeddings

    def _create_spatial_folds(
        self, embeddings: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Create spatially-coherent folds with class balance.

        Args:
            embeddings: UMAP embeddings (n_samples, 2)
            y: Target labels

        Returns:
            Fold assignments array

        """
        # Create initial spatial clusters
        fold_assignments = self._create_initial_clusters(embeddings)

        # Enforce class balance
        fold_assignments = self._enforce_class_balance(embeddings, y, fold_assignments)

        # Log final statistics
        self._log_fold_statistics(y, fold_assignments)

        return fold_assignments

    def _create_initial_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Create initial spatial clusters using agglomerative clustering.

        Args:
            embeddings: UMAP embeddings (n_samples, 2)

        Returns:
            Initial fold assignments array

        """
        logger.debug(
            f"Creating {self.n_splits} spatially contiguous regions "
            f"with agglomerative clustering"
        )
        clustering = AgglomerativeClustering(n_clusters=self.n_splits, linkage="ward")
        initial_assignments = clustering.fit_predict(embeddings)
        return initial_assignments.copy()

    def _calculate_min_samples_per_fold(self, n_class_samples: int) -> int:
        """
        Calculate minimum samples per fold for a class.

        Uses half of perfectly balanced as the minimum to allow flexibility.

        Args:
            n_class_samples: Total samples for this class

        Returns:
            Minimum samples required per fold

        """
        return max(1, int((n_class_samples / self.n_splits) / 2))

    def _find_nearest_sample(
        self,
        embeddings: np.ndarray,
        available_indices: list[int],
        fold_indices: np.ndarray,
    ) -> int:
        """
        Find sample nearest to fold boundary.

        Args:
            embeddings: UMAP embeddings (n_samples, 2)
            available_indices: Indices of available samples
            fold_indices: Indices of samples in target fold

        Returns:
            Index of nearest sample

        """
        if len(fold_indices) == 0:
            # Fold is empty, just take the first available
            return available_indices[0]

        fold_embeddings = embeddings[fold_indices]
        available_embeddings = embeddings[available_indices]

        # Find min distance to fold for each candidate
        min_dists = []
        for avail_emb in available_embeddings:
            dists = np.linalg.norm(fold_embeddings - avail_emb, axis=1)
            min_dists.append(np.min(dists))

        # Choose closest sample
        return available_indices[np.argmin(min_dists)]

    def _enforce_class_balance(
        self,
        embeddings: np.ndarray,
        y: np.ndarray,
        fold_assignments: np.ndarray,
    ) -> np.ndarray:
        """
        Enforce minimum class balance in each fold.

        Iteratively reassigns samples to ensure each fold has minimum
        representation from each class.

        Args:
            embeddings: UMAP embeddings (n_samples, 2)
            y: Target labels
            fold_assignments: Current fold assignments

        Returns:
            Updated fold assignments array

        """
        # Skip class balancing for continuous targets (regression)
        target_type = type_of_target(y)
        if target_type == "continuous":
            logger.debug("Skipping class balance enforcement for continuous target")
            return fold_assignments

        unique_classes = np.unique(y)
        logger.debug("Ensuring class balance in each fold")

        max_iterations = 100
        for iteration in range(max_iterations):
            all_satisfied = True
            reassigned_this_iter = set()

            for class_val in unique_classes:
                class_mask = y == class_val
                n_class_samples = np.sum(class_mask)
                min_per_fold = self._calculate_min_samples_per_fold(n_class_samples)

                if iteration == 0:
                    logger.debug(
                        f"Class {class_val}: {n_class_samples} samples, "
                        f"min {min_per_fold} per fold"
                    )

                all_satisfied &= self._balance_class_across_folds(
                    embeddings,
                    y,
                    fold_assignments,
                    class_val,
                    class_mask,
                    min_per_fold,
                    reassigned_this_iter,
                )

            if all_satisfied:
                logger.debug(
                    f"Class balance satisfied after {iteration + 1} iterations"
                )
                break

        return fold_assignments

    def _balance_class_across_folds(
        self,
        embeddings: np.ndarray,
        y: np.ndarray,
        fold_assignments: np.ndarray,
        class_val: int,
        class_mask: np.ndarray,
        min_per_fold: int,
        reassigned_this_iter: set[int],
    ) -> bool:
        """
        Balance a single class across all folds.

        Args:
            embeddings: UMAP embeddings (n_samples, 2)
            y: Target labels
            fold_assignments: Current fold assignments
            class_val: Class value to balance
            class_mask: Boolean mask for this class
            min_per_fold: Minimum samples required per fold
            reassigned_this_iter: Set of already reassigned sample indices

        Returns:
            True if all folds satisfy minimum, False otherwise

        """
        all_satisfied = True

        for fold_id in range(self.n_splits):
            fold_mask = fold_assignments == fold_id
            n_class_in_fold = np.sum(class_mask & fold_mask)

            if n_class_in_fold < min_per_fold:
                all_satisfied = False

                # Get available samples for reassignment
                available_indices = self._get_available_samples(
                    class_mask, fold_mask, reassigned_this_iter
                )

                if len(available_indices) == 0:
                    logger.warning(
                        f"Cannot add samples of class {class_val} "
                        f"to fold {fold_id} (need {min_per_fold}, "
                        f"have {n_class_in_fold})"
                    )
                    continue

                # Find and reassign nearest sample
                fold_indices = np.where(fold_mask)[0]
                nearest_idx = self._find_nearest_sample(
                    embeddings, available_indices, fold_indices
                )

                fold_assignments[nearest_idx] = fold_id
                reassigned_this_iter.add(nearest_idx)
                n_class_in_fold += 1
                logger.debug(
                    f"  Reassigned sample {nearest_idx} "
                    f"(class {class_val}) to fold {fold_id} "
                    f"(now {n_class_in_fold}/{min_per_fold})"
                )

        return all_satisfied

    def _get_available_samples(
        self,
        class_mask: np.ndarray,
        fold_mask: np.ndarray,
        reassigned_this_iter: set[int],
    ) -> list[int]:
        """
        Get samples available for reassignment.

        Args:
            class_mask: Boolean mask for target class
            fold_mask: Boolean mask for target fold
            reassigned_this_iter: Set of already reassigned indices

        Returns:
            List of available sample indices

        """
        available_mask = class_mask & ~fold_mask
        available_indices = np.where(available_mask)[0]
        return [idx for idx in available_indices if idx not in reassigned_this_iter]

    def _log_fold_statistics(self, y: np.ndarray, fold_assignments: np.ndarray) -> None:
        """
        Log final fold statistics.

        Args:
            y: Target labels
            fold_assignments: Final fold assignments

        """
        target_type = type_of_target(y)
        is_continuous = target_type == "continuous"

        # For continuous targets, just show counts on one line
        if is_continuous:
            fold_counts = [np.sum(fold_assignments == f) for f in range(self.n_splits)]
            logger.info(
                f"Fold sizes: {', '.join(f'fold {i}: {c}' for i, c in enumerate(fold_counts))}"
            )
        else:
            # For classification, show class distribution per fold
            for f in range(self.n_splits):
                fold_mask = fold_assignments == f
                y_fold = y[fold_mask]
                unique_fold, counts_fold = np.unique(y_fold, return_counts=True)
                logger.info(
                    f"Fold {f}: {len(y_fold)} samples, "
                    f"classes={dict(zip(unique_fold, counts_fold))}"
                )


# Register YAML representer and constructor for UMAPCVSplitter
def _umap_cv_splitter_representer(
    dumper: yaml.Dumper, splitter: UMAPCVSplitter
) -> yaml.Node:
    """
    YAML representer for UMAPCVSplitter.

    Converts UMAPCVSplitter object to YAML-serializable dict.

    Args:
        dumper: YAML dumper instance
        splitter: UMAPCVSplitter instance

    Returns:
        YAML mapping node

    """
    return dumper.represent_mapping("!UMAPCVSplitter", splitter.__getstate__())


def _umap_cv_splitter_constructor(
    loader: yaml.Loader, node: yaml.Node
) -> UMAPCVSplitter:
    """
    YAML constructor for UMAPCVSplitter.

    Reconstructs UMAPCVSplitter object from YAML dict.

    Args:
        loader: YAML loader instance
        node: YAML node to construct from

    Returns:
        UMAPCVSplitter instance

    """
    params = loader.construct_mapping(node, deep=True)
    return UMAPCVSplitter(**params)


# Register with both SafeDumper/SafeLoader (used by safe_dump/safe_load)
# and default Dumper/Loader
yaml.add_representer(UMAPCVSplitter, _umap_cv_splitter_representer)
yaml.add_constructor("!UMAPCVSplitter", _umap_cv_splitter_constructor)
yaml.SafeDumper.add_representer(UMAPCVSplitter, _umap_cv_splitter_representer)
yaml.SafeLoader.add_constructor("!UMAPCVSplitter", _umap_cv_splitter_constructor)
