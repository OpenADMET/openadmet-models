"""Evaluator for scaffold-based applicability domain assignments."""

import json

import numpy as np
import pandas as pd
from pydantic import Field

from openadmet.models.applicability_domain.scaffold import (
    ScaffoldApplicabilityDomain,
)
from openadmet.models.eval.eval_base import EvalBase, evaluators
from openadmet.models.eval.utils import ensure_2d


@evaluators.register("ApplicabilityDomainMetrics")
class ApplicabilityDomainMetrics(EvalBase):
    """
    Report scaffold applicability-domain bounds and coverage for test compounds.

    Requires a fitted :class:`ScaffoldApplicabilityDomain` serialized with
    joblib, supplied via ``ad_path``. The workflow passes test SMILES as
    ``X_test``; each test point receives either its scaffold's error bound or
    the global out-of-domain bound, and coverage is the fraction of observed
    absolute errors within the assigned bound.

    Attributes
    ----------
    ad_path : str
        Path to a joblib-serialized ScaffoldApplicabilityDomain.
    use_wandb : bool
        Whether to use wandb for logging.
    _data : dict
        Stores computed metrics for each task.
    _assignments : pandas.DataFrame
        Per-compound bound assignments written alongside the metrics.

    """

    ad_path: str = Field(
        ..., description="Path to serialized ScaffoldApplicabilityDomain"
    )
    use_wandb: bool = Field(False, description="Whether to use wandb")
    _data: dict = {}
    _assignments: pd.DataFrame | None = None

    def evaluate(
        self,
        y_true=None,
        y_pred=None,
        X_test=None,
        target_labels=None,
        **kwargs,
    ):
        """
        Assign applicability-domain bounds to test compounds and score coverage.

        Parameters
        ----------
        y_true : array-like
            Ground truth test values.
        y_pred : array-like
            Predicted test values.
        X_test : array-like of str
            Test SMILES strings.
        target_labels : list of str, optional
            List of target labels for each task.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If required inputs are missing or shapes are inconsistent.

        """
        if y_true is None or y_pred is None or X_test is None:
            raise ValueError("Must provide `y_true`, `y_pred`, and `X_test` SMILES")

        ad = ScaffoldApplicabilityDomain.load(self.ad_path)

        bounds = ad.bound(X_test)
        in_domain = ad.is_in_domain(X_test)

        # Convert to numpy array if needed
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        y_pred = ensure_2d(y_pred)
        y_true = ensure_2d(y_true)

        if y_true.shape[0] != len(X_test):
            raise ValueError(
                f"`y_true` and `X_test` must have the same number of samples, got "
                f"{y_true.shape[0]} and {len(X_test)}"
            )

        n_tasks = y_true.shape[1]
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self._assignments = pd.DataFrame(
            {
                "smiles": list(X_test),
                "ad_bound": bounds,
                "in_domain": in_domain,
            }
        )

        for task_id, task_label in enumerate(target_labels):
            abs_err = np.abs(y_true[:, task_id] - y_pred[:, task_id])
            covered = abs_err <= bounds

            self._data[task_label] = {
                "n_compounds": int(len(X_test)),
                "frac_in_domain": float(np.mean(in_domain)),
                "ad_coverage": float(np.mean(covered)),
                "ad_coverage_in_domain": float(
                    np.mean(covered[in_domain]) if in_domain.any() else np.nan
                ),
                "ad_coverage_out_domain": float(
                    np.mean(covered[~in_domain]) if (~in_domain).any() else np.nan
                ),
            }

            self._assignments[f"{task_label}_abs_err"] = abs_err
            self._assignments[f"{task_label}_covered"] = covered

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation results.

        Parameters
        ----------
        write : bool, default=False
            Whether to write the report to disk.
        output_dir : Path or str, optional
            Directory to write the report to.

        Returns
        -------
        dict
            Dictionary of computed metrics.

        """
        if write:
            self.write_report(output_dir)

        return self._data

    def write_report(self, output_dir):
        """
        Write the evaluation report to disk.

        Parameters
        ----------
        output_dir : Path or str
            Directory to write the report to.

        """
        json_path = output_dir / "applicability_domain_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self._data, f, indent=2)

        if self._assignments is not None:
            self._assignments.to_csv(
                output_dir / "applicability_domain_assignments.csv", index=False
            )
