"""Unit tests for the ApplicabilityDomainMetrics evaluator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from openadmet.models.applicability_domain.scaffold import (
    ScaffoldApplicabilityDomain,
)
from openadmet.models.eval.applicability_domain import ApplicabilityDomainMetrics
from openadmet.models.eval.eval_base import get_eval_class

BENZENE = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccc(Cl)cc1", "CCc1ccccc1"]


@pytest.fixture
def ad_path(tmp_path):
    """A fitted domain where benzene compounds bound at 1.0 and misc at 3.0."""
    smiles = BENZENE + ["C1CCCCC1"]
    errors = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 3.0])
    ad = ScaffoldApplicabilityDomain(min_count=4).fit(smiles, errors)
    path = tmp_path / "ad.pkl"
    ad.save(path)
    return path


def test_evaluator_registered():
    """The evaluator resolves through the registry."""
    assert get_eval_class("ApplicabilityDomainMetrics") is ApplicabilityDomainMetrics


def test_evaluate_computes_bounds_and_coverage(ad_path):
    """In-domain and out-of-domain test points get their respective bounds."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    # one in-domain (benzene), one out-of-domain (pyridine)
    X_test = ["Cc1ccccc1", "c1ccncc1"]
    y_true = np.array([[5.0], [5.0]])
    # benzene error 0.5 within its ~0.24 bound -> not covered;
    # pyridine error 0.5 within the global 3.0 bound -> covered
    y_pred = np.array([[5.5], [5.5]])

    ev.evaluate(y_true=y_true, y_pred=y_pred, X_test=X_test)

    metrics = ev.report()
    assert metrics["task_0"]["n_compounds"] == 2
    assert metrics["task_0"]["frac_in_domain"] == 0.5
    assert_allclose(metrics["task_0"]["ad_coverage"], 0.5)
    assert_allclose(metrics["task_0"]["ad_coverage_in_domain"], 0.0)
    assert_allclose(metrics["task_0"]["ad_coverage_out_domain"], 1.0)


def test_evaluate_requires_smiles(ad_path):
    """Missing X_test SMILES fails loudly."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    with pytest.raises(ValueError, match="X_test"):
        ev.evaluate(y_true=np.array([1.0]), y_pred=np.array([1.0]))


def test_report_writes_json_and_assignments(ad_path, tmp_path):
    """write=True produces the metrics JSON and per-compound CSV."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    ev.evaluate(
        y_true=np.array([[5.0]]),
        y_pred=np.array([[5.1]]),
        X_test=["Cc1ccccc1"],
    )
    ev.report(write=True, output_dir=tmp_path)
    assert (tmp_path / "applicability_domain_metrics.json").exists()
    assert (tmp_path / "applicability_domain_assignments.csv").exists()


def test_evaluate_rejects_mismatched_y_pred(ad_path):
    """A short y_pred must not broadcast silently over the test set."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    with pytest.raises(ValueError, match="equal shape"):
        ev.evaluate(
            y_true=np.array([[5.0], [5.0]]),
            y_pred=np.array([[5.0]]),
            X_test=["Cc1ccccc1", "c1ccncc1"],
        )


def test_evaluate_masks_missing_values(ad_path):
    """Missing targets or predictions are excluded from coverage denominators."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    X_test = ["Cc1ccccc1", "c1ccncc1"]
    y_true = np.array([[5.0], [np.nan]])
    y_pred = np.array([[5.5], [5.5]])
    ev.evaluate(y_true=y_true, y_pred=y_pred, X_test=X_test)

    m = ev.report()["task_0"]
    # only the benzene row is valid. Its 0.5 error exceeds the ~0.24 bound
    assert m["n_valid"] == 1
    assert m["n_missing"] == 1
    assert_allclose(m["ad_coverage"], 0.0)
    assert np.isnan(m["ad_coverage_out_domain"])
    # the missing row keeps a NaN error and an unset covered flag in the CSV
    assert np.isnan(ev._assignments["task_0_abs_err"].iloc[1])
    assert ev._assignments["task_0_covered"].isna().iloc[1]


def test_evaluate_rejects_wrong_target_labels(ad_path):
    """label count must match the number of task columns."""
    ev = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    with pytest.raises(ValueError, match="target_labels"):
        ev.evaluate(
            y_true=np.array([[5.0]]),
            y_pred=np.array([[5.0]]),
            X_test=["Cc1ccccc1"],
            target_labels=["a", "b"],
        )


def test_evaluator_instances_do_not_share_state(ad_path):
    """Metrics from one evaluation must not leak into another instance."""
    e1 = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    e2 = ApplicabilityDomainMetrics(ad_path=str(ad_path), n_resamples=10)
    e1.evaluate(
        y_true=np.array([[5.0]]),
        y_pred=np.array([[5.1]]),
        X_test=["Cc1ccccc1"],
    )
    assert e1.report()
    assert e2.report() == {}
