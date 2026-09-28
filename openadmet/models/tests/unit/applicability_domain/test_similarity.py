"""Unit tests for TanimotoApplicabilityDomain and domain comparisons."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from openadmet.models.applicability_domain.scaffold import (
    ScaffoldApplicabilityDomain,
)
from openadmet.models.applicability_domain.similarity import (
    TanimotoApplicabilityDomain,
)

BENZENE = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccc(Cl)cc1", "CCc1ccccc1"]


@pytest.fixture
def fitted_tad():
    """Domain fit on benzene-family compounds plus one distant structure."""
    smiles = BENZENE + ["C1CCCCC1"]
    errors = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 3.0])
    return TanimotoApplicabilityDomain(similarity_threshold=0.4).fit(smiles, errors)


def test_similar_query_gets_neighbor_bound(fitted_tad):
    """A query similar to the benzene cluster gets a bound from those neighbors."""
    bound = fitted_tad.bound(["Cc1ccccc1"])[0]
    # neighbors are the low-error benzenes, not the 3.0-error cyclohexane
    assert bound < 1.0
    assert fitted_tad.is_in_domain(["Cc1ccccc1"])[0]


def test_dissimilar_query_gets_global_bound(fitted_tad):
    """A query with no neighbors above threshold gets the global bound."""
    # cubane shares almost no Morgan bits with the training set
    bound = fitted_tad.bound(["C12C3C4C1C5C4C3C25"])[0]
    assert bound == fitted_tad.global_bound
    assert not fitted_tad.is_in_domain(["C12C3C4C1C5C4C3C25"])[0]


def test_unparseable_query_gets_global_bound(fitted_tad):
    """SMILES that do not parse cannot be fingerprinted; assign global."""
    assert fitted_tad.bound(["not_a_smiles"])[0] == fitted_tad.global_bound


def test_unfit_raises():
    """Unfit calls fail loudly."""
    tad = TanimotoApplicabilityDomain()
    with pytest.raises(ValueError, match="not fit"):
        tad.bound(["CCO"])
    with pytest.raises(ValueError, match="not fit"):
        tad.is_in_domain(["CCO"])


def test_save_load_roundtrip(tmp_path, fitted_tad):
    """A saved domain reproduces identical bounds after load."""
    path = tmp_path / "tad.pkl"
    fitted_tad.save(path)
    loaded = TanimotoApplicabilityDomain.load(path)
    queries = BENZENE[:1] + ["C1CCCCC1", "C12C3C4C1C5C4C3C25"]
    assert_allclose(loaded.bound(queries), fitted_tad.bound(queries))


def test_scaffold_vs_tanimoto_same_data():
    """Both domains produce bounds on the same data; they need not agree."""
    smiles = BENZENE + ["C1CCCCC1"]
    errors = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 3.0])
    sad = ScaffoldApplicabilityDomain(min_count=4).fit(smiles, errors)
    tad = TanimotoApplicabilityDomain().fit(smiles, errors)

    queries = ["Cc1ccccc1", "c1ccncc1", "not_a_smiles"]
    s_bounds = sad.bound(queries)
    t_bounds = tad.bound(queries)
    assert s_bounds.shape == t_bounds.shape == (3,)
    # toluene-like query: in-domain for both
    assert sad.is_in_domain(queries[:1])[0]
    assert tad.is_in_domain(queries[:1])[0]


def test_fit_applicability_domain_from_cv_data():
    """The CV evaluator's collected fold data fits a working domain."""
    from openadmet.models.eval.cross_validation import (
        PytorchLightningRepeatedKFoldCrossValidation,
    )

    cv_eval = PytorchLightningRepeatedKFoldCrossValidation(n_resamples=10)
    smiles = np.array(BENZENE + MISC_SMILES)
    cv_eval._ad_cv_data = (
        smiles,
        np.ones((len(smiles), 1)),
        np.ones((len(smiles), 1)) + np.array([0.1, 0.2, 0.15, 0.25, 0.1, 2.0])[:, None],
    )
    ad = cv_eval.fit_applicability_domain(min_count=4)
    assert ad.fitted
    assert ad.is_in_domain(["Cc1ccccc1"])[0]


MISC_SMILES = ["C1CCCCC1"]


def test_fit_applicability_domain_requires_collected_data():
    """fit_applicability_domain fails loudly before evaluate collects data."""
    from openadmet.models.eval.cross_validation import (
        PytorchLightningRepeatedKFoldCrossValidation,
    )

    cv_eval = PytorchLightningRepeatedKFoldCrossValidation(n_resamples=10)
    with pytest.raises(ValueError, match="collect_ad_errors"):
        cv_eval.fit_applicability_domain()
