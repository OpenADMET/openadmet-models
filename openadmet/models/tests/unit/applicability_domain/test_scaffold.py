"""Unit tests for ScaffoldApplicabilityDomain."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from openadmet.models.applicability_domain.scaffold import (
    ScaffoldApplicabilityDomain,
)

# Benzene-scaffolded compounds; all map to c1ccccc1
BENZENE = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccc(Cl)cc1", "CCc1ccccc1"]
# Cyclohexane scaffold, single occurrence so it lands in the misc bin
MISC = ["C1CCCCC1"]
# A different cyclic system absent from training
UNSEEN = ["c1ccncc1"]
BROKEN = ["not_a_smiles"]


@pytest.fixture
def fitted_ad():
    """Domain fit on 5 benzene compounds plus 1 misc cyclohexane."""
    smiles = BENZENE + MISC
    errors = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 2.0])
    return ScaffoldApplicabilityDomain(min_count=4).fit(smiles, errors)


def test_primary_scaffold_gets_own_bound(fitted_ad):
    """A benzene-family query receives the benzene-cluster error bound."""
    benzene_bound = float(np.percentile([0.1, 0.2, 0.15, 0.25, 0.1], 95.0))
    assert_allclose(fitted_ad.bound(["Cc1cc(O)ccc1"]), [benzene_bound])
    assert fitted_ad.is_in_domain(["Cc1cc(O)ccc1"])[0]


def test_misc_and_unseen_get_global_bound(fitted_ad):
    """Misc-scaffold, unseen-scaffold, and unparseable queries get the global bound."""
    # misc pool is the lone cyclohexane error
    assert fitted_ad.global_bound == 2.0
    bounds = fitted_ad.bound(MISC + UNSEEN + BROKEN)
    assert_allclose(bounds, [2.0, 2.0, 2.0])
    assert not fitted_ad.is_in_domain(MISC + UNSEEN)[0]
    assert not fitted_ad.is_in_domain(BROKEN)[0]


def test_acyclic_training_compounds_fall_into_misc():
    """Acyclic SMILES have no Murcko scaffold and join the misc bin."""
    smiles = ["CCCC", "CCO", "CCN", "CCCCl"]
    ad = ScaffoldApplicabilityDomain(min_count=4).fit(smiles, [0.5] * 4)
    assert not ad.is_in_domain(["CCCC"])[0]
    assert_allclose(ad.bound(["CCCC"]), [ad.global_bound])


def test_below_threshold_scaffold_uses_global():
    """A scaffold under min_count gets the global bound, not its own."""
    smiles = BENZENE[:3] + MISC
    ad = ScaffoldApplicabilityDomain(min_count=4).fit(smiles, [0.1] * 4)
    # benzene has only 3 members, under the min_count of 4
    assert not ad.is_in_domain(["Cc1ccccc1"])[0]


def test_unfit_and_length_mismatch_raise():
    """Unfit calls and mismatched inputs fail loudly."""
    ad = ScaffoldApplicabilityDomain()
    with pytest.raises(ValueError, match="not fit"):
        ad.bound(["CCO"])
    with pytest.raises(ValueError, match="equal length"):
        ad.fit(["CCO", "CCN"], [0.1])


def test_save_load_roundtrip(tmp_path, fitted_ad):
    """A saved domain reproduces the same bounds after load."""
    path = tmp_path / "ad.pkl"
    fitted_ad.save(path)
    loaded = ScaffoldApplicabilityDomain.load(path)
    queries = BENZENE[:1] + UNSEEN + BROKEN
    assert_allclose(loaded.bound(queries), fitted_ad.bound(queries))
    assert loaded.primary_bounds == fitted_ad.primary_bounds
