import pytest

from openadmet.models.anvil.data_spec import DataSpec
from openadmet.models.tests.unit.datafiles import intake_cat, nan_data, test_csv


def test_data_spec_from_csv():
    data_spec = DataSpec(
        type="intake",
        resource=test_csv,
        cat_entry="test_data",
        target_cols=["data1"],
        input_col="SMILES",
    )
    target, smiles = data_spec.read()
    assert len(target) == 30
    assert len(smiles) == 30


def test_data_spec_from_intake():
    data_spec = DataSpec(
        type="intake",
        resource=intake_cat,
        cat_entry="subsel",
        target_cols=["data1"],
        input_col="SMILES",
    )
    target, smiles = data_spec.read()
    assert len(target) == 30
    assert len(smiles) == 30


@pytest.mark.parametrize("dropna, expected_length", [(True, 3333), (False, 7196)])
def test_data_spec_dropna(dropna, expected_length):
    data_spec = DataSpec(
        type="intake",
        resource=nan_data,
        target_cols=["OPENADMET_LOGAC50"],
        input_col="OPENADMET_CANONICAL_SMILES",
        dropna=dropna,
    )

    target, smiles = data_spec.read()

    assert len(target) == expected_length
    assert len(smiles) == expected_length
