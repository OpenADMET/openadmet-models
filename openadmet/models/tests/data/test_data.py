from openadmet.models.anvil.data_spec import DataSpec
from openadmet.models.tests.datafiles import intake_cat, test_csv


def test_data_spec_from_csv():
    data_spec = DataSpec(
        type="intake",
        resource=test_csv,
        cat_entry="test_data",
        target_col="data1",
        smiles_col="SMILES",
    )
    target, smiles = data_spec.read()
    assert len(target) == 30
    assert len(smiles) == 30


def test_data_spec_from_intake():
    data_spec = DataSpec(
        type="intake",
        resource=intake_cat,
        cat_entry="subsel",
        target_col="data1",
        smiles_col="SMILES",
    )
    target, smiles = data_spec.read()
    assert len(target) == 30
    assert len(smiles) == 30
