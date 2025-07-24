from openadmet.models.anvil.data_spec import DataSpec
from openadmet.models.tests.datafiles import intake_cat, test_csv, nan_data


def test_data_spec_from_csv():
    data_spec = DataSpec(
        type="intake",
        resource=test_csv,
        cat_entry="test_data",
        target_cols=["data1"],
        input_col="SMILES"
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

def test_data_spec_dropna():
    data_spec = DataSpec(
        type="intake",
        resource=nan_data,
        target_cols=["OPENADMET_LOGAC50"],
        input_col="OPENADMET_CANONICAL_SMILES",
        dropna=True
    )
    data_spec2 = DataSpec(
        type="intake",
        resource=nan_data,
        target_cols=["OPENADMET_LOGAC50"],
        input_col="OPENADMET_CANONICAL_SMILES",
        dropna=False
    )

    target, smiles = data_spec.read()
    target2, smiles2, = data_spec2.read()
    assert len(target) == 3333
    assert len(smiles) == 3333
    assert len(target2) == 7196
    assert len(smiles2) == 7196
