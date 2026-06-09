import pytest

from openadmet.models.anvil.specification import DataSpec
from openadmet.models.tests.unit.datafiles import intake_cat, nan_data, test_csv


def test_data_spec_from_csv():
    """
    Validate loading data from a CSV file via DataSpec.

    Ensures that the data loader correctly reads the specified CSV, extracts the target and SMILES columns,
    and returns them as expected.
    """
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
    """
    Validate loading data from an Intake catalog.

    Intake allows for declarative data loading. This test checks that DataSpec can correctly interface
    with an Intake catalog to retrieve data.
    """
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
    """
    Test the `dropna` functionality in DataSpec.

    Verifies that rows with missing values in target columns are dropped when dropna=True,
    and preserved when dropna=False. This is critical for handling real-world datasets which often contain gaps.
    """
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
