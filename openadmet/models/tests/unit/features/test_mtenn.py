import numpy as np
import pytest
import pandas as pd
from openadmet.models.features.mtenn import MTENNFeaturizer, MTENNDataset
from openadmet.models.tests.unit.datafiles import ligand_pose

@pytest.fixture()
def cyp3a4_pose():
    """Fixture for ligand pose"""
    return ligand_pose

def test_mtenn_dataset(cyp3a4_pose):
    """Test MTENNDataset class for basic functionality"""
    # Create a mock dataset, with two identical complexes and a single target value
    complexes = [cyp3a4_pose, cyp3a4_pose]
    y = np.asarray([42, 43])
    dataset = MTENNDataset(complexes, y, ligand_resname="X5Y", ignore_h=True)

    # Check the length of the dataset
    assert len(dataset) == 2
    # Check the shape of the features

    feats = next(iter(dataset))

    assert feats["Y"] == 42
    assert feats["lig_mask"].numpy().shape == (3695,)
    assert feats["pos"].numpy().shape == (3695, 3)
    assert feats["Z"].numpy().shape == (3695,)
    assert feats["B"].numpy().shape == (3695,)

    # check the ligand mask, 38 atoms in the ligand
    assert feats["lig_mask"].numpy().sum() == 38



def test_mtenn_featurizer(cyp3a4_pose):
    ft = MTENNFeaturizer(
        ligand_resname="X5Y",
        ignore_h=True,
    )

    dataloader, _, _, _  = ft.featurize([cyp3a4_pose], pd.Series([42]))

    # Check the length of the dataloader
    assert len(dataloader) == 1
    # Check the shape of the features
    feats, y  = next(iter(dataloader))

    assert y.item() == 42
    assert feats[0]["lig"].numpy().shape == (3695,)
    assert feats[0]["pos"].numpy().shape == (3695, 3)
    assert feats[0]["z"].numpy().shape == (3695,)

    ##The following are not returned from featurizer
    #assert feats["B"].numpy().shape == (1, 3695)
    # check the ligand mask, 38 atoms in the ligand
    #assert feats["lig_mask"].numpy().sum() == 38
