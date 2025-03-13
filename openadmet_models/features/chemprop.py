from collections.abc import Iterable
from typing import Any

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from torch.utils.data import DataLoader

from openadmet_models.features.feature_base import FeaturizerBase, featurizers


@featurizers.register("ChemPropFeaturizer")
class ChemPropFeaturizer(FeaturizerBase):
    """
    ChemPropFeaturizer featurizer for molecules, relies on chemprop
    """

    normalize_targets: bool = True
    n_jobs: int = 4
    batch_size: int = 128
    shuffle: bool = False

    def _prepare(self):
        """
        Prepare the featurizer
        """

    def featurize(self, smiles: Iterable[str], y: Iterable[Any]) -> DataLoader:
        """
        Featurize a list of SMILES strings
        """
        y = y.to_numpy().reshape(-1, 1)
        dataset = MoleculeDataset(
            [MoleculeDatapoint.from_smi(smi, y_) for smi, y_ in zip(smiles, y)]
        )
        if self.normalize_targets:
            scaler = dataset.normalize_targets()
        else:
            scaler = None
        dataloader = build_dataloader(
            dataset,
            num_workers=self.n_jobs,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
        )
        return dataloader, scaler
