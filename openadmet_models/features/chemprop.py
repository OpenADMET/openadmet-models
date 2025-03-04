from collections.abc import Iterable
from typing import Any

from openadmet_models.features.feature_base import MolfeatFeaturizer, featurizers



@featurizers.register("ChemPropFeaturizer")
class ChemPropFeaturizer():
    """
    ChemPropFeaturizer featurizer for molecules, relies on chemprop
    [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)] backend
    """




    def _prepare(self):
        """
        Prepare the featurizer
        """


    def featurize(self, smiles: Iterable[str], y: Iterable[any]) -> pytorch.DataLoader:
        """
        Featurize a list of SMILES strings
        """
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)])
        if self.normalize_targets:
            scaler = dataset.normalize_targets()
        dataloader = build_dataloader(dataset, num_workers=self.n_jobs)
        return dataloader, scaler
