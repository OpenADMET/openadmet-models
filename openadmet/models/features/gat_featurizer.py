from typing import Optional, List, Any
from collections.abc import Iterable
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from loguru import logger

from rdkit import Chem

from .feature_base import FeaturizerBase, featurizers

@featurizers.register("GATGraphFeaturizer")
class GATGraphFeaturizer(FeaturizerBase):
    """
    Featurizer to convert SMILES strings into graph Data objects suitable for GAT-like models.
    It extracts atom features and bond features using RDKit.
    """
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0

    def _featurize_single_molecule(self, smiles: str, y_val: Optional[float] = None) -> Optional[Data]:
        """
        Converts a single SMILES string to a PyTorch Geometric Data object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None

        atom_features_list = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization()), # RDKit returns HybridizationType, convert to float
                float(atom.GetIsAromatic()),
                float(atom.GetMass()),
                float(atom.GetNumRadicalElectrons()),
                float(atom.IsInRing()),
            ]
            atom_features_list.append(features)

        if not atom_features_list: # Should not happen if MolFromSmiles was successful and molecule has atoms
            logger.warning(f"No atoms found for SMILES: {smiles} (mol object: {mol})")
            return None

        x = torch.tensor(atom_features_list, dtype=torch.float)

        edge_indices = []
        edge_features_list = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])

            bond_f = [
                float(bond.GetBondTypeAsDouble()),
                float(bond.GetIsAromatic()),
                float(bond.IsInRing()),
                float(bond.GetStereo()) # RDKit returns BondStereo, convert to float
            ]
            edge_features_list.extend([bond_f, bond_f])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features_list, dtype=torch.float) if edge_features_list else None
        else: # Handle molecules with a single atom (no bonds)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None # No edges, so no edge attributes

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if y_val is not None:
            data.y = torch.tensor([y_val], dtype=torch.float)

        return data

    def featurize(self, smiles: Iterable[str], y: Optional[Iterable[Any]] = None):
        """
        Featurize a list of SMILES strings into a PyTorch Geometric DataLoader.

        Args:
            smiles: An iterable of SMILES strings.
            y: Optional iterable of target values. Values will be attempted to be cast to float.

        Returns:
            Tuple of (DataLoader, indices, None, List[Data]). The DataLoader for training,
            indices of successfully featurized molecules, scaler (None for graphs), and the list of Data objects.
            Invalid SMILES or problematic molecules will be skipped (a warning will be logged).
        """

        data_objects = []
        successful_indices = []

        # Handle different types of y input no changing data_spec.py, making it more robust
        if y is not None:
            # If y is a pandas DataFrame, extract values from the first column
            if hasattr(y, 'values') and hasattr(y, 'iloc'):  # DataFrame check
                y = y.iloc[:, 0].tolist()  # Take first column and convert to list
            # If y is a pandas Series or other iterable, convert to list
            else:
                y = list(y)
        else:
            y = None

        for i, smi in enumerate(smiles):
            y_val = None
            if y is not None and i < len(y):
                try:
                    # Handle string representations of lists like '[5.1]'
                    target_str = str(y[i])
                    if target_str.startswith('[') and target_str.endswith(']'):
                        # Remove brackets and try to convert
                        target_str = target_str.strip('[]')
                    y_val = float(target_str)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert target value '{y[i]}' to float for SMILES '{smi}' at index {i}. "
                        "This target will be skipped."
                    )
                    # Continue processing but y_val remains None

            data = self._featurize_single_molecule(smi, y_val)
            if data is not None:
                data_objects.append(data)
                successful_indices.append(i)

        # Create DataLoader
        dataloader = DataLoader(
            data_objects,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        # Return dataloader, indices, scaler (None for GAT), and dataset (data_objects)
        return dataloader, np.array(successful_indices), None, data_objects
