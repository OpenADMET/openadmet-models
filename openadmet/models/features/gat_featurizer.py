from typing import Optional, List, Any
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from loguru import logger

try:
    from rdkit import Chem
except ImportError:

    logger.warning(
        "RDKit library not found. GraphFeaturizer will not be functional. "
        "Please install RDKit (e.g., conda install -c conda-forge rdkit)."
    )
    Chem = None # Set to None to check its availability later

# Assuming feature_base.py is in the same directory or accessible via PYTHONPATH
try:
    from .feature_base import FeaturizerBase, featurizers
except ImportError:
    logger.warning(
        "FeaturizerBase not found. GATGraphFeaturizer will not be registered. "
        "Ensure feature_base.py is accessible."
    )
    # Define dummy versions if not found, so the class definition doesn't break
    class FeaturizerBase:
        def __init__(self, *args, **kwargs):
            pass
    class featurizers:
        @staticmethod
        def register(name):
            def decorator(cls):
                return cls
            return decorator

@featurizers.register("GATGraphFeaturizer")
class GATGraphFeaturizer(FeaturizerBase):
    """
    Featurizer to convert SMILES strings into graph Data objects suitable for GAT-like models.
    It extracts atom features and bond features using RDKit.
    """

    def __init__(self):
        super().__init__()
        self._prepare()

    def _prepare(self):
        """
        Check if RDKit is available.
        """
        if Chem is None:
            raise ImportError(
                "RDKit library is not installed, which is essential for GATGraphFeaturizer. "
                "Please install RDKit."
            )
        logger.info("GATGraphFeaturizer initialized with RDKit.")

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

    def featurize(self, smiles_list: List[str], y_list: Optional[List[Any]] = None, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0):
        """
        Featurize a list of SMILES strings into a PyTorch Geometric DataLoader.

        Args:
            smiles_list: List of SMILES strings.
            y_list: Optional list of target values. Values will be attempted to be cast to float.
            batch_size: Batch size for DataLoader.
            shuffle: Whether to shuffle the data in DataLoader.
            num_workers: Number of worker processes for DataLoader.

        Returns:
            Tuple of (DataLoader, None, List[Data]). The DataLoader for training,
            scaler (None for graphs), and the list of Data objects.
            Invalid SMILES or problematic molecules will be skipped (a warning will be logged).
        """
        if Chem is None:
            logger.error("RDKit not available. Cannot featurize SMILES.")
            return DataLoader([]), None

        data_objects = []
        # Convert y_list to a list if it's a pandas Series to avoid indexing issues
        if y_list is not None:
            if hasattr(y_list, 'tolist'):
                y_list = y_list.tolist()
            elif hasattr(y_list, 'values'):
                y_list = y_list.values.tolist()
        
        for i, smiles_str in enumerate(smiles_list):
            y_val = None
            if y_list is not None and i < len(y_list):
                try:
                    # Handle string representations of lists like '[5.1]'
                    target_str = str(y_list[i])
                    if target_str.startswith('[') and target_str.endswith(']'):
                        # Remove brackets and try to convert
                        target_str = target_str.strip('[]')
                    y_val = float(target_str)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert target value '{y_list[i]}' to float for SMILES '{smiles_str}' at index {i}. "
                        "This target will be skipped."
                    )
                    # Continue processing but y_val remains None
            
            data = self._featurize_single_molecule(smiles_str, y_val)
            if data is not None:
                data_objects.append(data)
        
        # Create DataLoader
        dataloader = DataLoader(
            data_objects, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        
        # Return dataloader, scaler (None for GAT), and dataset (data_objects)
        return dataloader, None, data_objects  # Return None for scaler as we don't use one for graphs
