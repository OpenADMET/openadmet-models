from collections.abc import Iterable
from typing import Any
from pathlib import Path
from torch.utils.data import DataLoader
import MDAnalysis as mda
import torch
from typing import Union
from openadmet.models.features.feature_base import FeaturizerBase, featurizers
from torch.utils.data import Dataset
import numpy as np
import warnings
from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

def get_atomic_number(element: str) -> int:
    """
    Get the atomic number of an element
    """
    try:
        return ptable.GetAtomicNumber(element)
    except KeyError:
        raise ValueError(f"Element {element} not found in periodic table")

atomic_number_vfunc = np.vectorize(get_atomic_number)

class MTENNDataset(Dataset):
    """
    Custom dataset for MTENN models
    """

    def __init__(self, complexes: Iterable[Path], y: Iterable[Any], ligand_resname: Union[str, list[str]] = "LIG", ignore_h: bool = True):
        """
        """
        self.complexes = complexes
        if isinstance(ligand_resname, str):
            ligand_resname = [ligand_resname] * len(complexes)
        elif len(ligand_resname) != len(complexes):
            raise ValueError("ligand_resnames must be a string or a list of the same length as complexes")
        if len(complexes) != len(y):
            raise ValueError("complexes and y must be the same length")
        self.ligand_resname = ligand_resname
        self.y = y
        self.ignore_h = ignore_h

        # load and feauturize the complexes
        pos, Z, B, lig_mask = self._load_complexes(complexes, ligand_resname, ignore_h=self.ignore_h)
        self.pos = pos
        self.Z = Z
        self.B = B
        self.lig_mask = lig_mask

    @staticmethod
    def _load_complexes(complexes: Iterable[Path], ligand_resname, ignore_h: bool = True):
        """
        Load the complexes into MDAnalysis"""
        all_pos = []
        all_Z = []
        all_B = []
        all_lig_mask = []
        for complex, lig_resname in zip(complexes, ligand_resname):
            u = mda.Universe(complex)
            # Assuming the first protein is the one of interest
            protein = u.select_atoms("protein")
            # error on empty protein
            if protein.n_atoms == 0:
                raise ValueError(f"No protein found in {complex}")

            protein_pos = protein.positions
            protein_Z = atomic_number_vfunc(protein.atoms.elements)
            protein_B = protein.atoms.bfactors

            ligand = u.select_atoms(f"resname {lig_resname}")
            # error on empty ligand
            if ligand.n_atoms == 0:
                raise ValueError(f"No ligand found in {complex}")

            # error on more than one ligand
            if len(set(ligand.resids)) > 1:
                warnings.warn(f"More than one ligand found in {complex}")

            ligand_pos = ligand.positions
            ligand_Z = atomic_number_vfunc(ligand.atoms.elements)
            ligand_B = ligand.atoms.bfactors
            # concatenate protein and ligand positions
            pos = np.concatenate((protein_pos, ligand_pos), axis=0)
            # concatenate protein and ligand Z
            Z = np.concatenate((protein_Z, ligand_Z), axis=0)
            # concatenate protein and ligand B
            B = np.concatenate((protein_B, ligand_B), axis=0)

            lig_mask = np.zeros(pos.shape[0], dtype=bool)
            lig_mask[protein.n_atoms:] = True

            # cast them to torch tensors
            pos = torch.tensor(pos, dtype=torch.float32)
            Z = torch.tensor(Z, dtype=torch.int32)
            B = torch.tensor(B, dtype=torch.float32)


            # cast the mask to torch tensor
            lig_mask = torch.tensor(lig_mask, dtype=torch.bool)

            # Subset to remove Hs if desired
            if ignore_h:
                h_idx = Z == 1
                pos = pos[~h_idx]
                Z = Z[~h_idx]
                B = B[~h_idx]
                lig_mask = lig_mask[~h_idx]

            all_pos.append(pos)
            all_Z.append(Z)
            all_B.append(B)
            all_lig_mask.append(lig_mask)

        return all_pos, all_Z, all_B, all_lig_mask


    def __len__(self):
        return len(self.complexes)

    def __getitem__(self, idx):
        pos = self.pos[idx]
        Z = self.Z[idx]
        B = self.B[idx]
        lig_mask = self.lig_mask[idx]
        y = self.y[idx]

        return {
            "pos": pos,
            "Z": Z,
            "B": B,
            "lig_mask": lig_mask,
            "Y": y
        }


def _mtenn_collate_fn(batch):
    data_list = []
    targets = []

    for item in batch:
        data = {
                'pos': item['pos'],
                 'z': item['Z'].long(),
                 'lig': item['lig_mask'].bool()
                }

        data_list.append(data)

        targets.append(torch.tensor(item['Y'], dtype=torch.float32))
    targets = torch.stack(targets, dim=0)

    return data_list, targets



@featurizers.register("MTENNFeaturizer")
class MTENNFeaturizer(FeaturizerBase):
    """
    MTENNFeaturizer featurizer for molecules for downstream use in MTENN
    """
    ligand_resname: Union[str, list[str]] = "LIG"
    ignore_h: bool = True
    n_jobs: int = 4
    batch_size: int = 2
    shuffle: bool = False

    _dataset: MTENNDataset = None
    _dataloader: DataLoader = None

    def _prepare(self):
        """
        Prepare the featurizer
        """

    def featurize(self, complexes: Iterable[Path], y: Iterable[Any]) -> DataLoader:
        """
        Featurize a list of SMILES strings
        """
        y = y.to_numpy().reshape(-1, 1)

        self._dataset = MTENNDataset(
            complexes=complexes,
            y=y,
            ligand_resname=self.ligand_resname,
            ignore_h=self.ignore_h,
        )

        self._dataloader = DataLoader(self._dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.n_jobs, collate_fn=_mtenn_collate_fn)
        # return None for Scaler
        return self._dataloader, None
