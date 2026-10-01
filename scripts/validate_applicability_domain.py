"""Exploratory validation of the scaffold and Tanimoto applicability domains.

Trains a small model on AChE ChEMBL activity data, then checks whether
compounds the AD flags as out-of-domain actually carry larger errors than
in-domain compounds, under two regimes:

- random 80/20 split: test compounds share the scaffold pool; AD flags the
  rare-scaffold and low-similarity tail
- scaffold holdout: entire test scaffolds are disjoint from train

RandomForest on Morgan fingerprints; 5-fold out-of-fold errors fit the
domain bounds. Runs in about a minute.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.DataStructs import ConvertToNumpyArray
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from openadmet.models.applicability_domain.scaffold import (
    ScaffoldApplicabilityDomain,
)
from openadmet.models.applicability_domain.similarity import (
    TanimotoApplicabilityDomain,
)

DATA = "openadmet/models/tests/unit/test_data/AChE_CHEMBL4078_Landrum_maxcur.csv"


def morgan_fps(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keep = [], []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        keep.append(i)
    X = np.zeros((len(fps), 2048))
    for i, fp in enumerate(fps):
        ConvertToNumpyArray(fp, X[i])
    return X, np.asarray(keep)


def scaffold_groups(smiles_list):
    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        out.append(MurckoScaffoldSmiles(mol=mol) if mol is not None else s)
    return np.array(out)


def summarize(name, in_mask, test_err, bound, ad):
    n_in, n_out = int(in_mask.sum()), int((~in_mask).sum())
    print(f"\n== {name} ==")
    print(f"  in-domain n={n_in}  out-of-domain n={n_out}")
    cov = np.mean(test_err <= bound)
    print(f"  coverage (test_err <= bound) : {cov:.3f} "
          f"(target {ad.error_percentile / 100:.2f})")
    if n_out == 0 or n_in == 0:
        print("  degenerate split: no discrimination possible")
        return in_mask
    in_err, out_err = test_err[in_mask], test_err[~in_mask]
    top = test_err >= np.quantile(test_err, 0.9)
    print(f"  median |err|  in={np.median(in_err):.3f}  out={np.median(out_err):.3f}")
    print(f"  mean   |err|  in={in_err.mean():.3f}  out={out_err.mean():.3f}")
    print(f"  worst-decile flagged OOD: {(top & ~in_mask).sum()}/{top.sum()}")
    cov = np.mean(test_err <= bound)
    print(f"  coverage (test_err <= bound) : {cov:.3f} "
          f"(target {ad.error_percentile / 100:.2f})")
    if len(np.unique(bound)) > 1:
        rho, p = spearmanr(bound, test_err)
        print(f"  spearman(bound, |err|): {rho:.3f} (p={p:.1e})")
    else:
        print("  bound is constant on this split")
    rng = np.random.default_rng(0)
    null_gaps = []
    for _ in range(200):
        m = rng.random(len(test_err)) < (in_mask.mean())
        if m.any() and (~m).any():
            null_gaps.append(test_err[~m].mean() - test_err[m].mean())
    gap = out_err.mean() - in_err.mean()
    print(f"  in/out gap {gap:+.3f} vs random-flag gap "
          f"{np.mean(null_gaps):+.3f} +- {np.std(null_gaps):.3f}")
    return in_mask


def run_split(name, smiles, y, X, groups, test_mask):
    tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
    print(f"\n########## {name}: train={len(tr)} test={len(te)} "
          f"shared-scaffold fraction of test="
          f"{np.isin(groups[te], groups[tr]).mean():.3f}")

    oof = np.full(len(tr), np.nan)
    for f_tr, f_va in GroupKFold(5).split(X[tr], y[tr], groups[tr]):
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
        m.fit(X[tr][f_tr], y[tr][f_tr])
        oof[f_va] = np.abs(m.predict(X[tr][f_va]) - y[tr][f_va])

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    model.fit(X[tr], y[tr])
    test_err = np.abs(model.predict(X[te]) - y[te])
    print(f"train OOF MAE={oof.mean():.3f}  test MAE={test_err.mean():.3f}")

    for label, ad in [
        ("ScaffoldAD", ScaffoldApplicabilityDomain(5, 95.0)),
        ("TanimotoAD", TanimotoApplicabilityDomain(0.4, 95.0)),
        ("TanimotoAD@0.5", TanimotoApplicabilityDomain(0.5, 95.0)),
    ]:
        ad.fit(smiles[tr], oof)
        in_mask = ad.is_in_domain(smiles[te])
        summarize(label, in_mask, test_err, ad.bound(smiles[te]), ad)


def main():
    df = pd.read_csv(DATA).dropna(subset=["canonical_smiles", "pchembl_value"])
    smiles = df["canonical_smiles"].to_numpy()
    y = df["pchembl_value"].to_numpy(dtype=float)
    X, keep = morgan_fps(smiles)
    smiles, y, groups = smiles[keep], y[keep], scaffold_groups(smiles[keep])
    X = X  # row order already aligned to keep

    rng = np.random.default_rng(0)
    run_split("random 80/20", smiles, y, X, groups,
              rng.random(len(smiles)) < 0.2)

    counts = pd.Series(groups).value_counts().to_numpy()
    rare = set(pd.Series(groups).value_counts()[counts <= np.quantile(counts, 0.5)].index)
    mask = np.isin(groups, list(rare))
    # trim toward ~20% by dropping the largest rare-scaffold groups first
    if mask.mean() > 0.30:
        order = (pd.Series(groups)[mask]
                 .map(pd.Series(groups).value_counts())
                 .sort_values(ascending=False).index.to_numpy())
        drop = set(order[: int(len(order) - 0.2 * len(smiles))])
        mask = np.isin(np.arange(len(smiles)), list(drop)) | (mask & ~np.isin(np.arange(len(smiles)), list(drop)))
        mask = np.isin(np.arange(len(smiles)),
                       np.where(mask)[0]) & ~np.isin(np.arange(len(smiles)), list(drop))
    run_split("scaffold holdout", smiles, y, X, groups, mask)


if __name__ == "__main__":
    main()
