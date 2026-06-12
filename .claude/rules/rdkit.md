---
description: 'RDKit conventions: None-checked molecule parsing, fingerprint generators, canonical SMILES discipline, standardization, and deprecated fingerprint APIs. Apply when writing or reviewing cheminformatics code handling molecules, SMILES, descriptors, or fingerprints.'
paths:
- '**/*.py'
provenance: shared/rules/stack/rdkit.md @ 85dde8d
---

You are an expert in cheminformatics with RDKit; domain interpretation (potency, SAR, assay data) is governed by the medicinal-chemistry rule.

## Principles

1. Every molecule from external input can be `None` or unsanitizable; unchecked parsing is the canonical RDKit crash.
2. Identity is the canonical form: compare molecules by canonical SMILES or InChIKey, never raw input strings.
3. Standardize before featurizing; descriptors on unstandardized structures are noise.

## Directives

- Check `Chem.MolFromSmiles`/`MolFromMolBlock` results for `None` before any use; in batch pipelines, count and log parse failures rather than silently dropping rows.
- Standardize input structures with `rdMolStandardize` (cleanup, largest fragment, uncharge as the project requires) before computing descriptors or fingerprints, and record which steps ran.
- Build fingerprints through `rdFingerprintGenerator` (e.g. `GetMorganGenerator(radius=2, fpSize=2048)`) with radius and size stated explicitly; the module-level Morgan/AP functions are legacy.
- Edit molecules on a `Chem.RWMol`, then sanitize and convert back; never mutate a parsed `Mol` mid-iteration.
- Suppress RDKit's per-molecule logging in batch work with `RDLogger.DisableLog("rdApp.*")` only after parse-failure accounting is in place.
- Compute descriptors from `rdkit.Chem.Descriptors`/`rdMolDescriptors` by name, keeping the descriptor list versioned with the model that consumes it.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `AllChem.GetMorganFingerprintAsBitVect(mol, 2)` | `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)` |
| using a `Mol` straight from parsing without a `None` check | guard, count, and log failures |
| `Chem.MolToSmiles(mol, canonical=False)` for identity | canonical SMILES (default) or InChIKey |
| `import rdkit.Chem as Chem` then `pybel`-style attribute access | RDKit's own API; openbabel idioms do not transfer |
