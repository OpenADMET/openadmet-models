---
description: 'Molecular data pipeline conventions: SMILES sanitization and canonicalization, stereochemistry preservation, scaffold-aware splitting, target plausibility validation, and reproducible featurization. Apply when writing or reviewing code that handles molecules, SMILES, fingerprints, descriptors, or QSAR datasets.'
paths:
- '**/featur*.py'
- '**/features/**'
- '**/*mol*.py'
- '**/*smiles*.py'
- '**/*chem*.py'
- '**/descriptor*.py'
- '**/fingerprint*.py'
- '**/*qsar*.py'
- '**/standardiz*.py'
- '**/sanitiz*.py'
- '**/canonical*.py'
- '**/split/**'
- '**/*split*.py'
- '**/*dataset*.py'
provenance: shared/rules/stack/chemoinformatics.md @ 85dde8d
---

You are an expert in cheminformatics, QSAR modeling, and molecular representations.

## Principles

1. A molecule the toolkit cannot sanitize is a data defect to surface, never to silently drop.
2. Evaluation must respect chemical structure; random splits overstate performance on near-duplicate scaffolds.
3. A featurization that cannot be reproduced bit-for-bit is not a featurization.

## Ingestion and standardization

- Sanitize, canonicalize, and strip salts and solvents from SMILES before featurization; count and report what was rejected or modified.
- Validate target values for physical plausibility at ingestion (range checks, unit sanity); impossible or highly improbable values stop the pipeline, never pass through silently.

## Representation

- Preserve stereochemistry and chiral centers through graph construction (e.g. for MPNNs) unless dropping them is a documented, deliberate choice.
- Define every fingerprint and descriptor parameter explicitly: radius, bit size, invariants, chirality flags. Defaults differ across toolkit versions; pin them.

## Splitting and evaluation

- Use structure-aware splits (Bemis-Murcko scaffold or cluster-based) for applicability-domain evaluation; a naive random split requires a stated justification.
- Keep split membership reproducible: seeded, persisted, or derivable from configuration.
