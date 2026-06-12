---
description: 'Medicinal chemistry and SAR data conventions: potency units and log transforms, censored assay values, cross-assay comparability, structural alerts and drug-likeness as guidelines, and stereochemistry-aware interpretation. Apply when writing or reviewing code that handles bioactivity data, potency values, SAR tables, or compound property predictions.'
paths:
- '**/featur*.py'
- '**/features/**'
- '**/transform*.py'
- '**/transforms/**'
- '**/*mol*.py'
- '**/*smiles*.py'
- '**/*chem*.py'
- '**/descriptor*.py'
- '**/*compound*.py'
- '**/*potency*.py'
- '**/*assay*.py'
- '**/*sar*.py'
provenance: shared/rules/stack/medicinal-chemistry.md @ 695d847
---

You are an expert in medicinal chemistry, structure-activity relationships, and bioactivity data interpretation.

## Principles

1. Potency lives in log space; arithmetic on raw concentrations is almost always wrong.
2. An activity value is meaningless without its assay context; numbers from different assays are not comparable.
3. Drug-likeness rules are guidelines for triage, never hard pass/fail gates.

## Potency and assay data

- Convert potency to a log scale (pIC50, pKi) before averaging, modeling, or computing differences; never average raw IC50 or Ki values.
- Keep units explicit and consistent (nM throughout, or M throughout); a column mixing nM and µM is a defect.
- Handle censored values (`>`, `<` at assay limits) as censored, never as exact numbers; an inactive reported as `>30 µM` is not 30 µM.
- Do not pool or compare activities measured in different assays, targets, or conditions without an explicit, justified bridge.

## Structure and interpretation

- Treat Lipinski/Veber-style rules and structural alerts (PAINS, toxicophores) as flags for review, not automatic exclusion or inclusion.
- Respect stereochemistry and the relevant tautomer or protonation state; enantiomers and tautomers can differ in activity, so do not collapse them silently.
- Read structure-activity relationships across a matched series; do not generalize potency from a single analog or over-interpret a lone activity cliff.

## Reproducibility

- Record the data source, assay identifier, and any standardization applied to activities; an activity table without provenance cannot be trusted or reproduced.

## Anti-hallucination

| Banned | Correct |
|---|---|
| averaging or interpolating raw IC50 / Ki | average in log space (pIC50, pKi) |
| treating `>10 µM` as exactly 10 µM | handle as censored (right- or left-censored) |
| comparing activities across different assays | compare within a comparable assay, or bridge explicitly |
| mixing nM and µM in one field | one unit throughout, stated |
| Ro5 / PAINS as automatic pass-fail | flag for medicinal-chemistry review |
| collapsing stereoisomers or tautomers | preserve and distinguish them when activity depends on them |
