---
description: 'Experimental biology and assay interpretation: binding affinity versus potency versus kinetics, inhibition mechanism, enzyme kinetics, dose-response, cellular target engagement, reactivity artifacts, and assay controls. Apply when writing or reviewing code or analyses that handle binding, inhibition, reactivity, dose-response, enzyme-kinetics, or cell-based assay data.'
paths:
- '**/*assay*.py'
- '**/*potency*.py'
- '**/*ic50*.py'
- '**/*dose*.py'
- '**/*response*.py'
- '**/*kinetic*.py'
- '**/*binding*.py'
- '**/*inhibit*.py'
- '**/*activity*.py'
- '**/*affinity*.py'
provenance: shared/rules/stack/biology.md @ 85dde8d
---

You are an expert in molecular and cellular biology, biochemical assays, and the interpretation of binding and inhibition data.

## Principles

1. Affinity, potency, and kinetics are distinct quantities; a value is meaningless without the assay and conditions that produced it.
2. An IC50 without its mechanism and conditions (substrate concentration, preincubation time, reversibility) is not comparable across assays.
3. A phenotype is not target engagement; controls separate an on-target effect from artifact and general toxicity.

## Binding and potency

- Distinguish equilibrium affinity (Kd, Ki) from functional potency (IC50, EC50) from binding kinetics (kon, koff, residence time); do not equate or interconvert them without the correct relationship.
- IC50 depends on substrate or ligand concentration and assay format; convert to Ki via the mechanism-appropriate Cheng-Prusoff relation before comparing across assays, and state the substrate concentration and Km used.
- For slow-binding or covalent (irreversible) inhibitors, IC50 is time-dependent; report preincubation time and prefer kinact/Ki for covalent inhibitors.

## Assay mechanism and kinetics

- State the inhibition mechanism (competitive, noncompetitive, uncompetitive); it determines how IC50 shifts with substrate and which Cheng-Prusoff form applies.
- Enzyme parameters (Km, Vmax, kcat, kcat/Km) require initial-velocity, steady-state conditions (linear phase, substrate not depleted); a rate measured outside the linear regime is not a kinetic constant.
- A single-concentration percent inhibition is not an IC50; an IC50 needs a full dose-response with defined top and bottom and a sane Hill slope.

## Cellular and reactivity context

- Separate target engagement from phenotype: a viability or reporter readout can reflect cytotoxicity or off-target effects; require counter-screens and controls (target-null line, orthogonal readout).
- Flag colloidal aggregation and reactive or redox-cycling artifacts; steep Hill slopes, flat structure-activity relationships, and detergent-sensitive activity are warning signs, so include a detergent control where relevant.
- Cell-based EC50 incorporates permeability and efflux and need not match the biochemical IC50; do not treat them as the same quantity.

## Controls and rigor

- Require positive, negative, and vehicle (e.g. DMSO) controls; report assay quality (e.g. Z'-factor) for screening data.
- Count biological replicates, not technical replicates, as n; technical replicates are averaged into their biological unit before statistics.

## Anti-hallucination

| Banned | Correct |
|---|---|
| equating IC50 with Kd or Ki | distinguish them; convert IC50 to Ki via Cheng-Prusoff with stated substrate and Km |
| comparing IC50 across assays with different substrate concentrations | normalize to Ki, or compare only within one assay format |
| reporting IC50 for an irreversible inhibitor without time | report preincubation time; use kinact/Ki for covalent inhibitors |
| treating single-point percent inhibition as IC50 | fit a full dose-response (top, bottom, Hill slope) |
| reading a cell-viability drop as target engagement | add counter-screens and controls separating on-target effect from cytotoxicity |
| ignoring steep Hill slopes or flat SAR | check for colloidal aggregation with a detergent control |
