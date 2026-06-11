---
description: 'Machine learning methodology across the lifecycle: data leakage, training dynamics, evaluation discipline, train/serve consistency, and reproducibility. Apply when writing or reviewing data splits, training loops, loss functions, model evaluation, or inference and serving code.'
paths:
- '**/train*.py'
- '**/trainer/**'
- '**/*trainer*.py'
- '**/split/**'
- '**/*split*.py'
- '**/eval/**'
- '**/*eval*.py'
- '**/*metric*.py'
- '**/loss*.py'
- '**/features/**'
- '**/featur*.py'
- '**/architecture/**'
- '**/inference/**'
- '**/*inference*.py'
- '**/dataset*.py'
- '**/dataloader*.py'
- '**/active_learning/**'
provenance: shared/rules/stack/machine-learning.md @ 89ac50f
---

You are an expert in machine learning methodology across the lifecycle: data handling, training dynamics, evaluation, and deployment.

## Principles

1. A result computed on leaked data is not a result.
2. Evaluation must be able to detect the failure you care about; aggregate loss hides minority-class collapse.
3. Training and serving must share one preprocessing path; a divergence between them silently degrades a model that tested well.
4. A run that cannot be reproduced from configuration does not exist.

## Data discipline

- Split before any data-dependent transform: scalers, normalizers, and encoders are fit on the training split only, then applied to validation and test. Fitting on the full dataset is target leakage.
- Splitting logic is explicit and seeded; the split definition lives in configuration, never in inline constants.

## Training loop

- Zero gradients before each backward pass (`optimizer.zero_grad(set_to_none=True)`); never accumulate silently.
- Detach tensors when computing metrics or custom loss diagnostics (e.g. MVE, MSE breakdowns) outside the optimization step; metric computation must not extend the autograd graph.
- Log mean and variance (or the chosen uncertainty measure) for every validation batch so epoch-level performance and uncertainty are trackable.

## Evaluation

- Hold the test set out until the end; it is touched once, after model selection, never used for tuning.
- Choose metrics that match the decision the model serves (precision/recall, AUPRC, or calibration, not accuracy on imbalanced data); report per-class and per-cluster metrics alongside global aggregates.
- Compare against a baseline (a trivial predictor or prior model); an absolute metric without a baseline is uninterpretable.
- Calibrate and check probabilistic outputs; a model's confidence is a claim to verify, not to assume.
- Evaluation paths run under `torch.no_grad()` (or `inference_mode`) with `model.eval()`.

## Inference and serving

- Apply the exact preprocessing used in training at inference; reuse one code path rather than reimplementing it, so train/serve skew cannot creep in.
- Version the model artifact together with its preprocessing and the code that produced it; a checkpoint without its pipeline is not deployable.
- Validate input schema and distribution at the serving boundary; flag out-of-distribution inputs rather than scoring them silently.

## Configuration and reproducibility

- Hyperparameters (learning rate, dropout, early-stopping patience, batch size) live in a single configuration object surfaced at the entry point, never hardcoded inside execution blocks.
- Seed every RNG (framework, NumPy, Python) and prefer deterministic ops where feasible; an unseeded run is not reproducible.
- Version data, code, configuration, and environment together so a result can be regenerated; track experiments rather than relying on memory.

## Anti-hallucination

| Banned | Correct |
|---|---|
| using the test set for tuning or model selection | hold it out; touch once, after selection |
| reporting accuracy on imbalanced data | a metric matching the decision (precision/recall, AUPRC, calibration) |
| divergent preprocessing in training vs inference | one shared preprocessing path (no train/serve skew) |
| an absolute metric with no baseline | compare against a trivial or prior baseline |
| unseeded runs or nondeterministic evaluation | seed all RNGs; deterministic ops where feasible |
