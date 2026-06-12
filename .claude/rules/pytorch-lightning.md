---
description: 'Lightning 2.x conventions: the unified lightning namespace, LightningModule and DataModule boundaries, Trainer configuration, self.log usage, and removed 1.x hooks and flags. Apply when writing or reviewing Lightning modules, trainers, callbacks, or checkpointing.'
paths:
- '**/*.py'
provenance: shared/rules/stack/pytorch-lightning.md @ 85dde8d
---

You are an expert in Lightning 2.x on top of PyTorch; the pytorch rule governs tensor-level code.

## Principles

1. The LightningModule owns the science (model, loss, metrics); the Trainer owns the engineering (devices, precision, loops); crossing that line forfeits what Lightning buys.
2. Data setup lives in a `LightningDataModule`, keyed to its stage.
3. Reusable behavior is a callback, never a hook hand-edited into every module.

## Directives

- Import from the unified namespace: `import lightning as L`; subclass `L.LightningModule` / `L.LightningDataModule`.
- Configure hardware via `Trainer(accelerator=..., devices=..., precision=...)`; module code never calls `.to(device)`, Lightning places tensors.
- Record metrics with `self.log`/`self.log_dict` (with explicit `on_step`/`on_epoch` where the default is wrong); accumulate epoch-level state in instance attributes reset in `on_*_epoch_end`.
- Return the loss from `training_step`; optimization is automatic unless `automatic_optimization = False` is declared with a reason.
- Resume training via `trainer.fit(model, datamodule=dm, ckpt_path=...)`; checkpoint policy belongs in `ModelCheckpoint`, seeding in `L.seed_everything`.
- `save_hyperparameters()` in `__init__` so checkpoints are self-describing.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `import pytorch_lightning as pl` in new code | `import lightning as L` |
| `Trainer(gpus=4)` / `tpu_cores=` | `Trainer(accelerator="gpu", devices=4)` |
| `training_epoch_end` / `validation_epoch_end` hooks | `on_train_epoch_end` / `on_validation_epoch_end` with manual accumulation |
| `Trainer(resume_from_checkpoint=...)` | `trainer.fit(..., ckpt_path=...)` |
| `trainer.test()` silently reusing fit data | pass the module/datamodule explicitly |
