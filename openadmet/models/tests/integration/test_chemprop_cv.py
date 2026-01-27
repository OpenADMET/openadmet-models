"""Test ChemProp with cross-validation to catch YAML serialization issues."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.eval.cross_validation import (
    PytorchLightningRepeatedKFoldCrossValidation,
)
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.trainer.lightning import LightningTrainer


@pytest.mark.cpu
def test_chemprop_cv_yaml_serialization():
    """
    Test ChemProp with CV to ensure Lightning can save hyperparameters.
    
    This test specifically catches YAML serialization errors that occur
    when PyTorch Lightning tries to save model hyperparameters during
    cross-validation fold training.
    """
    # Create minimal test data (20 samples for fast testing)
    smiles_list = [
        "CCO",
        "CC(C)O",
        "CCCO",
        "CC(C)CO",
        "CCCCO",
        "CC(C)CCO",
        "CCCCCO",
        "CC(C)CCCO",
        "CCCCCCO",
        "CC(C)CCCCO",
        "CCCCCCCO",
        "CC(C)CCCCCO",
        "CCCCCCCCO",
        "CC(C)CCCCCCO",
        "CCCCCCCCCO",
        "CC(C)CCCCCCCO",
        "CCCCCCCCCCO",
        "CC(C)CCCCCCCCO",
        "CCCCCCCCCCCO",
        "CC(C)CCCCCCCCCO",
    ]
    target_values = [i * 0.1 for i in range(len(smiles_list))]
    
    df = pd.DataFrame({
        "smiles": smiles_list,
        "target": target_values,
    })
    
    # Create featurizer (CV evaluator will call it internally)
    featurizer = ChemPropFeaturizer()

    # CV evaluator expects DataFrames, not DataLoaders
    X = df[["smiles"]]
    y = df[["target"]]
    
    # Create model
    model = ChemPropModel(
        n_tasks=1,
        messages="bond",
        aggregation="norm",
        depth=2,  # Minimal depth for speed
        message_hidden_dim=100,  # Smaller for speed
        ffn_hidden_dim=100,  # Smaller for speed
        ffn_num_layers=1,
        dropout=0.0,
        warmup_epochs=0,  # No warmup for speed
    )
    model.build()
    
    # Create trainer
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LightningTrainer(
            max_epochs=1,  # Just 1 epoch for speed
            accelerator="cpu",
            devices=1,
            use_wandb=False,
            output_dir=Path(tmpdir),
        )
        trainer.model = model
        trainer.build()
        
        # Create CV evaluator (minimal folds/repeats for speed)
        cv_evaluator = PytorchLightningRepeatedKFoldCrossValidation(
            n_splits=2,  # Minimal splits
            n_repeats=1,  # Minimal repeats
            random_state=42,
        )
        
        # This should NOT raise a YAML serialization error
        # The bug occurs when Lightning tries to save hyperparameters
        # during fold training
        cv_evaluator.evaluate(
            model=model,
            X_train=X,
            y_train=y,
            X_all=X,
            y_all=y,
            featurizer=featurizer,
            trainer=trainer,
            tag="test_cv",
        )
        
        # If we get here, the test passed - no YAML serialization error
        assert cv_evaluator.data is not None
        assert len(cv_evaluator.data) > 0


if __name__ == "__main__":
    test_chemprop_cv_yaml_serialization()
    print("Test passed!")
