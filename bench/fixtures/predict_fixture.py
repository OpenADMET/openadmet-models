"""
Build and check a tiny trained model directory for the predict_smoke phase.

The script runs inside the candidate environment, so it needs only packages in
the inference group (PyYAML, NumPy, and openadmet.models). It mirrors the
``fingerprint_model_dir`` fixture in ``openadmet/models/tests/unit/conftest.py``:
a dummy regressor that always predicts 5.0 behind an ECFP featurizer, which
exercises RDKit, molfeat, and the inference loader without training anything.

Usage::

    python predict_fixture.py build ROOT   # writes ROOT/model and ROOT/input.csv
    python predict_fixture.py check CSV    # validates the predictions CSV
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

TAG = "FP"
TARGET = "task_0"
CONSTANT = 5.0
SMILES = ["CCO", "c1ccccc1", "CC(=O)OC"]
PRED_COLUMN = f"OADMET_PRED_{TAG}_{TARGET}"


def build(root: Path) -> None:
    """
    Write the model directory and the input CSV.

    Parameters
    ----------
    root : pathlib.Path
        Output directory; ``model`` and ``input.csv`` are created inside it.

    """
    import numpy as np
    import yaml

    from openadmet.models.architecture.dummy import DummyRegressorModel

    model_dir = root / "model"
    recipe_dir = model_dir / "recipe_components"
    recipe_dir.mkdir(parents=True, exist_ok=True)

    # Write the three recipe files the inference loader requires
    metadata = {
        "version": "v1",
        "driver": "sklearn",
        "name": "bench-predict-smoke",
        "build_number": 0,
        "description": "Predict smoke fixture",
        "tag": TAG,
        "authors": "bench",
        "email": "bench@example.com",
        "biotargets": ["test"],
        "tags": ["test"],
    }
    data_spec = {"type": "csv", "input_col": "MY_SMILES", "target_cols": [TARGET]}
    procedure = {
        "feat": {
            "type": "FingerprintFeaturizer",
            "params": {"fp_type": "ecfp", "n_jobs": 1},
        },
        "model": {"type": "DummyRegressorModel", "params": {}},
        "split": {
            "type": "ShuffleSplitter",
            "params": {"train_size": 0.8, "test_size": 0.2, "random_seed": 42},
        },
        "train": {"type": "SKLearnBasicTrainer", "params": {}},
    }
    for label, content in (
        ("metadata", metadata),
        ("data", data_spec),
        ("procedure", procedure),
    ):
        with (recipe_dir / f"{label}.yaml").open("w") as handle:
            yaml.safe_dump(content, handle)

    # Train and serialize the constant model
    model = DummyRegressorModel()
    model.train(np.zeros((3, 1)), np.full(3, CONSTANT))
    model.serialize(model_dir / "model.json", model_dir / "model.pkl")

    # Write the prediction input
    (root / "input.csv").write_text("MY_SMILES\n" + "\n".join(SMILES) + "\n")


def check(csv_path: Path) -> None:
    """
    Validate the predictions written by ``openadmet predict``.

    Parameters
    ----------
    csv_path : pathlib.Path
        Predictions CSV.

    Raises
    ------
    SystemExit
        With a message when the file is missing, has no prediction column, has
        the wrong row count, or contains a value other than the constant.

    """
    if not csv_path.exists():
        raise SystemExit(f"predictions file not found: {csv_path}")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or PRED_COLUMN not in rows[0]:
        columns = list(rows[0]) if rows else []
        raise SystemExit(f"column {PRED_COLUMN} missing; columns: {columns}")
    if len(rows) != len(SMILES):
        raise SystemExit(f"expected {len(SMILES)} rows, found {len(rows)}")
    values = [float(row[PRED_COLUMN]) for row in rows]
    if any(abs(v - CONSTANT) > 1e-6 for v in values):
        raise SystemExit(f"expected constant {CONSTANT}, found {values}")
    print(f"predict smoke ok: {len(rows)} rows, constant {CONSTANT}")


def main(argv: list[str]) -> int:
    """Dispatch the build and check subcommands."""
    if len(argv) != 2 or argv[0] not in ("build", "check"):
        print(__doc__)
        return 2
    action, target = argv[0], Path(argv[1])
    if action == "build":
        build(target)
    else:
        check(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
