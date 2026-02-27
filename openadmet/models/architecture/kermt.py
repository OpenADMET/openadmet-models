"""KERMT model implementation via the official KERMT command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
from typing import ClassVar

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, field_validator, model_validator

from openadmet.models.architecture.model_base import PickleableModelBase, models


def _to_1d_array(values) -> np.ndarray:
    """Convert common tabular containers to a flat NumPy array."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(
                "KERMTRegressorModel currently supports exactly one target."
            )
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    values = np.asarray(values)
    if values.ndim > 1:
        values = values.reshape(-1)
    return values


@dataclass
class KERMTCliRegressor:
    """Thin estimator wrapper that delegates training and prediction to KERMT CLI."""

    kermt_repo_path: str
    python_executable: str = "python3"
    checkpoint_path: str | None = None
    checkpoint_blob: bytes | None = None
    epochs: int = 30
    batch_size: int = 32
    metric: str = "mae"
    split_type: str = "random"
    split_sizes: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    no_cuda: bool = True
    no_features_scaling: bool = True
    extra_cli_args: tuple[str, ...] = ()

    def _resolve_repo_path(self) -> Path:
        """Resolve and validate the KERMT repository path."""
        repo_raw = self.kermt_repo_path or os.environ.get("KERMT_REPO_PATH", "")
        if not repo_raw:
            raise ValueError(
                "Path to KERMT repository is required. Set `kermt_repo_path` in the model "
                "or define KERMT_REPO_PATH."
            )
        repo = Path(repo_raw).expanduser().resolve()
        main_py = repo / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(
                f"Could not find KERMT CLI entrypoint at {main_py}. "
                "Expected a valid checkout of https://github.com/NVIDIA-Digital-Bio/KERMT."
            )
        return repo

    def _run_cli(self, args: list[str], repo: Path):
        """Run a KERMT CLI command with an adjusted PYTHONPATH."""
        env = os.environ.copy()
        current_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(repo) if not current_path else f"{repo}{os.pathsep}{current_path}"
        )
        completed = subprocess.run(
            args,
            cwd=repo,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "KERMT CLI command failed.\n"
                f"Command: {' '.join(args)}\n"
                f"Stdout:\n{completed.stdout}\n"
                f"Stderr:\n{completed.stderr}"
            )

    @staticmethod
    def _write_dataset(path: Path, smiles, target=None):
        """Write SMILES and an optional target to CSV for KERMT input."""
        data = {"smiles": smiles}
        if target is not None:
            data["target"] = target
        pd.DataFrame(data).to_csv(path, index=False)

    def _materialize_checkpoint(self, workdir: Path) -> Path:
        """Ensure a checkpoint file exists on disk and return its path."""
        if self.checkpoint_path:
            checkpoint = Path(self.checkpoint_path).expanduser().resolve()
            if checkpoint.exists():
                return checkpoint

        if self.checkpoint_blob is None:
            raise ValueError(
                "No trained KERMT checkpoint found. Train the model first or provide checkpoint_path."
            )

        checkpoint = workdir / "model.pt"
        checkpoint.write_bytes(self.checkpoint_blob)
        self.checkpoint_path = checkpoint.as_posix()
        return checkpoint

    def fit(self, X, y):
        """Train KERMT by invoking `main.py finetune` on a temporary CSV dataset."""
        smiles = _to_1d_array(X)
        targets = _to_1d_array(y).astype(float)
        if len(smiles) != len(targets):
            raise ValueError(
                f"X and y have incompatible lengths ({len(smiles)} != {len(targets)})."
            )

        repo = self._resolve_repo_path()
        run_dir = Path(tempfile.mkdtemp(prefix="kermt_train_"))
        train_csv = run_dir / "train.csv"
        self._write_dataset(train_csv, smiles=smiles, target=targets)

        split_sizes = [f"{size:.6f}" for size in self.split_sizes]
        cmd = [
            self.python_executable,
            "main.py",
            "finetune",
            "--data_path",
            train_csv.as_posix(),
            "--save_dir",
            run_dir.as_posix(),
            "--dataset_type",
            "regression",
            "--split_type",
            self.split_type,
            "--split_sizes",
            *split_sizes,
            "--num_folds",
            "1",
            "--ensemble_size",
            "1",
            "--epochs",
            str(self.epochs),
            "--batch_size",
            str(self.batch_size),
            "--metric",
            self.metric,
            "--seed",
            str(self.seed),
        ]
        if self.no_cuda:
            cmd.append("--no_cuda")
        if self.no_features_scaling:
            cmd.append("--no_features_scaling")
        if self.checkpoint_path:
            cmd.extend(["--checkpoint_path", self.checkpoint_path])
        if self.extra_cli_args:
            cmd.extend(self.extra_cli_args)

        logger.info("Starting KERMT training via CLI")
        self._run_cli(cmd, repo=repo)

        candidate_paths = [
            run_dir / "fold_0" / "model_0" / "model.pt",
            run_dir / "model_0" / "model.pt",
        ]
        checkpoint = next((p for p in candidate_paths if p.exists()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                "KERMT training completed but no checkpoint was found in expected output paths."
            )

        self.checkpoint_path = checkpoint.as_posix()
        self.checkpoint_blob = checkpoint.read_bytes()
        return self

    def predict(self, X) -> np.ndarray:
        """Predict with KERMT by invoking `main.py predict` on a temporary CSV dataset."""
        smiles = _to_1d_array(X)
        repo = self._resolve_repo_path()
        predict_dir = Path(tempfile.mkdtemp(prefix="kermt_predict_"))
        checkpoint = self._materialize_checkpoint(predict_dir)

        input_csv = predict_dir / "input.csv"
        output_csv = predict_dir / "predictions.csv"

        # KERMT expects task columns in the input CSV header for output naming.
        dummy_target = np.zeros(len(smiles), dtype=float)
        self._write_dataset(input_csv, smiles=smiles, target=dummy_target)

        cmd = [
            self.python_executable,
            "main.py",
            "predict",
            "--data_path",
            input_csv.as_posix(),
            "--output_path",
            output_csv.as_posix(),
            "--checkpoint_path",
            checkpoint.as_posix(),
            "--seed",
            str(self.seed),
        ]
        if self.no_cuda:
            cmd.append("--no_cuda")
        if self.no_features_scaling:
            cmd.append("--no_features_scaling")
        if self.extra_cli_args:
            cmd.extend(self.extra_cli_args)

        logger.info("Starting KERMT prediction via CLI")
        self._run_cli(cmd, repo=repo)

        if not output_csv.exists():
            raise FileNotFoundError(
                "KERMT prediction did not create an output CSV file."
            )

        preds = pd.read_csv(output_csv)
        pred_columns = [c for c in preds.columns if not c.lower().startswith("unnamed")]
        if not pred_columns:
            raise ValueError(
                "KERMT prediction output CSV does not contain prediction columns."
            )

        values = preds[pred_columns].to_numpy(dtype=float)
        if values.shape[1] == 1:
            return values[:, 0]
        return values


@models.register("KERMTRegressorModel")
class KERMTRegressorModel(PickleableModelBase):
    """
    KERMT regressor wrapper.

    This model integrates KERMT by delegating training/prediction to the official
    KERMT CLI in an external checkout.
    """

    type: ClassVar[str] = "KERMTRegressorModel"

    kermt_repo_path: str = Field(
        default="",
        description="Path to a local KERMT repository checkout.",
    )
    python_executable: str = "python3"
    checkpoint_path: str | None = None
    epochs: int = 30
    batch_size: int = 32
    metric: str = "mae"
    split_type: str = "random"
    split_sizes: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    no_cuda: bool = True
    no_features_scaling: bool = True
    extra_cli_args: list[str] = Field(default_factory=list)

    @field_validator("split_type")
    @classmethod
    def validate_split_type(cls, value: str) -> str:
        """Validate supported split strategies for CLI delegation."""
        allowed = {"random", "scaffold_balanced"}
        if value not in allowed:
            raise ValueError(f"split_type must be one of {sorted(allowed)}")
        return value

    @field_validator("split_sizes")
    @classmethod
    def validate_split_sizes(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Validate train/val/test split proportions."""
        if len(value) != 3:
            raise ValueError("split_sizes must contain exactly three values.")
        if any(part <= 0 for part in value):
            raise ValueError("split_sizes must contain positive fractions.")
        if not np.isclose(sum(value), 1.0):
            raise ValueError("split_sizes must sum to 1.0.")
        return value

    @model_validator(mode="after")
    def validate_metric(self) -> KERMTRegressorModel:
        """Validate KERMT metric compatibility for regression tasks."""
        allowed = {"rmse", "mae", "r2", "spearmanr"}
        if self.metric not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")
        return self

    def build(self):
        """Prepare the wrapped KERMT estimator."""
        if not self.estimator:
            self.estimator = KERMTCliRegressor(
                kermt_repo_path=self.kermt_repo_path,
                python_executable=self.python_executable,
                checkpoint_path=self.checkpoint_path,
                epochs=self.epochs,
                batch_size=self.batch_size,
                metric=self.metric,
                split_type=self.split_type,
                split_sizes=self.split_sizes,
                seed=self.seed,
                no_cuda=self.no_cuda,
                no_features_scaling=self.no_features_scaling,
                extra_cli_args=tuple(self.extra_cli_args),
            )
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X : array-like
            Raw SMILES strings.
        y : array-like
            Regression targets.

        """
        self.build()
        self.estimator.fit(X, y)

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Predict on input SMILES strings.

        Parameters
        ----------
        X : array-like
            Raw SMILES strings.
        kwargs : dict
            Unused keyword arguments for API compatibility.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        preds = self.estimator.predict(X)
        if preds.ndim == 1:
            return np.expand_dims(preds, axis=1)
        return preds
