"""Workflow implementations for Anvil models."""

import hashlib
import uuid
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from lightning import pytorch as pl
from loguru import logger
from pydantic import model_validator

from openadmet.models.anvil.workflow_base import AnvilWorkflowBase
from openadmet.models.drivers import DriverType
from openadmet.models.features.pairwise import PairwiseFeaturizer


def _safe_to_numpy(X):
    if isinstance(X, (pd.Series, pd.DataFrame)):
        return X.to_numpy()
    return X


class AnvilWorkflow(AnvilWorkflowBase):
    """Workflow for running basic Anvil configuration."""

    _driver_type: DriverType = DriverType.SKLEARN

    @model_validator(mode="after")
    def check_if_val_needed(self):
        """
        Check if validation set is needed or not.

        Raises
        ------
        ValueError
            If ensemble is specified but no validation set is requested.
        ValueError
            If validation set is requested but no ensemble is specified.

        """
        # Ensemble models require a validation set for uncertainty calibration
        if self.ensemble and self.split.val_size == 0:
            raise ValueError(
                "Ensemble models require a validation set for uncertainty calibration."
            )

        # Non-ensemble models do not use a validation set
        elif not self.ensemble and self.split.val_size != 0:
            raise ValueError(
                "Validation set requested, but not used in this workflow configuration."
            )

        return self

    @model_validator(mode="after")
    def check_no_finetuning(self):
        """
        Check that no fine-tuning paths are specified.

        Raises
        ------
        ValueError
            If fine-tuning paths are specified for either ensemble or single model.

        """
        # Ensemble specified
        if self.ensemble:
            # Fine-tuning paths specified
            if (self.ensemble_kwargs.get("param_paths") is not None) or (
                self.ensemble_kwargs.get("serial_paths") is not None
            ):
                raise ValueError(
                    "Finetuning from serialized ensemble models is not supported in this workflow."
                )

        # No ensemble
        else:
            # Fine-tuning paths supplied
            if (self.model_kwargs.get("param_path") is not None) or (
                self.model_kwargs.get("serial_path") is not None
            ):
                raise ValueError(
                    "Finetuning from serialized model is not supported in this workflow."
                )

        # All fine-tuning paths are None
        return self

    def _train(self, X_train_feat, y_train, output_dir):
        X_train_feat = _safe_to_numpy(X_train_feat)
        y_train = _safe_to_numpy(y_train)

        # Build model from scratch
        logger.info("Building model")
        self.model.build()
        logger.info("Model built")

        # Pass model to trainer
        logger.info("Setting model in trainer")
        self.trainer.model = self.model
        logger.info("Model set in trainer")

        # Commence model training
        logger.info("Training model")
        self.model = self.trainer.train(X_train_feat, y_train)
        logger.info("Model trained")

    def _train_ensemble(self, X_train_feat, y_train, output_dir, **kwargs):
        X_train_feat = _safe_to_numpy(X_train_feat)
        y_train = _safe_to_numpy(y_train)

        # Get bagging setting
        use_bagging = self.ensemble_kwargs.get("use_bagging")

        # Get global seed
        # Currently grabbing from `split`, should this be set separately?
        global_seed = self.split.random_state

        # Bootstrap iterations
        models = []
        for i in range(self.ensemble_kwargs["n_models"]):
            # Manage bootstrap directory
            bootstrap_dir = output_dir / f"bootstrap_{i}"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)

            # Bootstrap data if using bagging, if not specified default False
            if use_bagging:
                # Set seed for bootstrapping
                logger.info(
                    f"Using incremented seed={global_seed + i} for bootstrapping"
                )
                np.random.seed(global_seed + i)

                # Bootstrap train data
                logger.info("Bootstrapping train data")
                bootstrap_indices = np.random.choice(
                    np.arange(len(X_train_feat)), size=len(X_train_feat), replace=True
                )
                X_train_feat_bootstrap = X_train_feat[bootstrap_indices]
                y_train_bootstrap = y_train[bootstrap_indices]
                logger.info("Data bootstrapped")
            else:
                X_train_feat_bootstrap = X_train_feat
                y_train_bootstrap = y_train

            # Build model from scratch
            logger.info(
                f"Building model {i} using incremented seed={global_seed + i} to vary model initialization"
            )
            bootstrap_model = self.model.make_new()

            # Set seed for model
            if hasattr(bootstrap_model, "random_state"):
                bootstrap_model.random_state = global_seed + i
            else:
                logger.warning(
                    f"Model {bootstrap_model} does not support random_state seeding"
                )

            bootstrap_model.build()
            logger.info(f"Model {i} built")

            # Pass model to trainer
            logger.info(f"Setting model {i} in trainer")
            self.trainer.model = bootstrap_model
            logger.info(f"Model {i} set in trainer")

            # Train model on bootstrapped data
            logger.info(f"Training model {i}")
            bootstrap_model = self.trainer.train(
                X_train_feat_bootstrap, y_train_bootstrap
            )
            logger.info(f"Model {i} trained")

            # Add model to list
            models.append(bootstrap_model)

        # Create ensemble from trained models
        self.model = self.ensemble.from_models(models)

    def run(
        self,
        output_dir: PathLike = "anvil_training",
        debug: bool = False,
        tag: str = None,
    ) -> Any:
        """
        Run the workflow.

        Parameters
        ----------
        output_dir : PathLike, optional
            Directory to save outputs, by default "anvil_training"
        debug : bool, optional
            Whether to run in debug mode, by default False
        tag : str, optional
            Tag to override the one in the recipe, by default None

        Returns
        -------
        Any
            Result of the workflow run

        """
        # Override the model tag from yaml if provided in cli
        if tag is not None:
            model_tag = tag
        else:
            model_tag = self.metadata.tag

        target_labels = self.data_spec.target_cols

        # Set debug attribute
        self.debug = debug

        # Cast output directory to string, stripping any trailing separator
        output_dir = str(Path(output_dir))

        # Output directory already exists, create new handle
        if Path(output_dir).exists():
            # Make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:6]

            # Get the date and time in short format
            now = datetime.now().strftime("%Y-%m-%d")

            # Extended output directory
            output_dir = Path(output_dir + f"_{now}_{hsh}")

        # Output directory does not exist, handle as is
        else:
            output_dir = Path(output_dir)

        # Create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_output_dir = output_dir

        # Create data subdirectory
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Log output directory information
        logger.info(f"Running workflow from directory {output_dir}")

        # Log workflow driver selection
        logger.info(f"Running with driver {self._driver_type}")

        # Load data from YAML specification
        logger.info("Loading data")
        if self.data_spec.using_train_test:
            logger.info(
                "Using prespecified train/test resources from data specification"
            )
            X_train, X_val, X_test, y_train, y_val, y_test, X, y = self.data_spec.read()
            groups = None
        else:
            X, y = self.data_spec.read()
            # Split data into train, validation, and test sets
            logger.info("Splitting data from single resource")
            X_train, X_val, X_test, y_train, y_val, y_test, groups = self.split.split(
                X, y
            )
        logger.info("Data loaded")

        # Save splits to CSV outputs
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)

        # Save val if present
        if X_val is not None:
            X_val.to_csv(data_dir / "X_val.csv", index=False)
            y_val.to_csv(data_dir / "y_val.csv", index=False)

        # Test
        if X_test is not None:
            X_test.to_csv(data_dir / "X_test.csv", index=False)
            y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        # Featurize splits
        logger.info("Featurizing data")
        # Train
        X_train_feat, _ = self.feat.featurize(X_train)
        zarr.save(data_dir / "X_train_feat.zarr", X_train_feat)

        # Val
        if X_val is not None:
            X_val_feat, _ = self.feat.featurize(X_val)
            zarr.save(data_dir / "X_val_feat.zarr", X_val_feat)

        # Test
        if X_test is not None:
            X_test_feat, _ = self.feat.featurize(X_test)
            zarr.save(data_dir / "X_test_feat.zarr", X_test_feat)

        # featurize whole dataset also for CV if needed
        X_feat, _ = self.feat.featurize(X)

        # Transform data
        if self.transform:
            # Train
            logger.info("Transforming data")
            self.transform.fit(X_train_feat)
            X_train_feat = self.transform.transform(X_train_feat)
            zarr.save(data_dir / "X_train_feat_transformed.zarr", X_train_feat)

            # Val
            if X_val is not None:
                X_val_feat = self.transform.transform(X_val_feat)
                zarr.save(data_dir / "X_val_feat_transformed.zarr", X_val_feat)

            # Test
            if X_test is not None:
                X_test_feat = self.transform.transform(X_test_feat)
                zarr.save(data_dir / "X_test_feat_transformed.zarr", X_test_feat)

            # Whole dataset
            X_feat = self.transform.transform(X_feat)

            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        logger.info("Data featurized")

        # Train the model
        if self.ensemble:
            # Ensemble mode
            self._train_ensemble(X_train_feat, y_train, output_dir)

            # Calibrate
            self.model.calibrate_uncertainty(
                X_val_feat,
                y_val,
                method=self.ensemble_kwargs.get("calibration_method"),
            )

            # Save
            logger.info("Saving model")
            self.model.serialize(
                [
                    output_dir
                    / f"bootstrap_{i}"
                    / self.model.models[i]._model_json_name
                    for i in range(self.model.n_models)
                ],
                [
                    output_dir
                    / f"bootstrap_{i}"
                    / self.model.models[i]._model_save_name
                    for i in range(self.model.n_models)
                ],
                output_dir / self.model._calibration_model_save_name,
            )
            logger.info("Model saved")
        else:
            # Single-model mode
            self._train(X_train_feat, y_train, output_dir)

            # Save
            logger.info("Saving model")
            self.model.serialize(
                output_dir / self.model._model_json_name,
                output_dir / self.model._model_save_name,
            )
            logger.info("Model saved")

        if X_test is not None:
            # Predict on test set
            logger.info("Predicting")
            # Check if the model has predict_proba method (classification)
            if hasattr(self.model, "predict_proba"):
                y_pred = self.model.predict_proba(X_test_feat)
                y_std = None

            # Otherwise, regression
            else:
                if self.ensemble:
                    y_pred, y_std = self.model.predict(X_test_feat, return_std=True)
                else:
                    y_pred = self.model.predict(X_test_feat)
                    y_std = None
            logger.info("Predictions made")
        else:
            y_pred = None
            y_std = None
            logger.info("No test set specified, predictions skipped")

        # Run evaluation on train/test
        logger.info("Evaluating")
        for eval in self.evals:
            # Here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                y_std=y_std,
                model=self.model,
                X_train=X_train_feat,
                y_train=y_train,
                X_all=X_feat,
                groups=groups,
                y_all=y,
                tag=model_tag,
                target_labels=target_labels,
                output_dir=output_dir
            )

            # Write evaluation report
            eval.report(write=True, output_dir=output_dir)

        logger.info("Evaluation done")


class AnvilDeepLearningWorkflow(AnvilWorkflowBase):
    """Workflow for running deep learning Anvil configuration."""

    _driver_type: DriverType = DriverType.LIGHTNING

    @model_validator(mode="after")
    def check_no_transform(self):
        """
        Check that no transform step is specified.

        Raises
        ------
        ValueError
            If a transform step is specified in the recipe.

        """
        # Check that transform is not set
        if self.transform is not None:
            raise ValueError(
                "Transform step is not supported in this workflow. Please remove it from the recipe."
            )
        return self

    @model_validator(mode="after")
    def check_if_val_needed(self):
        """
        Check if validation set is needed or not.

        Raises
        ------
        ValueError
            If ensemble is specified but no validation set is requested.

        """
        # Ensemble models require a validation set for uncertainty calibration
        if self.ensemble and self.split.val_size == 0:
            raise ValueError(
                "Ensemble models require a validation set for uncertainty calibration."
            )

        return self

    @model_validator(mode="after")
    def check_finetuning_paths(self):
        """
        Check that finetuning path pairs are consistent and exist on disk.

        Both ``param_path`` and ``serial_path`` must be provided together (or
        neither). When both are provided, both paths must exist before training
        begins. The same requirement applies to ``param_paths`` / ``serial_paths``
        for ensemble workflows, which must additionally be equal-length lists.

        Raises
        ------
        ValueError
            If exactly one of the path pair is provided, if provided paths do
            not exist on disk, or if ensemble path lists have unequal length.

        """
        if not self.ensemble:
            param_path = self.model_kwargs.get("param_path")
            serial_path = self.model_kwargs.get("serial_path")
            if (param_path is None) != (serial_path is None):
                raise ValueError(
                    "Both param_path and serial_path must be provided together for finetuning."
                )
            if param_path is not None:
                if not Path(param_path).exists():
                    raise ValueError(f"param_path '{param_path}' does not exist.")
                if not Path(serial_path).exists():
                    raise ValueError(f"serial_path '{serial_path}' does not exist.")
        else:
            param_paths = self.ensemble_kwargs.get("param_paths")
            serial_paths = self.ensemble_kwargs.get("serial_paths")
            if (param_paths is None) != (serial_paths is None):
                raise ValueError(
                    "Both param_paths and serial_paths must be provided together for ensemble finetuning."
                )
            if param_paths is not None:
                if len(param_paths) != len(serial_paths):
                    raise ValueError(
                        "param_paths and serial_paths must have equal length."
                    )
                for p in param_paths:
                    if not Path(p).exists():
                        raise ValueError(f"param_path '{p}' does not exist.")
                for s in serial_paths:
                    if not Path(s).exists():
                        raise ValueError(f"serial_path '{s}' does not exist.")
        return self

    def _train(
        self, train_dataloader, val_dataloader, train_scaler, output_dir, **kwargs
    ):
        # Load model from disk
        if (
            self.model_kwargs.get("param_path") is not None
            and self.model_kwargs.get("serial_path") is not None
        ):
            logger.info("Loading model from disk, overrides any specified parameters.")
            self.model = self.model.deserialize(
                self.model_kwargs.get("param_path"),
                self.model_kwargs.get("serial_path"),
                scaler=train_scaler,
                **kwargs,
            )

            logger.info("Model loaded")

            # Optionally freeze weights
            if self.model_kwargs.get("freeze_weights") is not None:
                logger.info(f"Freezing model weights")
                self.model.freeze_weights(**self.model_kwargs.get("freeze_weights"))
                logger.info(f"Model weights frozen")

        # Build model from scratch
        else:
            logger.info(f"Building model")
            self.model.build(scaler=train_scaler, **kwargs)
            logger.info("Model built")

        # Pass model to trainer
        logger.info("Setting model in trainer")
        self.trainer.model = self.model
        logger.info("Model set in trainer")

        # Check if there is an output directory
        if not self.trainer.output_dir:
            self.trainer.output_dir = output_dir

        # Prepare the trainer
        logger.info("Preparing trainer")
        self.trainer.build(no_val=(val_dataloader is None))
        logger.info("Trainer prepared")

        # Commence model training
        logger.info("Training model")
        self.model = self.trainer.train(train_dataloader, val_dataloader)
        logger.info("Model trained")

    def _train_ensemble(self, X_train, y_train, val_dataloader, output_dir, **kwargs):
        # Safely cast to numpy
        X_train = _safe_to_numpy(X_train)
        y_train = _safe_to_numpy(y_train)

        # Check if there is an output directory
        if not self.trainer.output_dir:
            self.trainer.output_dir = output_dir

        # Get bagging setting
        use_bagging = self.ensemble_kwargs.get("use_bagging")

        # Get global seed
        # Currently grabbing from `split`, should this be set separately?
        global_seed = self.split.random_state

        # Bootstrap iterations
        models = []

        for i in range(self.ensemble_kwargs["n_models"]):
            # Manage bootstrap directory
            bootstrap_dir = output_dir / f"bootstrap_{i}"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)

            # Make new instances
            self.feat = self.feat.make_new()
            self.trainer = self.trainer.make_new()

            # Bootstrap data if using bagging, if not specified default False
            if use_bagging:
                # Set seed for bootstrapping
                logger.info(
                    f"Bootstrapping train data with incremented seed={global_seed + i}"
                )
                np.random.seed(global_seed + i)

                bootstrap_indices = np.random.choice(
                    np.arange(len(X_train)), size=len(X_train), replace=True
                )
                X_train_bootstrap = X_train[bootstrap_indices]
                y_train_bootstrap = y_train[bootstrap_indices]
                logger.info("Data bootstrapped")

            # Otherwise use full data for each model
            else:
                X_train_bootstrap = X_train
                y_train_bootstrap = y_train

            # Featurize splits
            logger.info("Featurizing train data")
            bootstrap_dataloader, _, bootstrap_scaler, bootstrap_dataset = (
                self.feat.featurize(
                    X_train_bootstrap,
                    y_train_bootstrap,
                )
            )

            # Save dataloader
            torch.save(
                bootstrap_dataloader,
                bootstrap_dir / "train_dataloader.pth",
            )
            logger.info("Data featurized")

            # Load model from disk
            if (self.ensemble_kwargs.get("param_paths") is not None) and (
                self.ensemble_kwargs.get("serial_paths") is not None
            ):
                logger.info(
                    f"Loading model {i} from disk, overrides any specified parameters."
                )
                self.model = self.model.deserialize(
                    self.ensemble_kwargs.get("param_paths")[i],
                    self.ensemble_kwargs.get("serial_paths")[i],
                    scaler=bootstrap_scaler,
                    **kwargs,
                )
                logger.info(f"Model {i} loaded")

                # Optionally freeze weights
                if self.model_kwargs.get("freeze_weights") is not None:
                    logger.info(f"Freezing weights for model {i}")
                    self.model.freeze_weights(**self.model_kwargs.get("freeze_weights"))
                    logger.info(f"Model {i} frozen")

            # Build model from scratch
            else:
                # Set seed for bootstrap model
                logger.info(
                    f"Building model {i} with incremented seed={global_seed + i} to vary model initialization"
                )
                pl.seed_everything(global_seed + i)

                self.model = self.model.make_new()
                self.model.build(scaler=bootstrap_scaler, **kwargs)
                logger.info(f"Model {i} built")

            # Pass model to trainer
            logger.info("Setting model in trainer")
            self.trainer.model = self.model
            logger.info("Model set in trainer")

            # Append bootstrap index to output directory
            self.trainer.output_dir = bootstrap_dir

            # Prepare the trainer
            logger.info("Preparing trainer")
            self.trainer.build()
            logger.info("Trainer prepared")

            # Commence model training
            logger.info("Training model")
            self.model = self.trainer.train(bootstrap_dataloader, val_dataloader)
            logger.info("Model trained")

            # Add model to list
            models.append(self.model)

        # Create ensemble from trained models
        self.model = self.ensemble.from_models(models)

    def run(
        self,
        output_dir: PathLike = "anvil_training",
        debug: bool = False,
        tag: str = None,
    ) -> Any:
        """
        Run the workflow.

        Parameters
        ----------
        output_dir : PathLike, optional
            Directory to save outputs, by default "anvil_training"
        debug : bool, optional
            Whether to run in debug mode, by default False
        tag : str, optional
            Tag to override the one in the recipe, by default None

        Returns
        -------
        Any
            Result of the workflow run

        """
        # Override the model tag from yaml if provided in cli
        if tag is not None:
            model_tag = tag
        else:
            model_tag = self.metadata.tag

        # Add target_cols for labeling in eval
        target_labels = self.data_spec.target_cols

        # Set debug attribute
        self.debug = debug

        # Cast output directory to string, stripping any trailing separator
        output_dir = str(Path(output_dir))

        # Output directory already exists, create new handle
        if Path(output_dir).exists():
            # Make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:6]

            # Get the date and time in short format
            now = datetime.now().strftime("%Y-%m-%d")

            # Extended output directory
            output_dir = Path(output_dir + f"_{now}_{hsh}")

        # Output directory does not exist, handle as is
        else:
            output_dir = Path(output_dir)

        # Create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_output_dir = output_dir

        # Create data subdirectory
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Log output directory information
        logger.info(f"Running workflow from directory {output_dir}")

        # Log workflow driver selection
        logger.info(f"Running with driver {self._driver_type}")

        # Load data from YAML specification
        logger.info("Loading data")
        if self.data_spec.using_train_test:
            logger.info(
                "Using prespecified train/test resources from data specification"
            )
            X_train, X_val, X_test, y_train, y_val, y_test, X, y = self.data_spec.read()
            groups = None
        else:
            X, y = self.data_spec.read()
            # Split data into train, validation, and test sets
            logger.info("Splitting data from single resource")
            X_train, X_val, X_test, y_train, y_val, y_test, groups = self.split.split(
                X, y
            )
        logger.info("Data loaded")

        # Save splits to CSV outputs
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        if X_val is not None:
            X_val.to_csv(data_dir / "X_val.csv", index=False)
        if X_test is not None:
            X_test.to_csv(data_dir / "X_test.csv", index=False)

        y_train.to_csv(data_dir / "y_train.csv", index=False)
        if y_val is not None:
            y_val.to_csv(data_dir / "y_val.csv", index=False)
        if y_test is not None:
            y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        # Featurize splits
        logger.info("Featurizing data")
        train_dataloader, _, train_scaler, train_dataset = self.feat.featurize(
            X_train,
            y_train,
        )
        torch.save(train_dataloader, output_dir / "train_dataloader.pth")

        if X_val is not None and y_val is not None:
            val_dataloader, _, _, val_dataset = self.feat.featurize(X_val, y_val)
            torch.save(val_dataloader, output_dir / "val_dataloader.pth")
        else:
            val_dataloader = None
            val_dataset = None
            logger.warning("Validation set is None, skipping validation dataloader")

        # Dataloader, indices, scaler, dataset
        if X_test is not None and y_test is not None:
            test_dataloader, _, _, test_dataset = self.feat.featurize(X_test, y_test)
            torch.save(test_dataloader, output_dir / "test_dataloader.pth")
        else:
            test_dataloader = None
            test_dataset = None
            logger.warning("Test set is None, skipping test validation dataloader")

        logger.info("Data featurized")

        kwargs = {}
        if isinstance(self.feat, PairwiseFeaturizer):
            kwargs["input_dim"] = train_dataset[0][0].shape[
                -1
            ]  # this is the dimension of # of features, e.g. 1024 for ECFP4, variable for descriptors
            logger.info(f"Input dim inferred as {kwargs['input_dim']}")
        else:
            logger.info("Input dim not inferred, assuming unpaired data")

        # Train
        if self.ensemble:
            # Ensemble mode
            self._train_ensemble(
                X_train,
                y_train,
                val_dataloader,
                output_dir,
                **kwargs,
            )

            # Calibrate
            self.model.calibrate_uncertainty(
                val_dataloader,
                y_val,
                method=self.ensemble_kwargs.get("calibration_method"),
                accelerator=self.trainer.accelerator,
                devices=self.trainer.devices,
            )

            # Save
            logger.info("Saving model")
            self.model.serialize(
                [
                    output_dir
                    / f"bootstrap_{i}"
                    / self.model.models[i]._model_json_name
                    for i in range(self.model.n_models)
                ],
                [
                    output_dir
                    / f"bootstrap_{i}"
                    / self.model.models[i]._model_save_name
                    for i in range(self.model.n_models)
                ],
                output_dir / self.model._calibration_model_save_name,
            )
            logger.info("Model saved")
        else:
            # Single-model mode
            self._train(
                train_dataloader,
                val_dataloader,
                train_scaler,
                output_dir,
                **kwargs,
            )

            # Save
            logger.info("Saving model")
            self.model.serialize(
                output_dir / self.model._model_json_name,
                output_dir / self.model._model_save_name,
            )
            logger.info("Model saved")

        # initialize y_pred & y_std; needed if training a no split model
        y_pred = None
        y_std = None

        if test_dataloader is not None:
            # Predict on test set
            logger.info("Predicting")
            if self.ensemble:
                y_pred, y_std = self.model.predict(
                    test_dataloader,
                    accelerator=self.trainer.accelerator,
                    devices=self.trainer.devices,
                    return_std=True,
                )
            else:
                y_pred = self.model.predict(
                    test_dataloader,
                    accelerator=self.trainer.accelerator,
                    devices=self.trainer.devices,
                )
                y_std = None
            logger.info("Predictions made")
        else:
            logger.info("No test set specified, predictions skipped")

        # Run evaluation on train/test
        logger.info("Evaluating")

        # Get wandb bool from trainer
        use_wandb = self.trainer.use_wandb

        # Run evaluation on train/test
        for eval in self.evals:
            # Here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                y_std=y_std,
                model=self.model,
                X_train=train_dataloader,
                y_train=train_dataloader,
                X_all=X,
                groups=groups,
                y_all=y,
                featurizer=self.feat,
                trainer=self.trainer,
                use_wandb=use_wandb,
                tag=model_tag,
                target_labels=target_labels,
            )

            # Write evaluation report
            eval.report(write=True, output_dir=output_dir)

        logger.info("Evaluation done")


_DRIVER_TO_CLASS = {
    DriverType.SKLEARN: AnvilWorkflow,
    DriverType.LIGHTNING: AnvilDeepLearningWorkflow,
}
