import hashlib
import uuid
from abc import abstractmethod
from datetime import datetime
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Literal

import fsspec
import torch
import yaml
import zarr
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, model_validator

from openadmet.models.anvil.data_spec import DataSpec
from openadmet.models.architecture.model_base import ModelBase, get_mod_class
from openadmet.models.eval.eval_base import EvalBase, get_eval_class
from openadmet.models.features.feature_base import FeaturizerBase, get_featurizer_class
from openadmet.models.registries import *  # noqa: F401, F403
from openadmet.models.split.split_base import SplitterBase, get_splitter_class
from openadmet.models.trainer.trainer_base import TrainerBase, get_trainer_class

_SECTION_CLASS_GETTERS = {
    "feat": get_featurizer_class,
    "model": get_mod_class,
    "split": get_splitter_class,
    "eval": get_eval_class,
    "train": get_trainer_class,
    "INVALID": lambda x: None,
}


class SpecBase(BaseModel):
    """
    Base class for specifications.
    """

    def to_yaml(self, path, **storage_options):
        """
        Write specification to YAML file.
        """

        # Open file stream
        with fsspec.open(path, "w", **storage_options) as stream:
            # Safe dump the model to stream
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_yaml(cls, path, **storage_options):
        """
        Load specification from YAML file.
        """

        # Open file stream
        with fsspec.open(path, "r", **storage_options) as stream:
            # Safe load the model from stream
            data = yaml.safe_load(stream)

        # Pass YAML content to class constructor
        return cls(**data)


class Drivers(StrEnum):
    """
    Enum for the drivers.
    """

    PYTORCH = "pytorch"
    SKLEARN = "sklearn"


class Metadata(SpecBase):
    """
    Metadata specification.
    """

    version: Literal["v1"] = Field(
        ..., description="The version of the metadata schema."
    )
    driver: str = Field(
        Drivers.SKLEARN.value, description="The driver for the workflow."
    )

    name: str = Field(..., description="The name of the workflow.")
    build_number: int = Field(
        ...,
        ge=0,
        description="The build number of the workflow (must be non-negative).",
    )
    description: str = Field(..., description="Description of the workflow.")
    tag: str = Field(..., description="Primary tag for the workflow.")
    authors: str = Field(..., description="Name of the authors.")
    email: EmailStr = Field(..., description="Email address of the contact person.")
    # date_created: datetime = Field(
    #     ..., alias="date-created", description="Date when the workflow was created."
    # )
    biotargets: list[str] = Field(
        ..., description="List of biotargets associated with the workflow."
    )
    tags: list[str] = Field(..., description="Additional tags for the workflow.")


class AnvilSection(SpecBase):
    """
    Anvil specification section base class.
    """

    type: str
    params: dict = {}
    section_name: ClassVar[str] = "INVALID"

    def to_class(self):
        return _SECTION_CLASS_GETTERS[self.section_name](self.type)(**self.params)


class SplitSpec(AnvilSection):
    """
    Data split specification.
    """

    section_name: ClassVar[str] = "split"


class FeatureSpec(AnvilSection):
    """
    Featurization specification.
    """

    section_name: ClassVar[str] = "feat"


class ModelSpec(AnvilSection):
    """
    Model specification.
    """

    section_name: ClassVar[str] = "model"


class TrainerSpec(AnvilSection):
    """
    Trainer specification.
    """

    section_name: ClassVar[str] = "train"


class EvalSpec(AnvilSection):
    """
    Evaluation specification.
    """

    section_name: ClassVar[str] = "eval"


class ProcedureSpec(SpecBase):
    """
    Procedure specification.
    """

    section_name: ClassVar[str] = "procedure"

    split: SplitSpec
    feat: FeatureSpec
    model: ModelSpec
    train: TrainerSpec


class ReportSpec(SpecBase):
    """
    Report specification.
    """

    section_name: ClassVar[str] = "report"
    eval: list[EvalSpec]


class AnvilSpecification(BaseModel):
    """
    Full specification for Anvil workflow.
    """

    metadata: Metadata
    data: DataSpec
    procedure: ProcedureSpec
    report: ReportSpec

    # Need repetition of YAML loaders here to properly set `anvil_dir`
    # and to not expose to_yaml and from_yaml to the user
    @classmethod
    def from_recipe(cls, yaml_path: PathLike, **storage_options):
        """
        Load specification from YAML recipe file.
        """

        # Load YAML file
        of = fsspec.open(yaml_path, "r", **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)

        # Parse parent protocol
        parent = of.fs.unstrip_protocol(of.fs._parent(yaml_path))

        # Instantiate specification with loaded data
        instance = cls(**data)

        # Set the anvil_dir
        instance.data.template_anvil_dir(parent)

        return instance

    def to_recipe(self, path, **storage_options):
        """
        Write specification to YAML recipe file.
        """

        # Open file stream
        with fsspec.open(path, "w", **storage_options) as stream:
            # Safe dump the model to stream
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_multi_yaml(
        cls,
        metadata_yaml="metadata.yaml",
        procedure_yaml="procedure.yaml",
        data_yaml="data.yaml",
        report_yaml="eval.yaml",
        **storage_options,
    ):
        """
        Load specification from multiple YAML files.
        """

        # Load YAML files
        metadata = Metadata.from_yaml(metadata_yaml, **storage_options)
        data = DataSpec.from_yaml(data_yaml, **storage_options)
        procedure = ProcedureSpec.from_yaml(procedure_yaml, **storage_options)
        report = ReportSpec.from_yaml(report_yaml, **storage_options)

        # Instantiate the class with loaded data
        return cls(metadata=metadata, data=data, procedure=procedure, report=report)

    def to_multi_yaml(
        self,
        metadata_yaml="metadata.yaml",
        procedure_yaml="procedure.yaml",
        data_yaml="data.yaml",
        report_yaml="eval.yaml",
        **storage_options,
    ):
        """
        Write specification to multiple YAML files.
        """

        # Write each section to its own YAML file
        self.metadata.to_yaml(metadata_yaml, **storage_options)
        self.data.to_yaml(data_yaml, **storage_options)
        self.procedure.to_yaml(procedure_yaml, **storage_options)
        self.report.to_yaml(report_yaml, **storage_options)

    def to_workflow(self):
        """
        Convert the specification to a workflow object.
        """

        logger.info("Making workflow from specification")

        return _DRIVER_TO_CLASS[self.metadata.driver](
            metadata=self.metadata,
            data_spec=self.data,
            model=self.procedure.model.to_class(),
            transform=None,
            split=self.procedure.split.to_class(),
            feat=self.procedure.feat.to_class(),
            trainer=self.procedure.train.to_class(),
            evals=[eval.to_class() for eval in self.report.eval],
            parent_spec=self,
        )


class AnvilWorkflowBase(BaseModel):
    """
    Base class for Anvil workflows.
    """

    metadata: Metadata
    data_spec: DataSpec
    transform: Any
    split: SplitterBase
    feat: FeaturizerBase
    model: ModelBase
    trainer: TrainerBase
    evals: list[EvalBase]
    parent_spec: AnvilSpecification
    debug: bool = False

    @abstractmethod
    def run(self, output_dir: PathLike = "anvil_run", debug: bool = False) -> Any: ...


class AnvilWorkflow(AnvilWorkflowBase):
    """
    Workflow for running basic Anvil configuration.
    """

    driver: Drivers = Drivers.SKLEARN

    @model_validator(mode="after")
    def check_for_no_val(self):
        # Check that val_size is set to zero
        if self.split.val_size != 0:
            raise ValueError(
                "Validation set requested, but not supported in this workflow."
            )
        return self

    def run(
        self, output_dir: PathLike = "anvil_run", debug: bool = False, tag: str = None
    ) -> Any:
        """
        Run the workflow.
        """

        # Override the model tag from yaml if provided in cli
        if tag is not None:
            model_tag = tag
        else:
            model_tag = self.metadata.tag

        target_labels = self.data_spec.target_cols

        # Set debug attribute
        self.debug = debug

        # Cast output directory to string
        output_dir = str(output_dir)

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

        # Create data subdirectory
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write recipe to output directory
        self.parent_spec.to_recipe(output_dir / "anvil_recipe.yaml")

        # Split recipe into components and save
        recipe_components = Path(output_dir / "recipe_components")
        recipe_components.mkdir(parents=True, exist_ok=True)
        self.parent_spec.to_multi_yaml(
            metadata_yaml=recipe_components / "metadata.yaml",
            procedure_yaml=recipe_components / "procedure.yaml",
            data_yaml=recipe_components / "data.yaml",
            report_yaml=recipe_components / "eval.yaml",
        )

        # Log output directory information
        logger.info(f"Running workflow from directory {output_dir}")

        # Log workflow driver selection
        logger.info(f"Running with driver {self.driver}")

        # Load data from YAML specification
        logger.info("Loading data")
        X, y = self.data_spec.read()
        logger.info("Data loaded")

        # Transform data
        logger.info("Transforming data")
        if self.transform:
            X = self.transform.transform(X)
            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        # Split data into train, validation, and test sets
        logger.info("Splitting data")
        X_train, _, X_test, y_train, _, y_test = self.split.split(X, y)

        # Save splits to CSV outputs
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        # Featurize splits
        logger.info("Featurizing data")
        X_train_feat, _ = self.feat.featurize(X_train)
        zarr.save(data_dir / "X_train_feat.zarr", X_train_feat)

        X_test_feat, _ = self.feat.featurize(X_test)
        zarr.save(data_dir / "X_test_feat.zarr", X_test_feat)

        logger.info("Data featurized")

        # Build model
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

        # Save serialized model
        logger.info("Saving model")
        self.model.serialize(output_dir / "model.json", output_dir / "model.pkl")
        logger.info("Model saved")

        # Predict on test set
        logger.info("Predicting")
        # Check if the model has predict_proba method (classification)
        if hasattr(self.model, "predict_proba"):
            y_pred = self.model.predict_proba(X_test_feat)

        # Otherwise, regression
        else:
            y_pred = self.model.predict(X_test_feat)
        logger.info("Predictions made")

        # Run evaluation on train/test
        logger.info("Evaluating")
        for eval in self.evals:
            # Here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                model=self.model,
                X_train=X_train_feat,
                y_train=y_train,
                tag=model_tag,
                target_labels=target_labels,
            )

            # Write evaluation report
            eval.report(write=True, output_dir=output_dir)

        logger.info("Evaluation done")


class AnvilDeepLearningWorkflow(AnvilWorkflowBase):
    """
    Workflow for running deep learning Anvil configuration.
    """

    driver: Drivers = Drivers.PYTORCH

    def run(
        self, output_dir: PathLike = "anvil_run", debug: bool = False, tag: str = None
    ) -> Any:
        """
        Run the workflow
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

        # Cast output directory to string
        output_dir = str(output_dir)

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

        # Create data subdirectory
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write recipe to output directory
        self.parent_spec.to_recipe(output_dir / "anvil_recipe.yaml")

        # Split recipe into components and save
        recipe_components = Path(output_dir / "recipe_components")
        recipe_components.mkdir(parents=True, exist_ok=True)
        self.parent_spec.to_multi_yaml(
            metadata_yaml=recipe_components / "metadata.yaml",
            procedure_yaml=recipe_components / "procedure.yaml",
            data_yaml=recipe_components / "data.yaml",
            report_yaml=recipe_components / "eval.yaml",
        )

        # Log output directory information
        logger.info(f"Running workflow from directory {output_dir}")

        # Log workflow driver selection
        logger.info(f"Running with driver {self.driver}")

        # Load data from YAML specification
        logger.info("Loading data")
        X, y = self.data_spec.read()
        logger.info("Data loaded")

        # Transform data
        logger.info("Transforming data")
        if self.transform:
            X = self.transform.transform(X)
            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        # Split data into train, validation, and test sets
        logger.info("Splitting data")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split.split(X, y)

        # Save splits to CSV outputs
        X_train.to_csv(data_dir / "X_train.csv", index=False)
        if X_val is not None:
            X_val.to_csv(data_dir / "X_val.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        if y_val is not None:
            y_val.to_csv(data_dir / "y_val.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        # Featurize splits
        logger.info("Featurizing data")
        train_dataloader, _, train_scaler, train_dataset = self.feat.featurize(
            X_train, y_train
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
        test_dataloader, _, _, test_dataset = self.feat.featurize(X_test, y_test)
        torch.save(test_dataloader, output_dir / "test_dataloader.pth")
        logger.info("Data featurized")

        # Build model
        logger.info("Building model")
        self.model.build(scaler=train_scaler)
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
        self.trainer.prepare()
        logger.info("Trainer prepared")

        # Commence model training
        logger.info("Training model")
        self.model = self.trainer.train(train_dataloader, val_dataloader)
        logger.info("Model trained")

        # Save serialized model
        logger.info("Saving model")
        self.model.serialize(output_dir / "model.json", output_dir / "model.pth")
        logger.info("Model saved")

        # Predict on test set
        logger.info("Predicting")
        y_pred = self.model.predict(
            test_dataloader,
            accelerator=self.trainer.accelerator,
            devices=self.trainer.devices,
        )

        logger.info("Predictions made")

        # Run evaluation on train/test
        logger.info("Evaluating")

        # Get wandb bool from trainer
        use_wandb = self.trainer.use_wandb

        # Run evaluation on train/test
        for eval in self.evals:
            # Here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test.values,  # Pass as array instead of series
                y_pred=y_pred,
                model=self.model,
                X_train=train_dataloader,
                y_train=train_dataloader,
                X_train_raw=X_train,
                y_train_raw=y_train,
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
    Drivers.SKLEARN: AnvilWorkflow,
    Drivers.PYTORCH: AnvilDeepLearningWorkflow,
}
