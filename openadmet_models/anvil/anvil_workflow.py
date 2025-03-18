import hashlib
import uuid
from abc import abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import fsspec
import torch
import yaml
import zarr
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from openadmet_models.data.data_spec import DataSpec
from openadmet_models.eval.eval_base import EvalBase, get_eval_class
from openadmet_models.features.feature_base import FeaturizerBase, get_featurizer_class
from openadmet_models.models.model_base import ModelBase, get_model_class
from openadmet_models.registries import *  # noqa: F401, F403
from openadmet_models.split.split_base import SplitterBase, get_splitter_class
from openadmet_models.trainer.trainer_base import TrainerBase, get_trainer_class
from openadmet_models.util.types import Pathy

_SECTION_CLASS_GETTERS = {
    "feat": get_featurizer_class,
    "model": get_model_class,
    "split": get_splitter_class,
    "eval": get_eval_class,
    "train": get_trainer_class,
    "INVALID": lambda x: None,
}


class SpecBase(BaseModel):

    def to_yaml(self, path, **storage_options):
        with fsspec.open(path, "w", **storage_options) as stream:
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_yaml(cls, path, **storage_options):
        of = fsspec.open(path, "r", **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)
        return cls(**data)


class Drivers(StrEnum):
    """
    Enum for the drivers
    """

    PYTORCH = "pytorch"
    SKLEARN = "sklearn"


class Metadata(SpecBase):
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
    type: str
    params: dict = {}
    section_name: ClassVar[str] = "INVALID"

    def to_class(self):
        return _SECTION_CLASS_GETTERS[self.section_name](self.type)(**self.params)


class SplitSpec(AnvilSection):
    section_name: ClassVar[str] = "split"


class FeatureSpec(AnvilSection):
    section_name: ClassVar[str] = "feat"


class ModelSpec(AnvilSection):
    section_name: ClassVar[str] = "model"


class TrainerSpec(AnvilSection):
    section_name: ClassVar[str] = "train"


class EvalSpec(AnvilSection):
    section_name: ClassVar[str] = "eval"


class ProcedureSpec(SpecBase):
    section_name: ClassVar[str] = "procedure"

    split: SplitSpec
    feat: FeatureSpec
    model: ModelSpec
    train: TrainerSpec


class ReportSpec(SpecBase):
    section_name: ClassVar[str] = "report"
    eval: list[EvalSpec]


class AnvilSpecification(BaseModel):
    metadata: Metadata
    data: DataSpec
    procedure: ProcedureSpec
    report: ReportSpec

    # need repetition of YAML loaders here to properly set anvil_dir
    # and to not expose to_yaml and from_yaml to the user
    @classmethod
    def from_recipe(cls, yaml_path: Pathy, **storage_options):
        of = fsspec.open(yaml_path, "r", **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)
        parent = of.fs.unstrip_protocol(of.fs._parent(yaml_path))
        instance = cls(**data)
        # make sure to set the anvil_dir
        instance.data.template_anvil_dir(parent)
        return instance

    def to_recipe(self, path, **storage_options):
        with fsspec.open(path, "w", **storage_options) as stream:
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
        metadata = Metadata.from_yaml(metadata_yaml, **storage_options)
        data = DataSpec.from_yaml(data_yaml, **storage_options)
        procedure = ProcedureSpec.from_yaml(procedure_yaml, **storage_options)
        report = ReportSpec.from_yaml(report_yaml, **storage_options)
        return cls(metadata=metadata, data=data, procedure=procedure, report=report)

    def to_multi_yaml(
        self,
        metadata_yaml="metadata.yaml",
        procedure_yaml="procedure.yaml",
        data_yaml="data.yaml",
        report_yaml="eval.yaml",
        **storage_options,
    ):
        self.metadata.to_yaml(metadata_yaml, **storage_options)
        self.data.to_yaml(data_yaml, **storage_options)
        self.procedure.to_yaml(procedure_yaml, **storage_options)
        self.report.to_yaml(report_yaml, **storage_options)

    def to_workflow(self):
        metadata = self.metadata
        data_spec = self.data
        transform = None
        split = self.procedure.split.to_class()
        feat = self.procedure.feat.to_class()
        model = self.procedure.model.to_class()
        trainer = self.procedure.train.to_class()
        evals = [eval.to_class() for eval in self.report.eval]

        logger.info("Making workflow from specification")

        return _DRIVER_TO_CLASS[self.metadata.driver](
            metadata=metadata,
            data_spec=data_spec,
            model=model,
            transform=transform,
            split=split,
            feat=feat,
            trainer=trainer,
            evals=evals,
            parent_spec=self,
        )


class AnvilWorkflowBase(BaseModel):
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
    def run(self, output_dir: Pathy = "anvil_run", debug: bool = False) -> Any: ...


class AnvilWorkflow(AnvilWorkflowBase):

    driver: Drivers = Drivers.SKLEARN

    def run(self, output_dir: Pathy = "anvil_run", debug: bool = False, tag: str = None) -> Any:
        """
        Run the workflow
        """
        #override the model tag from yaml if provided in cli
        if tag is not None:
            model_tag = tag
        else:
            model_tag = self.metadata.tag

        self.debug = debug
        output_dir = str(output_dir)
        if Path(output_dir).exists():
            # make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:6]
            # get the date and time in short format
            now = datetime.now().strftime("%Y-%m-%d")
            output_dir = Path(output_dir + f"_{now}_{hsh}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # write recipe to output directory
        self.parent_spec.to_recipe(output_dir / "anvil_recipe.yaml")

        recipe_components = Path(output_dir / "recipe_components")
        recipe_components.mkdir(parents=True, exist_ok=True)
        self.parent_spec.to_multi_yaml(
            metadata_yaml=recipe_components / "metadata.yaml",
            procedure_yaml=recipe_components / "procedure.yaml",
            data_yaml=recipe_components / "data.yaml",
            report_yaml=recipe_components / "eval.yaml",
        )

        logger.info(f"Running workflow from directory {output_dir}")

        logger.info(f"Running with driver {self.driver}")

        logger.info("Loading data")
        X, y = self.data_spec.read()
        logger.info("Data loaded")

        logger.info("Transforming data")
        if self.transform:
            X = self.transform.transform(X)
            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = self.split.split(X, y)

        X_train.to_csv(data_dir / "X_train.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        logger.info("Featurizing data")
        X_train_feat, _ = self.feat.featurize(X_train)
        zarr.save(data_dir / "X_train_feat.zarr", X_train_feat)

        X_test_feat, _ = self.feat.featurize(X_test)
        zarr.save(data_dir / "X_test_feat.zarr", X_test_feat)
        logger.info("Data featurized")

        logger.info("Building model")
        self.model.build()
        logger.info("Model built")

        logger.info("Setting model in trainer")
        self.trainer.model = self.model
        logger.info("Model set in trainer")

        logger.info("Training model")
        self.model = self.trainer.train(X_train_feat, y_train)
        logger.info("Model trained")

        logger.info("Saving model")
        self.model.serialize(output_dir / "model.json", output_dir / "model.pkl")
        logger.info("Model saved")

        logger.info("Predicting")
        y_pred = self.model.predict(X_test_feat)
        logger.info("Predictions made")

        logger.info("Evaluating")
        for eval in self.evals:
            # here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                model=self.model,
                X_train=X_train_feat,
                y_train=y_train,
                tag=model_tag,
            )
            eval.report(write=True, output_dir=output_dir)
        logger.info("Evaluation done")


class AnvilDeepLearningWorkflow(AnvilWorkflowBase):

    driver: Drivers = Drivers.PYTORCH

    def run(self, output_dir: Pathy = "anvil_run", debug: bool = False, tag: str = None) -> Any:
        """
        Run the workflow
        """
        #override the model tag from yaml if provided in cli
        if tag is not None:
            model_tag = tag
        else:
            model_tag = self.metadata.tag

        self.debug = debug
        output_dir = str(output_dir)
        if Path(output_dir).exists():
            # make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:6]
            # get the date and time in short format
            now = datetime.now().strftime("%Y-%m-%d")
            output_dir = Path(output_dir + f"_{now}_{hsh}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # write recipe to output directory
        self.parent_spec.to_recipe(output_dir / "anvil_recipe.yaml")

        recipe_components = Path(output_dir / "recipe_components")
        recipe_components.mkdir(parents=True, exist_ok=True)
        self.parent_spec.to_multi_yaml(
            metadata_yaml=recipe_components / "metadata.yaml",
            procedure_yaml=recipe_components / "procedure.yaml",
            data_yaml=recipe_components / "data.yaml",
            report_yaml=recipe_components / "eval.yaml",
        )

        logger.info(f"Running workflow from directory {output_dir}")

        logger.info(f"Running with driver {self.driver}")

        logger.info("Loading data")
        X, y = self.data_spec.read()
        logger.info("Data loaded")

        logger.info("Transforming data")
        if self.transform:
            X = self.transform.transform(X)
            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = self.split.split(X, y)

        X_train.to_csv(data_dir / "X_train.csv", index=False)
        X_test.to_csv(data_dir / "X_test.csv", index=False)
        y_train.to_csv(data_dir / "y_train.csv", index=False)
        y_test.to_csv(data_dir / "y_test.csv", index=False)

        logger.info("Data split")

        logger.info("Featurizing data")
        train_dataloader, train_scaler = self.feat.featurize(X_train, y_train)
        torch.save(train_dataloader, output_dir / "train_dataloader.pth")

        test_dataloader, test_scaler = self.feat.featurize(X_test, y_test)
        torch.save(test_dataloader, output_dir / "test_dataloader.pth")
        logger.info("Data featurized")

        logger.info("Building model")
        self.model.build(scaler=train_scaler)
        logger.info("Model built")

        logger.info("Setting model in trainer")
        self.trainer.model = self.model
        logger.info("Model set in trainer")

        # check if there is a output dir
        if not self.trainer.output_dir:
            self.trainer.output_dir = output_dir

        logger.info("Preparing trainer")
        self.trainer.prepare()
        logger.info("Trainer prepared")

        logger.info("Training model")
        self.model = self.trainer.train(train_dataloader)
        logger.info("Model trained")

        logger.info("Saving model")
        self.model.serialize(output_dir / "model.json", output_dir / "model.pth")
        logger.info("Model saved")

        logger.info("Predicting")
        y_pred = self.model.predict(test_dataloader)
        logger.info("Predictions made")

        logger.info("Evaluating")

        use_wandb = self.trainer.use_wandb

        for eval in self.evals:
            # here all the data is passed to the evaluator, but some evaluators may only need a subset
            eval.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                model=self.model,
                X_train=train_dataloader,
                y_train=train_dataloader,
                use_wandb=use_wandb,
                tag=model_tag,
            )
            eval.report(write=True, output_dir=output_dir)
        logger.info("Evaluation done")


_DRIVER_TO_CLASS = {
    Drivers.SKLEARN: AnvilWorkflow,
    Drivers.PYTORCH: AnvilDeepLearningWorkflow,
}
