import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
import zarr
import fsspec
import yaml
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


class Metadata(SpecBase):
    version: Literal["v1"] = Field(
        ..., description="The version of the metadata schema."
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

        return AnvilWorkflow(
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


class AnvilWorkflow(BaseModel):
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

    def run(self, output_dir: Pathy = "anvil_run", debug: bool=False) -> Any:
        """
        Run the workflow
        """
        self.debug = debug
        output_dir = str(output_dir)
        if Path(output_dir).exists():
            # make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:6]
            # get the date and time in short format
            now = datetime.now().strftime("%Y-%m-%d")
            output_dir = Path(output_dir + f"{now}_{hsh}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

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

        if self.debug:
            # save the split data to CSVs 
            X_train.to_csv(output_dir / "X_train.csv", index=False)
            X_test.to_csv(output_dir / "X_test.csv", index=False)
            zarr.save(output_dir /"y_train.zarr", y_train)
            zarr.save(output_dir / "y_test.zarr", y_test)
        logger.info("Data split")

        logger.info("Featurizing data")
        X_train_feat, _ = self.feat.featurize(X_train)
        if self.debug:
            # save the featurized data to CSVs 
            zarr.save(output_dir /"X_train_feat.zarr", X_train_feat)
        X_test_feat, _ = self.feat.featurize(X_test)
        if self.debug:
            # save the featurized data to CSVs 
            zarr.save(output_dir /"X_test_feat.zarr", X_test_feat)
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
        self.model.to_model_json_and_pkl(
            output_dir / "model.json", output_dir / "model.pkl"
        )
        logger.info("Model saved")

        logger.info("Predicting")
        preds = self.model.predict(X_test_feat)
        logger.info("Predictions made")

        logger.info("Evaluating")
        for eval in self.evals:
            eval.evaluate(y_test, preds)
            eval.report(write=True, output_dir=output_dir)
        logger.info("Evaluation done")
