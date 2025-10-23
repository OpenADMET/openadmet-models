"""Specification models for Anvil workflows."""

from os import PathLike
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import fsspec
import intake
import jinja2
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from openadmet.models.active_learning.ensemble_base import (
    get_ensemble_class,
)
from openadmet.models.anvil import Drivers
from openadmet.models.architecture.model_base import get_mod_class
from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.features.feature_base import get_featurizer_class
from openadmet.models.registries import *  # noqa: F401, F403
from openadmet.models.split.split_base import get_splitter_class
from openadmet.models.trainer.trainer_base import get_trainer_class
from openadmet.models.transforms.transform_base import (
    get_transform_class,
)

_SECTION_CLASS_GETTERS = {
    "feat": get_featurizer_class,
    "model": get_mod_class,
    "ensemble": get_ensemble_class,
    "split": get_splitter_class,
    "eval": get_eval_class,
    "train": get_trainer_class,
    "transform": get_transform_class,
    "INVALID": lambda x: None,
}


class DataSpec(BaseModel):
    """
    Data specification for the workflow.

    Attributes
    ----------
    type : str
        The type of data source (e.g., 'csv', 'yaml').
    resource : str
        The path or URL to the data resource.
    cat_entry : Optional[str]
        The catalog entry name if the resource is a YAML catalog.
    target_cols : Union[str, list[str]]
        The target column(s) in the dataset.
    input_col : str
        The input column in the dataset.
    anvil_dir : Optional[str]
        The base directory for relative paths.
    dropna : Optional[bool]
        Whether to drop rows with NaN values.
    train_resource : Optional[str]
        The path or URL to the training data resource (if using separate train/test).
    test_resource : Optional[str]
        The path or URL to the testing data resource (if using separate train/test).
    val_resource : Optional[str]
        The path or URL to the validation data resource (if using separate train/test).
    _catalog : Optional[intake.catalog.Catalog]
        The intake catalog object if the resource is a YAML file.
    _using_train_test : bool
        Whether using separate train and test resources.

    """

    type: str
    resource: Optional[str] = None

    cat_entry: Optional[str] = None
    target_cols: Union[str, list[str]]
    input_col: str
    anvil_dir: Optional[str] = None
    dropna: Optional[bool] = False
    train_resource: Optional[str] = None
    test_resource: Optional[str] = None
    val_resource: Optional[str] = None

    _catalog: Optional[intake.catalog.Catalog] = None
    _using_train_test: bool = False

    @property
    def using_train_test(self):
        """Whether using separate train and test resources."""
        return self._using_train_test

    @model_validator(mode="after")
    def check_resource_test_train(self):
        """Ensure that either resource or train/test/val resources are provided, not both."""
        if self.resource and (
            self.train_resource or self.test_resource or self.val_resource
        ):
            raise ValueError(
                "Specify either `resource` or `train_resource`/`test_resource`/`val_resource`, not both."
            )
        if self.train_resource or self.test_resource or self.val_resource:
            if not (self.train_resource and self.test_resource):
                raise ValueError(
                    "`train_resource` and `test_resource` must both be specified when using separate resources."
                )
            self._using_train_test = True
        return self

    @field_validator("target_cols", mode="before")
    @classmethod
    def check_target_cols_input(cls, v):
        """Ensure target_cols is always a list."""
        if isinstance(v, str):
            return [v]
        else:
            return v

    # validator to template the resource with ANVIL_DIR if present
    @model_validator(mode="after")
    def template_resource(self):
        """
        Template the resource with ANVIL_DIR if present.

        Returns
        -------
        self : DataSpec
            The DataSpec instance with the templated resource.

        """
        if self.anvil_dir:
            if self.resource:
                template = jinja2.Template(self.resource)
                self.resource = template.render(ANVIL_DIR=self.anvil_dir)
        return self

    def template_anvil_dir(self, anvil_dir: Path):
        """Template all resources with ANVIL_DIR if present."""
        self.anvil_dir = anvil_dir

        for attr in ["resource", "train_resource", "test_resource", "val_resource"]:
            value = getattr(self, attr, None)
            if value:
                setattr(self, attr, jinja2.Template(value).render(ANVIL_DIR=anvil_dir))

    def read(self) -> tuple[pd.Series, pd.Series]:
        """
        Read the data from the resource.

        Returns
        -------
        input: pd.Series
            The input data (e.g., SMILES strings)
        targets: pd.Series
            The target data (e.g., properties to predict)

        """
        return (
            self._read_train_test_val()
            if self._using_train_test
            else self._read_single_resource()
        )

    @staticmethod
    def _read_csv_or_parquet(resource: str) -> pd.DataFrame:
        """Read data from a CSV or Parquet resource."""
        if resource.endswith(".csv"):
            return intake.open_csv(resource).read()
        elif any(resource.endswith(x) for x in [".parquet", ".pq", ".pqt"]):
            return intake.open_parquet(resource).read()
        raise ValueError(f"Unsupported resource type: {resource}")

    def _read_train_test_val(self) -> tuple[pd.Series, ...]:
        """Read data from separate train/test/validation resources."""
        if not self.train_resource or not self.test_resource:
            raise ValueError("Both train_resource and test_resource must be specified.")

        def read_split(resource: str, split_name: str) -> pd.DataFrame:
            if resource.endswith((".yaml", ".yml")):
                raise ValueError(
                    "YAML catalogs not supported with train/test resources."
                )
            data = self._read_csv_or_parquet(resource)
            if "_split" in data.columns:
                raise ValueError(
                    f"{split_name.capitalize()} data should not contain a '_split' column."
                )
            data["_split"] = split_name
            return data

        # Read and combine data
        splits_to_read = [(self.train_resource, "train"), (self.test_resource, "test")]
        if self.val_resource:
            splits_to_read.append((self.val_resource, "val"))

        combined = pd.concat(
            [read_split(resource, split) for resource, split in splits_to_read]
        )

        target_cols = (
            self.target_cols
            if isinstance(self.target_cols, list)
            else [self.target_cols]
        )
        combined = combined[[self.input_col] + target_cols + ["_split"]]

        # Handle NaN values
        n_before = len(combined)
        if self.dropna:
            combined = combined.dropna().reset_index(drop=True)
            logger.info(
                f"{n_before} total rows. {n_before - len(combined)} NaN rows were dropped."
            )
        else:
            logger.info(f"{n_before} total rows. 0 NaN rows were dropped.")

        # Split and return (X values, then Y values)
        train = combined[combined["_split"] == "train"]
        test = combined[combined["_split"] == "test"]

        val = combined[combined["_split"] == "val"] if self.val_resource else None

        X_train = train[self.input_col]
        X_test = test[self.input_col]
        X_val = val[self.input_col] if val is not None else None

        y_train = train[self.target_cols]
        y_test = test[self.target_cols]
        y_val = val[self.target_cols] if val is not None else None

        # also return full X, y for reference
        X = combined[self.input_col]
        y = combined[self.target_cols]

        return X_train, X_val, X_test, y_train, y_val, y_test, X, y

    def _read_single_resource(self) -> tuple[pd.Series, pd.Series]:
        """Read data from a single resource."""
        # Read data
        if self.resource.endswith((".yaml", ".yml")):
            if not self.cat_entry:
                raise ValueError("cat_entry must be specified for YAML resources.")
            self._catalog = intake.open_catalog(self.resource)
            if self.cat_entry not in self._catalog:
                raise ValueError(
                    f"cat_entry '{self.cat_entry}' not found in catalog '{self.resource}'."
                )
            data = self._catalog[self.cat_entry]().read()
        else:
            data = self._read_csv_or_parquet(self.resource)

        # Select and clean columns
        target_cols = (
            self.target_cols
            if isinstance(self.target_cols, list)
            else [self.target_cols]
        )
        combined = data[[self.input_col] + target_cols]

        n_before = len(combined)
        if self.dropna:
            combined = combined.dropna().reset_index(drop=True)
            logger.info(
                f"{n_before} total rows. {n_before - len(combined)} NaN rows were dropped."
            )
        else:
            logger.info(f"{n_before} total rows. 0 NaN rows were dropped.")

        return combined[self.input_col], combined[self.target_cols]

    @property
    def catalog(self):
        """Get the intake catalog if the resource is a YAML file."""
        return self._catalog

    def to_yaml(self, path, **storage_options):
        """
        Write specification to YAML file.

        Parameters
        ----------
        path : str or PathLike
            The file path to write the YAML content to.
        storage_options : dict, optional
            Additional options to pass to the file system (e.g., for S3, GCS).

        """
        with fsspec.open(path, "w", **storage_options) as stream:
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_yaml(cls, path, **storage_options):
        """
        Load specification from YAML file.

        Parameters
        ----------
        path : str or PathLike
            The file path to read the YAML content from.
        storage_options : dict, optional
            Additional options to pass to the file system (e.g., for S3, GCS).

        Returns
        -------
        instance : DataSpec
            An instance of the DataSpec class populated with data from the YAML file.

        """
        of = fsspec.open(path, "r", **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)
        return cls(**data)


class SpecBase(BaseModel):
    """Base class for specifications."""

    def to_yaml(self, path, **storage_options):
        """
        Write specification to YAML file.

        Parameters
        ----------
        path : str or PathLike
            The file path to write the YAML content to.
        storage_options : dict, optional
            Additional options to pass to the file system (e.g., for S3, GCS).

        """
        # Open file stream
        with fsspec.open(path, "w", **storage_options) as stream:
            # Safe dump the model to stream
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_yaml(cls, path, **storage_options):
        """
        Load specification from YAML file.

        Parameters
        ----------
        path : str or PathLike
            The file path to read the YAML content from.
        storage_options : dict, optional
            Additional options to pass to the file system (e.g., for S3, GCS)

        Returns
        -------
        instance : SpecBase
            An instance of the specification class populated with data from the YAML file.

        """
        # Open file stream
        with fsspec.open(path, "r", **storage_options) as stream:
            # Safe load the model from stream
            data = yaml.safe_load(stream)

        # Pass YAML content to class constructor
        return cls(**data)


class Metadata(SpecBase):
    """
    Metadata specification.

    Attributes
    ----------
    version : Literal["v1"]
        The version of the metadata schema.
    driver : str
        The driver for the workflow.
    name : str
        The name of the workflow.
    build_number : int
        The build number of the workflow (must be non-negative).
    description : str
        Description of the workflow.
    tag : str
        Primary tag for the workflow.
    authors : str
        Name of the authors.
    email : EmailStr
        Email address of the contact person.
    biotargets : list[str]
        List of biotargets associated with the workflow.
    tags : list[str]
        Additional tags for the workflow.

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
    biotargets: list[str] = Field(
        ..., description="List of biotargets associated with the workflow."
    )
    tags: list[str] = Field(..., description="Additional tags for the workflow.")


class AnvilSection(SpecBase):
    """
    Anvil specification section base class.

    Attributes
    ----------
    type : Optional[str]
        The type of the section.
    params : dict
        The parameters for the section.
    section_name : ClassVar[str]
        The name of the section.

    """

    type: str | None = None
    params: dict = {}
    section_name: ClassVar[str] = "INVALID"

    def to_class(self):
        """
        Convert the specification to the corresponding class instance.

        Returns
        -------
        instance : object
            An instance of the class corresponding to the section type.

        """
        return _SECTION_CLASS_GETTERS[self.section_name](self.type)(**self.params)


class SplitSpec(AnvilSection):
    """Data split specification."""

    section_name: ClassVar[str] = "split"


class FeatureSpec(AnvilSection):
    """Featurization specification."""

    section_name: ClassVar[str] = "feat"


class ModelSpec(AnvilSection):
    """
    Model specification.

    Attributes
    ----------
    section_name : ClassVar[str]
        The name of the section.
    param_path : Optional[str]
        The path to the model parameters file.
    serial_path : Optional[str]
        The path to the model serialization file.
    freeze_weights : Optional[dict]
        A dictionary specifying which layers to freeze during training.

    """

    section_name: ClassVar[str] = "model"
    param_path: str | None = None
    serial_path: str | None = None
    freeze_weights: dict | None = None

    @model_validator(mode="after")
    def check_paths(self):
        """
        Ensure both param_path and serial_path are provided together.

        Returns
        -------
        self : ModelSpec
            The validated ModelSpec instance.

        """
        # Both specified
        if self.param_path and self.serial_path:
            return self

        # Neither specified
        if not self.param_path and not self.serial_path:
            return self

        raise ValueError(
            "Both `param_path` and `serial_path` must be provided together."
        )

    @model_validator(mode="after")
    def check_freeze_weights(self):
        """
        Ensure freeze weights is supplied for only applicable model types.

        Returns
        -------
        self : ModelSpec
            The validated ModelSpec instance.

        """
        # Check if weight freezing selected
        if self.freeze_weights:
            # Attempt freezing model weights
            try:
                model = self.to_class()
                model.build()
                model.freeze_weights()

            # Raise error here if not implemented
            except NotImplementedError:
                raise ValueError(f"Weight freezing not implemented for {self.type}.")

        return self


class EnsembleSpec(AnvilSection):
    """
    Ensemble specification.

    Attributes
    ----------
    section_name : ClassVar[str]
        The name of the section.
    n_models : int
        The number of models in the ensemble.
    calibration_method : str
        The calibration method to use.
    param_paths : Optional[list[str]]
        The list of parameter file paths for the ensemble models.
    serial_paths : Optional[list[str]]
        The list of serialization file paths for the ensemble models.

    """

    section_name: ClassVar[str] = "ensemble"
    n_models: int
    calibration_method: str | None = "isotonic-regression"
    param_paths: list[str] | None = None
    serial_paths: list[str] | None = None

    @field_validator("calibration_method")
    def check_method(cls, value):
        """Validate the calibration method."""
        allowed = ["isotonic-regression", "scaling-factor", None]
        if value not in allowed:
            raise ValueError(
                f"Invalid calibration method: {value}. Valid options are: {allowed}."
            )
        return value

    @field_validator("n_models")
    def check_n_models(cls, value):
        """Ensure ensemble has more than one model."""
        if value < 2:
            raise ValueError("Ensemble must have more than one model.")
        return value

    @model_validator(mode="after")
    def check_paths(self):
        """Ensure both param_paths and serial_paths are provided together."""
        # Both specified
        if self.param_paths and self.serial_paths:
            # Check lengths match
            if len(self.param_paths) != len(self.serial_paths):
                raise ValueError(
                    "Parameter and serial paths must have the same length."
                )

            # Check matches model count
            if len(self.param_paths) != self.n_models:
                raise ValueError(
                    f"Number of parameter ({len(self.param_paths)}) and serial paths ({len(self.serial_paths)}) must "
                    f"match the number of models ({self.n_models})."
                )

            return self

        # Neither specified
        if not self.param_paths and not self.serial_paths:
            return self

        raise ValueError(
            "Both `param_paths` and `serial_paths` must be provided together."
        )


class TrainerSpec(AnvilSection):
    """Trainer specification."""

    section_name: ClassVar[str] = "train"


class EvalSpec(AnvilSection):
    """Evaluation specification."""

    section_name: ClassVar[str] = "eval"


class TransformSpec(AnvilSection):
    """Transform specification."""

    section_name: ClassVar[str] = "transform"


class ProcedureSpec(SpecBase):
    """Procedure specification."""

    section_name: ClassVar[str] = "procedure"

    split: SplitSpec
    feat: FeatureSpec
    model: ModelSpec
    ensemble: EnsembleSpec | None = None
    train: TrainerSpec
    transform: Optional[TransformSpec] = None  # Optional transform step


class ReportSpec(SpecBase):
    """Report specification."""

    section_name: ClassVar[str] = "report"
    eval: list[EvalSpec]


class AnvilSpecification(BaseModel):
    """Full specification for Anvil workflow."""

    metadata: Metadata
    data: DataSpec
    procedure: ProcedureSpec
    report: ReportSpec

    @classmethod
    def from_recipe(cls, yaml_path: PathLike, **storage_options):
        """Load specification from YAML recipe file."""
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
        """Write specification to YAML recipe file."""
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
        """Load specification from multiple YAML files."""
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

        Parameters
        ----------
        metadata_yaml : str or PathLike, optional
            The file path for the metadata YAML file. Default is 'metadata.yaml'.
        procedure_yaml : str or PathLike, optional
            The file path for the procedure YAML file. Default is 'procedure.yaml'.
        data_yaml : str or PathLike, optional
            The file path for the data YAML file. Default is 'data.yaml'.
        report_yaml : str or PathLike, optional
            The file path for the report YAML file. Default is 'eval.yaml'.
        storage_options : dict, optional
            Additional options to pass to the file system (e.g., for S3, GCS)

        """
        # Write each section to its own YAML file
        self.metadata.to_yaml(metadata_yaml, **storage_options)
        self.data.to_yaml(data_yaml, **storage_options)
        self.procedure.to_yaml(procedure_yaml, **storage_options)
        self.report.to_yaml(report_yaml, **storage_options)

    def to_workflow(self):
        """Convert the specification to a workflow object."""
        logger.info("Making workflow from specification")

        # Import here to avoid circular import
        from openadmet.models.anvil.workflow import _DRIVER_TO_CLASS

        return _DRIVER_TO_CLASS[self.metadata.driver](
            metadata=self.metadata,
            data_spec=self.data,
            model=self.procedure.model.to_class(),
            ensemble=self.procedure.ensemble.to_class()
            if self.procedure.ensemble
            else None,
            transform=self.procedure.transform.to_class()
            if self.procedure.transform
            else None,
            split=self.procedure.split.to_class(),
            feat=self.procedure.feat.to_class(),
            trainer=self.procedure.train.to_class(),
            evals=[eval.to_class() for eval in self.report.eval],
            parent_spec=self,
        )
