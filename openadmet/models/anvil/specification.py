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
    """Data specification for the workflow."""

    type: str
    resource: str
    cat_entry: Optional[str] = None
    target_cols: Union[str, list[str]]
    input_col: str
    anvil_dir: Optional[str] = None
    dropna: Optional[bool] = False

    _catalog: Optional[intake.catalog.Catalog] = None

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
            template = jinja2.Template(self.resource)
            self.resource = template.render(ANVIL_DIR=self.anvil_dir)
        return self

    def template_anvil_dir(self, anvil_dir: Path):
        """Template the resource with ANVIL_DIR if present."""
        self.anvil_dir = anvil_dir
        template = jinja2.Template(self.resource)
        self.resource = template.render(ANVIL_DIR=anvil_dir)

    def read(self) -> tuple[pd.Series, pd.Series]:
        """
        Read the data from the resource.

        Returns
        -------
        input: pd.Series
            The input data (e.g., SMILES strings)
        targets: pd.Series
            The target data (e.g., properties to predict)

        Raises
        ------
        ValueError
            If the resource type is not supported

        """
        # if YAML, parse as intake catalog
        if self.resource.endswith(".yaml") or self.resource.endswith(".yml"):
            self._catalog = intake.open_catalog(self.resource)
            data = self._catalog[self.cat_entry].read()

        # if CSV, parse using intake
        elif self.resource.endswith(".csv"):
            data = intake.open_csv(self.resource).read()

        elif any(self.resource.endswith(x) for x in [".parquet", ".pq", ".pqt"]):
            data = intake.open_parquet(self.resource).read()
        else:
            raise ValueError(f"Unsupported resource type: {self.resource}")

        # combine input and targets for joint NaN handling
        combined = data[
            [self.input_col]
            + (
                self.target_cols
                if isinstance(self.target_cols, list)
                else [self.target_cols]
            )
        ]

        # get number of combined rows for logging
        n_before = len(combined)

        if self.dropna:
            cleaned_combined = combined.dropna().reset_index(drop=True)
            n_after = len(cleaned_combined)
            n_dropped = n_before - n_after
        else:
            n_dropped = 0
            cleaned_combined = combined

        # Split the data again
        input_clean = cleaned_combined[self.input_col]
        targets_clean = cleaned_combined[self.target_cols]

        logger.info(f"{n_before} total rows. {n_dropped} NaN rows were dropped.")

        return input_clean, targets_clean

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

        Returns
        -------
        None

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

        Returns
        -------
        None

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
            Additional options to pass to the file system (e.g., for S3, GCS

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
    """Metadata specification."""

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
    """Anvil specification section base class."""

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

        Raises
        ------
        ValueError
            If the section_name is invalid or the type is not recognized.

        """
        return _SECTION_CLASS_GETTERS[self.section_name](self.type)(**self.params)


class SplitSpec(AnvilSection):
    """Data split specification."""

    section_name: ClassVar[str] = "split"


class FeatureSpec(AnvilSection):
    """Featurization specification."""

    section_name: ClassVar[str] = "feat"


class ModelSpec(AnvilSection):
    """Model specification."""

    section_name: ClassVar[str] = "model"
    param_path: str | None = None
    serial_path: str | None = None

    @model_validator(mode="after")
    def check_paths(self):
        """
        Ensure both param_path and serial_path are provided together.

        Returns
        -------
        self : ModelSpec
            The validated ModelSpec instance.

        Raises
        ------
        ValueError
            If only one of param_path or serial_path is provided.

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


class EnsembleSpec(AnvilSection):
    """Ensemble specification."""

    section_name: ClassVar[str] = "ensemble"
    n_models: int
    param_paths: list[str] | None = None
    serial_paths: list[str] | None = None

    @field_validator("n_models")
    def check_n_models(cls, value):
        """
        Ensure ensemble has more than one model.

        Parameters
        ----------
        value : int
            The number of models in the ensemble.

        Returns
        -------
        value : int
            The validated number of models.

        """
        if value < 2:
            raise ValueError("Ensemble must have more than one model.")
        return value

    @model_validator(mode="after")
    def check_paths(self):
        """
        Ensure both param_paths and serial_paths are provided together.

        Returns
        -------
        self : EnsembleSpec
            The validated EnsembleSpec instance.

        Raises
        ------
        ValueError
            If only one of param_paths or serial_paths is provided, or if their lengths do not
            match the number of models.

        """
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
            Additional options to pass to the file system (e.g., for S3, GCS

        Returns
        -------
        None

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
