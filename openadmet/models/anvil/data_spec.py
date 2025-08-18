from pathlib import Path
from typing import Optional, Union

import fsspec
import intake
import jinja2
import pandas as pd
import yaml
from pydantic import BaseModel, model_validator, field_validator
from loguru import logger

class DataSpec(BaseModel):
    """
    Data specification for the workflow
    """

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
        if isinstance(v, str):
            return [v]
        else:
            return v

    # validator to template the resource with ANVIL_DIR if present
    @model_validator(mode="after")
    def template_resource(self):
        if self.anvil_dir:
            template = jinja2.Template(self.resource)
            self.resource = template.render(ANVIL_DIR=self.anvil_dir)
        return self

    def template_anvil_dir(self, anvil_dir: Path):
        """
        Template the resource with ANVIL_DIR if present
        """
        self.anvil_dir = anvil_dir
        template = jinja2.Template(self.resource)
        self.resource = template.render(ANVIL_DIR=anvil_dir)

    def read(self) -> tuple[pd.Series, pd.Series]:
        """
        Read the data from the resource
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
        combined = data[[self.input_col] + (self.target_cols if isinstance(self.target_cols, list) else [self.target_cols])]

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
        return self._catalog

    def to_yaml(self, path, **storage_options):
        with fsspec.open(path, "w", **storage_options) as stream:
            yaml.safe_dump(self.model_dump(), stream)

    @classmethod
    def from_yaml(cls, path, **storage_options):
        of = fsspec.open(path, "r", **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)
        return cls(**data)
