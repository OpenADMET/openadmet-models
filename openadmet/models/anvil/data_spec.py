from pathlib import Path
from typing import Optional

import fsspec
import intake
import jinja2
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel, model_validator


class DataSpec(BaseModel):
    """
    Data specification for the workflow
    """

    type: str
    resource: str
    cat_entry: Optional[str] = None
    target_cols: list[str]
    input_col: str
    anvil_dir: Optional[str] = None

    _catalog: Optional[intake.catalog.Catalog] = None

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

        elif self.resource.endswith(".parquet") or self.resource.endswith(".parq") or self.resource.endswith(".pq") or self.resource.endswith(".pqt"):
            data = intake.open_parquet(self.resource).read()
        else:
            raise ValueError(f"Unsupported resource type: {self.resource}")
        # now read the target columns and smiles column
        targets = data[self.target_cols]
        input = data[self.input_col]

        return input, targets

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
