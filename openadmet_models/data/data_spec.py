from enum import Enum
from typing import Optional, Tuple

import intake
import jinja2
import pandas as pd
from pydantic import BaseModel

from openadmet_models.util.types import Pathy


class DataSpecTypes(str, Enum):
    """
    Types of data specifications
    """

    INTAKE = "intake"


class DataSpec(BaseModel):
    """
    Data specification for the workflow
    """

    type: DataSpecTypes
    resource: str
    cat_entry: Optional[str] = None
    target_col: str
    smiles_col: str
    anvil_dir: Pathy = None

    def read(self) -> tuple[pd.Series, pd.Series]:
        """
        Read the data from the resource
        """

        # if YAML, parse as intake catalog
        if self.resource.endswith(".yaml") or self.resource.endswith(".yml"):
            # template the resource with ANVIL_DIR if present
            if self.anvil_dir:
                template = jinja2.Template(self.resource)
                self.resource = template.render(ANVIL_DIR=self.anvil_dir)
                print(self.resource)

            catalog = intake.open_catalog(self.resource)
            data = catalog[self.cat_entry].read()

        # if CSV, parse using intake
        elif self.resource.endswith(".csv"):
            data = intake.open_csv(self.resource).read()

        # now read the target columns and smiles column
        target = data[self.target_col]
        smiles = data[self.smiles_col]

        return smiles, target
