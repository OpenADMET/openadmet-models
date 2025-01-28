from pydantic import BaseModel
from typing import Optional, Tuple
import pandas as pd
import intake
from enum import Enum

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
    target_cols: str
    smiles_col: str

    def read(self) -> Tuple[pd.Series, pd.Series]:
        """
        Read the data from the resource
        """
        # if YAML, parse as intake catalog
        if self.resource.endswith('.yaml') or self.resource.endswith('.yml'):
            catalog = intake.open_catalog(self.resource)
            data = catalog[self.cat_entry].read()

        # if CSV, parse using intake
        elif self.resource.endswith('.csv'):
            data = intake.open_csv(self.resource).read()

        print(data)
        # now read the target columns and smiles column
        target = data[self.target_cols]
        smiles = data[self.smiles_col]

        return target, smiles