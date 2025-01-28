import intake
from pydantic import BaseModel


def read_intake_data(resource: str, cat_entry: str,  target_cols: str, smiles_col: str) -> BaseModel:
    # if YAML, parse as intake catalog

    if resource_name.endswith('.yaml') or resource_name.endswith('.yml'):
        catalog = intake.open_catalog(resource_name)
        data = catalog[cat_entry].read()

    # if CSV, parse using intake
    elif resource_name.endswith('.csv'):
        data = intake.open_csv(resource_name).read()

    # now read the target columns and smiles column
    target = data[target_cols]
    smiles = data[smiles_col]

    return target, smiles
    
     


