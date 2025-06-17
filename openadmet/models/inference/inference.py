from pathlib import Path

import pandas as pd
from loguru import logger
from rdkit.Chem import PandasTools
from openadmet.models.anvil.workflow import Metadata, ProcedureSpec
from openadmet.models.anvil.data_spec import DataSpec

from typing import Union, List

def load_anvil_model_and_metadata(model_dir):
    """Load the Anvil model from the specified path"""
    logger.info(f"Loading model from {model_dir}")
    # Load from recipe directory
    recipe_components_dir = model_dir / "recipe_components"
    if not recipe_components_dir.exists():
        raise FileNotFoundError(
            f"Model path {model_dir} does not contain recipe components"
        )
    # Load the specification
    procedure_spec = recipe_components_dir / "procedure.yaml"
    if not procedure_spec.exists():
        raise FileNotFoundError(
            f"Model path {model_dir} does not contain procedure.yaml"
        )

    metadata_spec = recipe_components_dir / "metadata.yaml"
    if not metadata_spec.exists():
        raise FileNotFoundError(
            f"Model path {model_dir} does not contain metadata.yaml"
        )
    # Load the metadata
    metadata = Metadata.from_yaml(metadata_spec)


    data_spec = recipe_components_dir / "data.yaml"
    if not data_spec.exists():
        raise FileNotFoundError(
            f"Model path {model_dir} does not contain data.yaml"
        )
    # Load the data specification
    data = DataSpec.from_yaml(data_spec)

    # Load the procedure specification
    procedure_spec = ProcedureSpec.from_yaml(procedure_spec)
    feat = procedure_spec.feat.to_class()
    model = procedure_spec.model.to_class()


    # Deserialize the model
    loaded_model = model.deserialize(
        param_path=model_dir / model._model_json_name,
        serial_path=model_dir / model._model_save_name,
    )

    return loaded_model, feat, metadata, data



def predict(
    input_path: str,
    input_col: str,
    model_dir: Union[str, Path, list[Union[str, Path]]],
    write_csv: bool = False,
    output_path: str = None,
    debug: bool = False,
    accelerator: str = "gpu",
    log: bool = True,
    **kwargs
):
    """Predict using a trained model"""


    if not log:
        logger.remove()
        logger.add(lambda msg: None)

    logger.info("Starting prediction")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Model directories: {model_dir}")
    logger.info(f"Write CSV: {write_csv}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Input column: {input_col}")
    # load input data
    if isinstance(input_path, pd.DataFrame):
        data = input_path

    elif isinstance(input_path, Path) or isinstance(input_path, str):
        if input_path.endswith(".csv"):
            data = pd.read_csv(input_path)
        elif input_path.endswith(".sdf"):
            data = PandasTools.LoadSDF(input_path, smilesName=input_col)
    else:
        raise ValueError(
            "Input path must be a pandas DataFrame, a CSV file, or an SDF file"
        )

    if input_col not in data.columns:
        raise ValueError(f"Column {input_col} not found in input data")

    # check if model dir is a list or a single path
    if isinstance(model_dir, (str, Path)):
        logger.debug(f"Model directory is a single path: {model_dir}")
        model_dir = [model_dir]



    logger.info(f"Model directories: {model_dir}")

    # Mute output from FutureWarning and DeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Load the models
    for i, model_path in enumerate(model_dir):
        logger.info(f"Loading model {i} from {model_path}")

        # Load the model and metadata
        model, feat, metadata, data_spec = load_anvil_model_and_metadata(Path(model_path))

        tasknames = data_spec.target_cols
        logger.info(f"Model {i} has tasks: {tasknames}")

        logger.debug("Model metadata:")
        logger.debug(metadata)
        logger.debug(f"Model: {model.estimator}")
        logger.debug(f"Feature: {feat}")
        # Returns a variable length tuple, first element is the featurized data or a dataloader
        # Second element is the indices of the original input that were featurized
        feat_data = feat.featurize(data[input_col])
        # Features or dataloader
        X_feat = feat_data[0]

        # Indices of the original input that were featurized
        X_indices = feat_data[1]
        # Make the actual model predictions
        predictions = model.predict(X_feat, accelerator=accelerator)

        for j, taskname in enumerate(tasknames):
            predictions_tag = f"OADMET_PRED_{metadata.tag}_{taskname}"
            if predictions_tag in data.columns:
                raise ValueError(
                    f"Output file already contains a '{predictions_tag}' column"
                )

            # Add the predictions to the data DataFrame
            data[predictions_tag] = pd.Series(predictions[:,j], index=X_indices)
            logger.info(f"Predictions for model {i} task {j} saved to column '{predictions_tag}'")

    logger.info("Finished prediction")
    logger.info(f"Predictions saved to {output_path}")
    # remove ROMol column if it exists
    if "ROMol" in data.columns:
        data.drop(columns=["ROMol"], inplace=True)
    # remove ID column if it exists
    if "ID" in data.columns:
        data.drop(columns=["ID"], inplace=True)

    if write_csv:
        data.to_csv(output_path, index=False)

    return data
