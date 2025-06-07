from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rdkit.Chem import PandasTools

from openadmet.models.anvil.workflow import Metadata, ProcedureSpec


def load_anvil_model_and_metadata(model_dir):
    """Load the Anvil model from the specified path"""
    # load from recipe directory
    recipe_components_dir = model_dir / "recipe_components"
    if not recipe_components_dir.exists():
        raise FileNotFoundError(
            f"Model path {model_dir} does not contain recipe components"
        )
    # load the specification
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
    # load the metadata
    metadata = Metadata.from_yaml(metadata_spec)

    # load the procedure specification
    procedure_spec = ProcedureSpec.from_yaml(procedure_spec)
    feat = procedure_spec.feat.to_class()
    model = procedure_spec.model.to_class()

    # deserialize the model
    loaded_model = model.deserialize(
        param_path=model_dir / model._model_json_name,
        serial_path=model_dir / model._model_save_name,
    )

    return loaded_model, feat, metadata


@click.command()
@click.option(
    "--input-path",
    help="Path to the input CSV file or SDF containing structures",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--input-col",
    help="Column name in the CSV file containing input structure or OPENADMET_SMILES",
    default="OPENADMET_SMILES",
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Path to a trained model directory as trained by `openadmet anvil`",
    required=True,
    type=click.Path(exists=True),
    multiple=True,
)
@click.option(
    "--output-path",
    help="Path to the output CSV file for predictions",
    default="predictions.csv",
    show_default=True,
    required=True,
    type=click.Path(exists=False, writable=True),
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
def predict(input_path, input_col, model_dir, output_path, debug):
    """Predict using a trained model"""
    logger.info("Starting prediction")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Model directories: {model_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Input column: {input_col}")
    # load input data
    if input_path.endswith(".csv"):
        data = pd.read_csv(input_path)
    elif input_path.endswith(".sdf"):
        data = PandasTools.LoadSDF(input_path, smilesName=input_col)

    if input_col not in data.columns:
        raise ValueError(f"Column {input_col} not found in input data")

    if "predictions" in data.columns:
        raise ValueError("Output file already contains a 'predictions' column")

    # Load the models
    for i, model_path in enumerate(model_dir):
        logger.info(f"Loading model {i} from {model_path}")
        # load the model and metadata
        model, feat, metadata = load_anvil_model_and_metadata(Path(model_path))

        logger.debug("Model metadata:")
        logger.debug(metadata)
        logger.debug(f"Model: {model.estimator}")
        logger.debug(f"Feature: {feat}")
        X_feat, _ = feat.featurize(data[input_col])

        predictions = model.predict(X_feat)

        # will need to change for multi-target models
        predictions_tag = f"OADMET_PRED_{metadata.tag}"
        if predictions_tag in data.columns:
            raise ValueError(
                f"Output file already contains a '{predictions_tag}' column"
            )

        data[predictions_tag] = predictions

    logger.info("Finished prediction")
    logger.info(f"Predictions saved to {output_path}")
    # remove ROMol column if it exists
    if "ROMol" in data.columns:
        data.drop(columns=["ROMol"], inplace=True)
    # remove ID column if it exists
    if "ID" in data.columns:
        data.drop(columns=["ID"], inplace=True)
    data.to_csv(output_path, index=False)
