import pandas as pd
from rdkit.Chem import PandasTools
from openadmet.models.anvil.workflow import ProcedureSpec, Metadata
from pathlib import Path
from loguru import logger
from openadmet.models.cli.predict import load_anvil_model_and_metadata

def predict(input_path:str,
            input_col:str,
            model_dir:str,
            write_csv:bool = False,
            output_path:str = None,
            debug:bool = False,
            accelerator: str = "gpu"):
    """Predict using a trained model"""
    logger.info("Starting prediction")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Model directories: {model_dir}")
    logger.info(f"Write CSV: {write_csv}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Input column: {input_col}")
    # load input data
    if input_path.endswith(".csv"):
        data = pd.read_csv(input_path)
    elif input_path.endswith(".sdf"):
        data = PandasTools.LoadSDF(input_path, smilesName=input_col)

    if input_col not in data.columns:
        raise ValueError(f"Column {input_col} not found in input data")

    # Load the models
    for i, model_path in enumerate(model_dir):
        logger.info(f"Loading model {i} from {model_path}")
        # load the model and metadata
        model, feat, metadata = load_anvil_model_and_metadata(Path(model_path))

        logger.debug("Model metadata:")
        logger.debug(metadata)
        logger.debug(f"Model: {model.estimator}")
        logger.debug(f"Feature: {feat}")
        # returns a variable length tuple, first element is the featurized data or a dataloader
        feat_data  = feat.featurize(data[input_col])
        X_feat = feat_data[0]
        # make the actual model predictions
        predictions = model.predict(X_feat, accelerator=accelerator)

        # will need to change for multi-target models
        predictions_tag = f"OADMET_PRED_{metadata.tag}"
        if predictions_tag in data.columns:
            raise ValueError(f"Output file already contains a '{predictions_tag}' column")

        data[predictions_tag] = predictions

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
