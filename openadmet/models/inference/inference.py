"""Inference functions for trained models."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from rdkit.Chem import PandasTools

from openadmet.models.active_learning.acquisition import _ACQUISITION_FUNCTIONS
from openadmet.models.active_learning.ensemble_base import EnsembleBase
from openadmet.models.anvil.specification import DataSpec, Metadata, ProcedureSpec
from openadmet.models.features.pairwise import (
    PairwiseAugmentedDataset,
    PairwiseFeaturizer,
)


def load_anvil_model_and_metadata(model_dir):
    """
    Load the Anvil model from the specified path.

    Parameters
    ----------
    model_dir : Union[str, Path]
        Path to the directory containing the trained model and its metadata.

    Returns
    -------
    tuple
        A tuple containing the loaded model, feature object, metadata, and data specification.

    """
    # Safely cast to Path
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

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
        raise FileNotFoundError(f"Model path {model_dir} does not contain data.yaml")
    # Load the data specification
    data = DataSpec.from_yaml(data_spec)

    # Load the procedure specification
    procedure_spec = ProcedureSpec.from_yaml(procedure_spec)
    feat = procedure_spec.feat.to_class()
    model = procedure_spec.model.to_class()

    # Load model ensemble
    if procedure_spec.ensemble is not None:
        # Get ensemble class
        ensemble = procedure_spec.ensemble.to_class()

        # Deserialize the model ensemble
        loaded_model = ensemble.deserialize(
            param_paths=list(model_dir.glob(f"*/{model._model_json_name}")),
            serial_paths=list(model_dir.glob(f"*/{model._model_save_name}")),
            mod_class=model,
            calibration_path=model_dir / ensemble._calibration_model_save_name,
        )

    # Load single model
    else:
        # Deserialize the model
        loaded_model = model.deserialize(
            param_path=model_dir / model._model_json_name,
            serial_path=model_dir / model._model_save_name,
        )

    return loaded_model, feat, metadata, data


def _generate_pairwise_df(
    data, input_col, feat, predictions, predictions_tag, std_tag
) -> pd.DataFrame:
    """Generate a DataFrame for pairwise predictions."""
    smiles = data[input_col].values
    pairwise_dataset = PairwiseAugmentedDataset(smiles, None, how=feat.how_to_pair)
    pairs = pairwise_dataset.idxs  # list of (i, j) tuples

    smiles_i = [smiles[i] for i, j in pairs]
    smiles_j = [smiles[j] for i, j in pairs]
    pred = predictions[:, j]

    pairwise_df = pd.DataFrame(
        {
            f"{input_col}_i": smiles_i,
            f"{input_col}_j": smiles_j,
            predictions_tag: pred,
        }
    )

    pairwise_df[std_tag] = pd.Series(predictions[:, j], index=pairwise_df.index)

    pairwise_df[input_col] = (
        pairwise_df[f"{input_col}_i"] + " - " + pairwise_df[f"{input_col}_j"]
    )

    return pairwise_df


def predict(
    input_path: str,
    input_col: str,
    model_dir: Union[str, Path, list[Union[str, Path]]],
    write_csv: bool = False,
    output_csv: str = None,
    debug: bool = False,
    accelerator: str = "gpu",
    log: bool = True,
    aq_fxn_args: dict | None = None,
    **kwargs,
):
    """
    Predict using a trained model.

    Parameters
    ----------
    input_path : Union[str, Path, pd.DataFrame]
        Path to the input data file (CSV or SDF or parquet) or a pandas DataFrame.
    input_col : str
        Name of the column containing SMILES strings.
    model_dir : Union[str, Path, list[Union[str, Path]]]
        Path(s) to the directory(ies) containing the trained model(s) and their metadata.
    write_csv : bool, optional
        Whether to write the output to a CSV file. Default is False.
    output_csv : str, optional
        Path to the output CSV file. If None, defaults to 'predictions.csv' in
        the current directory. Default is None.
    debug : bool, optional
        Whether to enable debug logging. Default is False.
    accelerator : str, optional
        Accelerator to use for prediction ('cpu' or 'gpu'). Default is 'gpu'.
    log : bool, optional
        Whether to enable logging. Default is True.
    aq_fxn_args : dict, optional
        Dictionary of acquisition function names and their arguments to compute
        additional metrics. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the input data along with prediction results.

    """
    if not log:
        logger.remove()
        logger.add(lambda msg: None)

    logger.info("Starting prediction")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Model directories: {model_dir}")
    logger.info(f"Write CSV: {write_csv}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Input column: {input_col}")
    logger.info(f"Accelerator: {accelerator}")
    # load input data
    if isinstance(input_path, pd.DataFrame):
        data = input_path
    elif isinstance(input_path, Path) or isinstance(input_path, str):
        if input_path.endswith(".csv"):
            data = pd.read_csv(input_path)
        elif input_path.endswith(".sdf"):
            data = PandasTools.LoadSDF(input_path, smilesName=input_col)
        elif input_path.endswith(".parquet"):
            data = pd.read_parquet(input_path).reset_index(drop=True)
        else:
            raise ValueError("Path must lead to a CSV or SDF or parquet file")
    else:
        raise ValueError(
            "Input path must be a pandas DataFrame, a CSV file, a parquet file, or an SDF file"
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
        model, feat, metadata, data_spec = load_anvil_model_and_metadata(
            Path(model_path)
        )

        if isinstance(model, EnsembleBase):
            logger.info(
                f"Loaded model ensemble {i} from {model_path}, with {model.n_models} submodels"
            )
        else:
            logger.info(f"Loaded model {i} from {model_path}")

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

        # if ensemble, return std as well
        if isinstance(model, EnsembleBase):
            predictions, std = model.predict(
                X_feat, accelerator=accelerator, return_std=True
            )
        else:
            predictions = model.predict(X_feat, accelerator=accelerator)
            std = np.full(predictions.shape, np.nan)

        for j, taskname in enumerate(tasknames):
            predictions_tag = f"OADMET_PRED_{metadata.tag}_{taskname}"
            std_tag = f"OADMET_STD_{metadata.tag}_{taskname}"

            if predictions_tag in data.columns or std_tag in data.columns:
                raise ValueError(
                    f"Output file already contains a '{predictions_tag}' column or '{std_tag}' column"
                )

            if isinstance(feat, PairwiseFeaturizer):
                logger.info(
                    "Detected pairwise featurizer, generating pairwise output DataFrame"
                )
                data = _generate_pairwise_df(
                    data, input_col, feat, predictions, predictions_tag, std_tag
                )

            else:
                # Add the predictions to the data DataFrame
                data[predictions_tag] = pd.Series(predictions[:, j], index=X_indices)
                data[std_tag] = pd.Series(std[:, j], index=X_indices)

            logger.info(
                f"Predictions for model {i} task {j} saved to column '{predictions_tag}', std saved to column '{std_tag}'"
            )

            # Add acquisition function results
            if aq_fxn_args is not None:
                for aq_fxn, aq_args in aq_fxn_args.items():
                    aq_tag = f"OADMET_{aq_fxn.upper()}_{metadata.tag}_{taskname}"
                    aq_result = _ACQUISITION_FUNCTIONS[aq_fxn](
                        predictions[:, j], std[:, j], **aq_args
                    )
                    data[aq_tag] = pd.Series(aq_result, index=X_indices)
                    logger.info(
                        f"Acquisition function '{aq_fxn}' for model {i} task {j} saved to column '{aq_tag}'"
                    )

    logger.info("Finished prediction")
    logger.info(f"Predictions saved to {output_csv}")
    # remove ROMol column if it exists
    if "ROMol" in data.columns:
        data.drop(columns=["ROMol"], inplace=True)
    # remove ID column if it exists
    if "ID" in data.columns:
        data.drop(columns=["ID"], inplace=True)

    if write_csv:
        data.to_csv(output_csv, index=False)

    return data
