import click

from openadmet.models.inference.inference import predict as inference_func


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
    "--output-csv",
    help="Path to the output CSV file for predictions",
    default="predictions.csv",
    show_default=True,
    required=True,
    type=click.Path(exists=False, writable=True),
)
@click.option(
    "--accelerator",
    help="One of either cpu or gpu, default is gpu",
    required=False,
    default="gpu",
    type=str,
)
@click.option("--debug", is_flag=True, help="Enable debug mode", default=False)
def predict(input_path, input_col, model_dir, output_csv, debug, accelerator):
    """Predict using a trained model"""
    inference_func(
        input_path=input_path,
        input_col=input_col,
        model_dir=model_dir,
        write_csv=True,
        output_csv=output_csv,
        debug=debug,
        accelerator=accelerator,
    )
