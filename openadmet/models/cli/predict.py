import click

from openadmet.models.inference.inference import predict as inference_func


@click.command()
@click.option(
    "--input-path",
    help="Path to the input CSV file or SDF containing structures.",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--input-col",
    help="Column name in the CSV file containing input structure or OPENADMET_SMILES.",
    default="OPENADMET_SMILES",
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Path to a trained model directory as trained by `openadmet anvil`.",
    required=True,
    type=click.Path(exists=True),
    multiple=True,
)
@click.option(
    "--output-csv",
    help="Path to the output CSV file for predictions.",
    default="predictions.csv",
    show_default=True,
    required=True,
    type=click.Path(exists=False, writable=True),
)
@click.option(
    "--accelerator",
    help="Hardware to use for inference.",
    required=False,
    default="gpu",
    type=click.Choice(
        ["cpu", "gpu", "tpu", "ipu", "mps", "auto"], case_sensitive=False
    ),
    show_default=True,
)
@click.option(
    "--aq-fxn",
    "aq_fxns",
    help="Acquisition function(s) to use for active learning.",
    type=click.Choice(["ucb", "ei", "pi"], case_sensitive=False),
    multiple=True,
)
@click.option("--beta", type=float, multiple=True)
@click.option("--best-y", type=float, multiple=True)
@click.option("--xi", type=float, multiple=True)
@click.option("--debug", is_flag=True, help="Enable debug mode", default=False)
def predict(
    input_path,
    input_col,
    model_dir,
    output_csv,
    accelerator,
    aq_fxns,
    beta,
    best_y,
    xi,
    debug,
):
    aq_fxn_args = _validate_aq_fxns(aq_fxns, beta, best_y, xi)
    """Predict using a trained model"""
    inference_func(
        input_path=input_path,
        input_col=input_col,
        model_dir=model_dir,
        write_csv=True,
        output_csv=output_csv,
        debug=debug,
        accelerator=accelerator,
        aq_fxn_args=aq_fxn_args,
    )


def _validate_aq_fxns(aq_fxns, beta, best_y, xi):
    # Tuple to list
    aq_fxns = list(aq_fxns)
    beta = list(beta)
    best_y = list(best_y)
    xi = list(xi)

    # Check that each allowed acquisition function is specified at most once
    if aq_fxns.count("ucb") > 1:
        raise ValueError("UCB can only be specified once.")
    if aq_fxns.count("ei") > 1:
        raise ValueError("EI can only be specified once.")
    if aq_fxns.count("pi") > 1:
        raise ValueError("PI can only be specified once.")

    # Check that `beta` is supplied the same number of times as ucb
    if aq_fxns.count("ucb") != len(beta):
        raise ValueError("Field `beta` must be specified for UCB acquisition.")

    # Check that `best_y` and `xi` are supplied the same number of times as ei and pi combined
    best_y_xi_count = aq_fxns.count("ei") + aq_fxns.count("pi")
    if best_y_xi_count != len(best_y) or best_y_xi_count != len(xi):
        raise ValueError(
            "Fields `best_y` and `xi` must be specified once per EI and/or PI acquisition."
        )

    # Parse acquisition function arguments
    result = {}
    for aq_fxn in aq_fxns:
        if aq_fxn == "ucb":
            result[aq_fxn] = {"beta": beta.pop(0)}
        elif aq_fxn in ["ei", "pi"]:
            result[aq_fxn] = {"xi": xi.pop(0), "best_y": best_y.pop(0)}

    return result
