import click

from openadmet.models.active_learning.acquisition import _QUERY_STRATEGIES
from openadmet.models.inference.inference import predict as inference_func


class AcquisitionGroup:
    """Tracks acquisition function and its arguments."""

    def __init__(self, name):
        self.name = name
        self.params = {}

    def __repr__(self):
        return f"{self.name}({self.params})"


class AcquisitionParser(click.Option):
    def __init__(self, *args, **kwargs):
        self.seen_aqs = []
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        # Reset global state for each parse
        ctx.ensure_object(dict)
        ctx.obj.setdefault("aq_groups", [])

        aq_groups = ctx.obj["aq_groups"]

        # Handle "--aq-fxn" specially
        if self.name == "aq-fxn":
            for aq in opts.get("aq-fxn", []):
                aq_groups.append(AcquisitionGroup(aq))
            opts.pop("aq-fxn", None)

        # Handle parameters like beta, best-y, xi
        for param in ("beta", "best_y", "xi"):
            if param in opts:
                values = opts.pop(param)
                for val in values:
                    if not aq_groups:
                        raise click.UsageError(f"--{param} must follow an --aq option")
                    aq_groups[-1].params[param] = val

        return super().handle_parse_result(ctx, opts, args)


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
    help="One of either cpu or gpu",
    required=False,
    default="gpu",
    type=click.Choice(["gpu", "cpu", "mls"], case_sensitive=False),
    show_default=True,
)
# Acquisition function arguments
@click.option(
    "--aq-fxn",
    type=click.Choice(_QUERY_STRATEGIES.keys(), case_sensitive=False),
    multiple=True,
    cls=AcquisitionParser,
    help="Acquisition function",
)
@click.option(
    "--beta",
    type=float,
    multiple=True,
    cls=AcquisitionParser,
    help="Tradeoff parameter (higher = more exploration). Used for UCB.",
)
@click.option(
    "--best-y",
    type=float,
    multiple=True,
    cls=AcquisitionParser,
    help="Best observed value so far. Used for EI, PI.",
)
@click.option(
    "--xi",
    type=float,
    multiple=True,
    cls=AcquisitionParser,
    help="Exploration-exploitation tradeoff parameter. Used for EI, PI.",
)
@click.option("--debug", is_flag=True, help="Enable debug mode", default=False)
@click.pass_context
def predict(ctx, input_path, input_col, model_dir, output_csv, debug, accelerator):
    aq_groups = ctx.obj["aq_groups"]
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
