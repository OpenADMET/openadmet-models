"""ChemProp and Chemeleon model implementations."""

import json
import types
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import models, nn

from lightning import pytorch as pl
from loguru import logger
from pydantic import PrivateAttr, field_validator, model_validator

from openadmet.models.architecture.lightning_model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry


def configure_optimizers(self) -> dict:
    """
    Configure optimizers and learning rate schedulers.

    Returns
    -------
    dict
        A dictionary containing the optimizer and learning rate scheduler configurations.

    """
    # Separate parameters into MPNN and FFN groups
    mpnn_params = []
    ffn_params = []

    for name, param in self.named_parameters():
        if "predictor" in name:
            ffn_params.append(param)
        else:
            mpnn_params.append(param)

    # Set the optimizer base learning rates to their peak values
    param_groups = [
        {
            "params": mpnn_params,
            "lr": self.mpnn_lr,
            "weight_decay": self.mpnn_weight_decay,
        },
        {
            "params": ffn_params,
            "lr": self.ffn_lr,
            "weight_decay": self.ffn_weight_decay,
        },
    ]

    opt = torch.optim.AdamW(param_groups)

    if self.scheduler == "plateau":
        # Compute per-group LR floors proportional to each group's peak,
        # preserving the ratio final_lr / max_lr across param groups
        min_lrs = [
            group["lr"] * (self.final_lr / self.max_lr) for group in param_groups
        ]

        # Configure the reduce on plateau scheduler
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.monitor_metric_mode,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=min_lrs,
        )

        lr_sched_config = {
            "scheduler": lr_sched,
            "monitor": self.monitor_metric,
            "interval": "epoch",
            "frequency": 1,
        }
    elif self.scheduler == "noam":
        # Raw batch count per epoch is the correct unit for interval="step" scheduling;
        # estimated_stepping_batches is in optimizer-step units (divided by grad accumulation)
        # and would shorten warmup when accumulate_grad_batches > 1
        steps_per_epoch = getattr(self.trainer, "num_training_batches", None)
        if steps_per_epoch is None or steps_per_epoch == float("inf"):
            if isinstance(
                self.trainer.estimated_stepping_batches, int
            ) and self.trainer.estimated_stepping_batches != float("inf"):
                # Convert optimizer steps back to batch steps for gradient accumulation
                grad_accum = getattr(self.trainer, "accumulate_grad_batches", 1)
                steps_per_epoch = (
                    self.trainer.estimated_stepping_batches * grad_accum
                ) // max(1, self.trainer.max_epochs)
            else:
                logger.warning(
                    "Could not determine steps_per_epoch from trainer; falling back to 1000. "
                    "Noam schedule timing will be incorrect unless the dataset has exactly "
                    "1000 batches per epoch."
                )
                steps_per_epoch = 1000

        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.trainer.max_epochs == -1:
            if warmup_steps == 0:
                # No warmup and no epoch budget means no way to calibrate decay; hold at max_lr
                logger.warning(
                    "noam scheduler with max_epochs=-1 and warmup_epochs=0 cannot calibrate "
                    "decay; LR will be constant at max_lr for the entire run. "
                    "Set max_epochs or warmup_epochs > 0 to enable a meaningful schedule."
                )
                cooldown_steps = 0
            else:
                logger.warning(
                    "Setting cooldown steps to 100 times the warmup steps for infinite training."
                )
                cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            if cooldown_epochs <= 0:
                logger.warning(
                    f"warmup_epochs ({self.warmup_epochs}) >= max_epochs "
                    f"({self.trainer.max_epochs}); the cooldown phase has zero steps and "
                    "the LR will drop to final_lr on the first post-warmup step"
                )
                cooldown_steps = 0
            else:
                cooldown_steps = cooldown_epochs * steps_per_epoch

        # Convert absolute learning rates into scaling factors relative to max_lr
        # When mpnn_lr != max_lr, the MPNN group's absolute starting LR is
        # mpnn_lr * (init_lr / max_lr), not init_lr; the schedule shape is preserved
        # proportionally for each param group around its own peak
        init_factor = self.init_lr / self.max_lr
        final_factor = self.final_lr / self.max_lr

        # Lambda reaches exactly 1.0 at step == warmup_steps and exactly final_factor
        # at step == warmup_steps + cooldown_steps, with no discontinuity at either boundary
        # When both phases are zero (no warmup, infinite or unset max_epochs), the schedule
        # is a constant at max_lr rather than silently collapsing to final_lr
        def lr_lambda(step: int) -> float:
            if warmup_steps == 0 and cooldown_steps == 0:
                # No schedule configured; hold at max_lr for the entire run
                return 1.0
            if warmup_steps > 0 and step <= warmup_steps:
                # Linear ramp from init_lr to max_lr; owns the peak step
                return init_factor + (step / warmup_steps) * (1.0 - init_factor)
            elif cooldown_steps > 0 and step <= warmup_steps + cooldown_steps:
                # Geometric decay; no division guard needed since we require cooldown_steps > 0
                decay_frac = (step - warmup_steps) / cooldown_steps
                return final_factor**decay_frac
            else:
                return final_factor

        lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

    return {"optimizer": opt, "lr_scheduler": lr_sched_config}


def _warn_if_no_val_dataloader(self) -> None:
    """
    Emit a warning when plateau scheduler has no validation dataloader.

    Bound to on_train_start so num_val_batches is fully populated by Lightning
    before this check runs. configure_optimizers fires too early and sees an
    empty list even when a val split is configured.
    """
    if self.scheduler != "plateau":
        return
    num_val_batches = getattr(self.trainer, "num_val_batches", None)
    if num_val_batches is None:
        # Trainer did not expose the attribute; assume val exists
        return
    if hasattr(num_val_batches, "__iter__"):
        has_val = any(n > 0 for n in num_val_batches)
    else:
        # Integer path: 0 correctly means no validation
        has_val = num_val_batches > 0
    if not has_val:
        logger.warning(
            f"scheduler='plateau' monitors '{self.monitor_metric}' but no validation "
            "dataloader is configured; the scheduler will never step and training will "
            "fail when Lightning cannot find the metric. Use scheduler='noam' for "
            "train-only runs."
        )


@model_registry.register("ChemPropModel")
class ChemPropModel(LightningModelBase):
    """
    ChemProp regression model.

    This class implements a ChemProp-based regression model using message passing neural networks (MPNNs)
    for molecular property prediction. It supports various configurations for message passing, aggregation,
    and feed-forward network (FFN) layers. Can be initialized from the CheMeleon foundation model [REF], overriding
    settings for depth, message hidden dim, messages, and aggregation.

    Attributes
    ----------
    type : str
        The type of the model.
    n_tasks : int
        Number of prediction tasks.
    messages : str
        Type of message passing ("bond" or "atom").
    aggregation : str
        Aggregation method ("mean" or "norm"). Default is "mean", matching the original
        ChemProp paper baseline. "norm" uses a learned normalization parameter instead.
    depth : int
        Number of message passing steps.
    message_hidden_dim : int
        Hidden dimension size for message passing.
    ffn_hidden_dim : int
        Hidden dimension size for the feed-forward network.
    ffn_num_layers : int
        Number of layers in the feed-forward network. Default is 2 (one hidden layer);
        setting to 1 reduces the FFN to a single linear readout, making ffn_hidden_dim
        irrelevant.
    normalized_targets : bool
        Whether targets are normalized.
    batch_norm : bool
        Whether to use batch normalization.
    dropout : float
        Dropout rate.
    from_chemeleon : bool
        Whether to use the CheMeleon foundation model. Deprecated; use
        ``from_foundation='chemeleon'`` instead.
    monitor_metric : str
        The metric to monitor during training. Default is "val_loss".
    metric_list : list
        List of metrics to use for evaluation. Default is ["mse", "mae", "rmse"].
    scheduler : str
        Learning rate scheduler ("noam" or "plateau"). Default is "noam".

        Selection depends on the training regime:

        - "noam": for fixed-length training where the epoch budget is known in
          advance. The learning rate follows a preset trajectory, a linear ramp
          followed by a smooth decay across the configured run. This is the
          original ChemProp recipe and the appropriate default for standard
          from-scratch training. It depends on max_epochs being set correctly;
          an incorrect or open-ended budget distorts the trajectory.
        - "plateau": for runs whose length is not fixed in advance, such as
          early-stopped training or fine-tuning from a foundation model. The
          learning rate is reduced only when the monitored metric stops
          improving, adapting to observed progress rather than a preset
          timeline. Requires a validation set.

        For open-ended training (max_epochs=-1) or early stopping, prefer
        "plateau"; "noam" cannot shape its trajectory without a known budget.
    max_lr : float
        Peak learning rate (global reference). Default is 1e-3.
    final_lr : float, optional
        Floor LR for each param group. Defaults to max_lr * 0.01. When mpnn_lr or ffn_lr
        differ from max_lr, the absolute floor for that group is group_lr * (final_lr /
        max_lr); the ratio is preserved proportionally.
    weight_decay : float
        Global weight decay. Default is 0.0.
    mpnn_lr : float, optional
        Peak learning rate for the MPNN param group. If None, defaults to max_lr.
    ffn_lr : float, optional
        Peak learning rate for the FFN param group. If None, defaults to max_lr.
    mpnn_weight_decay : float, optional
        Weight decay for the MPNN param group. If None, defaults to weight_decay.
    ffn_weight_decay : float, optional
        Weight decay for the FFN param group. If None, defaults to weight_decay.
    warmup_epochs : int, optional
        [Noam only] Number of linear-ramp epochs before geometric decay. If None (default),
        resolves to 2 (matching the original ChemProp paper default). Setting this field
        with scheduler="plateau" raises ValueError. The schedule shape depends on max_epochs
        being set correctly in the Lightning Trainer; leaving it at the Lightning default
        (1000) when actual training runs shorter will under-decay the LR.
    init_lr : float, optional
        [Noam only] Starting LR at the beginning of the warmup ramp. Defaults to
        max_lr * 0.1. When mpnn_lr or ffn_lr differ from max_lr, the absolute starting LR
        for that group is group_lr * (init_lr / max_lr); the schedule shape is preserved
        proportionally around each group's peak.
    reduce_lr_factor : float, optional
        [Plateau only] Multiplicative factor applied when a plateau is detected. If None
        (default), resolves to 0.5. Must be < 1.0. Setting with scheduler="noam" raises
        ValueError.
    reduce_lr_patience : int, optional
        [Plateau only] Epochs with no improvement before LR is reduced. If None (default),
        resolves to 5. Setting with scheduler="noam" raises ValueError.
    monitor_metric_mode : str
        Direction for metric monitoring: "min" for loss-style metrics, "max" for score-style
        metrics. Default is "min". Currently consumed by the plateau scheduler; must also
        match any early-stopping callback monitoring the same metric.

    """

    # Meta parameters for this class
    type: ClassVar[str] = "ChemPropModel"

    # ChemProp parameters
    n_tasks: int = 1
    messages: str = "bond"
    aggregation: str = "mean"
    depth: int = 3
    message_hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    normalized_targets: bool = True
    batch_norm: bool = False
    dropout: float = 0.0
    from_foundation: str | None = None
    from_chemeleon: bool = False
    monitor_metric: str = "val_loss"
    metric_list: list = ["mse", "mae", "rmse"]

    # Select scheduler among "noam" or "plateau"
    scheduler: str = "noam"

    # Global defaults (master values)
    max_lr: float = 1e-3
    weight_decay: float = 0.0

    # Component overrides (optional - inherit from masters if None)
    mpnn_lr: float | None = None
    ffn_lr: float | None = None
    mpnn_weight_decay: float | None = None
    ffn_weight_decay: float | None = None

    # Scheduler specifics (optional - inherit from max_lr if None)
    init_lr: float | None = None
    final_lr: float | None = None

    # Noam-only parameters (None = 0, no warmup unless explicitly requested)
    warmup_epochs: int | None = None

    # Plateau-only parameters (None = use scheduler defaults)
    reduce_lr_factor: float | None = None
    reduce_lr_patience: int | None = None

    # Direction for plateau scheduler; must match any early-stopping callback on the same metric
    monitor_metric_mode: str = "min"

    _n_tasks: int = 1
    _explicit_init_fields: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="before")
    @classmethod
    def handle_from_chemeleon_compat(cls, data: dict) -> dict:
        """Translate deprecated ``from_chemeleon`` flag into ``from_foundation``."""
        if not isinstance(data, dict):
            return data
        from_chemeleon = data.get("from_chemeleon", False)
        from_foundation = data.get("from_foundation")
        if from_chemeleon:
            import warnings

            warnings.warn(
                "from_chemeleon is deprecated; use from_foundation='chemeleon' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if from_foundation is not None and from_foundation != "chemeleon":
                raise ValueError(
                    f"Cannot specify both from_chemeleon and user-specified from_foundation: {from_foundation}"
                )
            data["from_foundation"] = "chemeleon"
        return data

    def __init__(self, **data):
        """Initialize the model and track explicitly provided fields."""
        explicit_init_fields = set(data)
        super().__init__(**data)
        self._explicit_init_fields = explicit_init_fields.intersection(
            type(self).model_fields.keys()
        )

    @model_validator(mode="after")
    def resolve_hyperparameters(self) -> "ChemPropModel":
        """
        Resolve hyperparameters using global defaults and component overrides pattern.

        Logic:
        - Resolve learning rates:
            - init_lr -> max_lr * 0.1
            - final_lr -> max_lr * 0.01
            - mpnn_lr -> max_lr
            - ffn_lr -> max_lr
        - Resolve weight decays:
            - mpnn_weight_decay -> weight_decay
            - ffn_weight_decay -> weight_decay
        - Fill scheduler-specific defaults (only for the active scheduler):
            - noam: warmup_epochs -> 2
            - plateau: reduce_lr_factor -> 0.5, reduce_lr_patience -> 5
        """
        # Resolve LRs
        if self.init_lr is None:
            self.init_lr = self.max_lr * 0.1
        if self.final_lr is None:
            self.final_lr = self.max_lr * 0.01
        if self.mpnn_lr is None:
            self.mpnn_lr = self.max_lr
        if self.ffn_lr is None:
            self.ffn_lr = self.max_lr

        # Resolve weight decays
        if self.mpnn_weight_decay is None:
            self.mpnn_weight_decay = self.weight_decay
        if self.ffn_weight_decay is None:
            self.ffn_weight_decay = self.weight_decay

        # Fill scheduler-specific defaults only for the active scheduler
        if self.scheduler == "noam":
            if self.warmup_epochs is None:
                self.warmup_epochs = 2
        elif self.scheduler == "plateau":
            if self.reduce_lr_factor is None:
                self.reduce_lr_factor = 0.5
            if self.reduce_lr_patience is None:
                self.reduce_lr_patience = 5

        return self

    @model_validator(mode="after")
    def validate_scheduler_params(self) -> "ChemPropModel":
        """
        Ensure scheduler-specific parameters are valid for the chosen scheduler.

        Cross-scheduler params use None as the "not set" sentinel so this validator
        can distinguish user-supplied values from unset fields without relying on
        model_fields_set (which only tracks explicitly provided keys).
        """
        if self.scheduler == "noam":
            if self.reduce_lr_factor is not None:
                raise ValueError(
                    "reduce_lr_factor is not compatible with noam scheduler"
                )
            if self.reduce_lr_patience is not None:
                raise ValueError(
                    "reduce_lr_patience is not compatible with noam scheduler"
                )
        elif self.scheduler == "plateau":
            if self.warmup_epochs is not None:
                raise ValueError(
                    "warmup_epochs is not compatible with plateau scheduler"
                )
            # reduce_lr_factor is filled by resolve_hyperparameters before this runs
            if self.reduce_lr_factor is not None and self.reduce_lr_factor >= 1.0:
                raise ValueError("reduce_lr_factor must be < 1.0 for plateau scheduler")
        return self

    @model_validator(mode="after")
    def set_n_tasks(self) -> "ChemPropModel":
        """
        Set the number of tasks for the model.

        Returns
        -------
        ChemPropModel
            The updated model instance.

        """
        self._n_tasks = self.n_tasks
        return self

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        """
        Validate the messages parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        """
        if value not in ["bond", "atom"]:
            raise ValueError("Messages must be either 'bond' or 'atom'")
        return value

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value):
        """
        Validate the aggregation parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        """
        if value not in ["mean", "norm"]:
            raise ValueError("Aggregation must be either 'mean' or 'norm'")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value):
        """
        Validate the scheduler parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        """
        if value not in ["noam", "plateau"]:
            raise ValueError("Scheduler must be either 'noam' or 'plateau'")
        return value

    @field_validator("monitor_metric_mode")
    @classmethod
    def validate_monitor_metric_mode(cls, value):
        """
        Validate the monitor_metric_mode parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        """
        if value not in ["min", "max"]:
            raise ValueError("monitor_metric_mode must be either 'min' or 'max'")
        return value

    def _get_output_transform(self, scaler):
        """
        Convert scaler to the output transform needed for predictions.

        Parameters
        ----------
        scaler : object
            The scaler to use for unscaling predictions.

        Returns
        -------
        nn.UnscaleTransform or None
            The output transform to apply to predictions.

        """
        if scaler is not None:
            output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        elif self.normalized_targets:
            # Expects the targets to be normalized, likely to be loaded from state dict
            output_transform = nn.UnscaleTransform(
                [1] * self.n_tasks, [0] * self.n_tasks
            )
        else:
            output_transform = None
        return output_transform

    def build(self, scaler=None):
        """
        Prepare and build the ChemProp model.

        Downloads and loads the CheMeleon foundation model if specified, otherwise
        constructs a new MPNN model with the given configuration.

        Parameters
        ----------
        scaler : object, optional
            Scaler for target normalization.

        Returns
        -------
        self : ChemPropModel
            The current instance with the estimator built.

        """
        if not self.estimator:
            _METRIC_TO_LOSS = {
                "mse": nn.metrics.MSE(),
                "mae": nn.metrics.MAE(),
                "rmse": nn.metrics.RMSE(),
            }
            metric_list = [_METRIC_TO_LOSS[metric] for metric in self.metric_list]
            if self.from_foundation:
                if self.from_foundation == "chemeleon":
                    foundation_mp = self._download_chemeleon(save_dir=Path.home())
                    logger.warning(
                        "Using CheMeleon overrides settings for depth, message_hidden_dim, messages, and aggregation"
                    )
                else:
                    logger.info(f"Loading foundation model from {self.from_foundation}")
                    foundation_mp = self._load_foundation_model(
                        Path(self.from_foundation)
                    )
                aggr = nn.MeanAggregation()
                mp = nn.BondMessagePassing(**foundation_mp["hyper_parameters"])
                mp.load_state_dict(foundation_mp["state_dict"])
                self.message_hidden_dim = mp.output_dim
                logger.warning(
                    "Using a foundation model overrides settings for depth, message_hidden_dim, messages, and aggregation"
                )
            else:
                aggregation_cls = (
                    nn.MeanAggregation
                    if self.aggregation == "mean"
                    else nn.NormAggregation
                )
                message_cls = (
                    nn.BondMessagePassing
                    if self.messages == "bond"
                    else nn.AtomMessagePassing
                )

                # Create the model
                mp = message_cls(
                    d_h=self.message_hidden_dim, depth=self.depth, dropout=self.dropout
                )
                aggr = aggregation_cls()

            ffn = nn.RegressionFFN(
                n_tasks=self.n_tasks,
                input_dim=self.message_hidden_dim,
                hidden_dim=self.ffn_hidden_dim,
                n_layers=self.ffn_num_layers,
                output_transform=self._get_output_transform(scaler),
                dropout=self.dropout,
            )

            # warmup_epochs, init_lr, max_lr, and final_lr are MPNN constructor parameters,
            # so Lightning records them in hparams.yaml automatically. Omit them for plateau
            # to avoid misleading entries. Plateau-specific params (reduce_lr_factor, etc.)
            # are not constructor args and are set as plain attributes below, so they never
            # appear in hparams regardless of scheduler — no plateau_kwargs needed
            noam_kwargs = (
                dict(
                    warmup_epochs=self.warmup_epochs,
                    init_lr=self.init_lr,
                    max_lr=self.max_lr,
                    final_lr=self.final_lr,
                )
                if self.scheduler == "noam"
                else {}
            )

            # Create the MPNN model
            mpnn = models.MPNN(
                message_passing=mp,
                agg=aggr,
                predictor=ffn,
                batch_norm=self.batch_norm,
                metrics=metric_list,
                **noam_kwargs,
            )

            # Ensure scheduler is always recorded in Lightning hparams. For plateau, also
            # correct the LR keys that MPNN stored from its constructor defaults (which may
            # differ from the user's configured values since noam_kwargs is empty for plateau)
            # and remove Noam-only keys that do not apply
            mpnn.hparams.update({"scheduler": self.scheduler})
            if self.scheduler == "plateau":
                mpnn.hparams.update({"max_lr": self.max_lr, "final_lr": self.final_lr})
                mpnn.hparams.pop("warmup_epochs", None)
                mpnn.hparams.pop("init_lr", None)

            # Pass monitor metric from "model" to "module"
            # This is necessary to support subclasses of LightningModuleBase, as `monitor_metric`
            # is needed at the "module" level for use in both `configure_optimizers` and `LightningTrainer`
            mpnn.monitor_metric = self.monitor_metric

            # Attach custom optimization parameters to the MPNN instance
            mpnn.mpnn_weight_decay = self.mpnn_weight_decay
            mpnn.ffn_weight_decay = self.ffn_weight_decay
            mpnn.mpnn_lr = self.mpnn_lr
            mpnn.ffn_lr = self.ffn_lr
            mpnn.final_lr = self.final_lr
            mpnn.max_lr = self.max_lr
            mpnn.reduce_lr_factor = self.reduce_lr_factor
            mpnn.reduce_lr_patience = self.reduce_lr_patience
            mpnn.scheduler = self.scheduler
            mpnn.monitor_metric_mode = self.monitor_metric_mode

            # Bind the custom configure_optimizers method
            mpnn.configure_optimizers = types.MethodType(configure_optimizers, mpnn)

            # Bind the val-split check to on_train_start where num_val_batches is reliable
            mpnn.on_train_start = types.MethodType(_warn_if_no_val_dataloader, mpnn)

            self.estimator = mpnn

        else:
            logger.warning("Model already exists, skipping build")

        return self

    def _download_chemeleon(self, save_dir: Path) -> Path:
        """
        Download the CheMeleon foundation model.

        Parameters
        ----------
        save_dir : Path
            Directory to save the downloaded model.

        Returns
        -------
        Path
            Path to the downloaded model file.

        """
        ckpt_dir = save_dir / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        model_path = ckpt_dir / "chemeleon_mp.pt"
        if not model_path.exists():
            logger.info(
                f"Downloading CheMeleon Foundation model from Zenodo (https://zenodo.org/records/15460715) to {model_path}"
            )
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                model_path,
            )
        else:
            logger.info(f"Loading cached CheMeleon from {model_path}")
        return torch.load(model_path, weights_only=True)

    def _load_foundation_model(self, model_path: Path) -> dict:
        """
        Load a foundation model from the specified path.

        Parameters
        ----------
        model_path : Path
            Path to the foundation model file.

        Returns
        -------
        dict
            The loaded foundation model state.

        """
        if not model_path.exists():
            raise FileNotFoundError(f"Foundation model not found at {model_path}")
        return torch.load(model_path, weights_only=True)

    def train(self, dataloader, scaler=None):
        """
        Train the model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data.
        scaler : object, optional
            Scaler for target normalization.

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    # Fields always included in the serialized artifact regardless of whether the user
    # set them explicitly. Covers two categories:
    #   - Structural fields: determine model graph shape and checkpoint compatibility;
    #     omitting any of these from the artifact makes reloading fragile when the class
    #     default ever changes
    #   - Resolved LR fields: computed from max_lr at init time and needed for exact
    #     schedule reproduction on reload
    # Scheduler-specific fields (warmup_epochs, reduce_lr_factor, reduce_lr_patience) are
    # in this set but are None for the inactive scheduler; serialize() drops None entries
    # so only the active scheduler's fields appear in the artifact
    _RESOLVED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            # Structural
            "scheduler",
            "n_tasks",
            "depth",
            "message_hidden_dim",
            "ffn_hidden_dim",
            "ffn_num_layers",
            "aggregation",
            "messages",
            "batch_norm",
            "dropout",
            "normalized_targets",
            # Resolved LRs
            "init_lr",
            "final_lr",
            "mpnn_lr",
            "ffn_lr",
            "mpnn_weight_decay",
            "ffn_weight_decay",
            # Scheduler-specific (None for inactive scheduler; excluded below)
            "warmup_epochs",
            "reduce_lr_factor",
            "reduce_lr_patience",
        }
    )

    def serialize(self, param_path="model.json", serial_path="model.pth"):
        """
        Save the model with explicitly provided fields plus resolved LR hyperparameters.

        Parameters
        ----------
        param_path: PathLike
            Path to save the model parameters to
        serial_path: PathLike
            Path to save the serialized model to

        """
        # Exclude None entries so inactive-scheduler fields don't appear in the artifact
        non_none_resolved = {
            k for k in self._RESOLVED_FIELDS if getattr(self, k, None) is not None
        }
        explicit_params = self.model_dump(
            include=self._explicit_init_fields | non_none_resolved
        )
        with open(param_path, "w") as f:
            json.dump(explicit_params, f, indent=2)
        self.save(serial_path)

    def make_new(self) -> "ChemPropModel":
        """Copy parameters to a new model instance without copying the estimator."""
        explict_params = self.model_dump(
            include=self._explicit_init_fields, exclude={"estimator"}
        )
        return self.__class__(**explict_params)

    def predict(
        self, X: np.ndarray, accelerator="gpu", devices=1, **kwargs
    ) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data for prediction.
        accelerator : str, optional
            Accelerator type to use ("gpu" or "cpu").
        devices : int, optional
            Number of devices to use for prediction.
        **kwargs
            Additional keyword arguments for the trainer.

        Returns
        -------
        np.ndarray
            Model predictions.

        """
        if not self.estimator:
            raise AttributeError("Model not trained")

        self.estimator.eval()

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=False,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=devices,
            )
            preds = trainer.predict(self.estimator, X)
        return torch.cat(preds).numpy()

    def freeze_weights(
        self, message_passing: bool = True, batch_norm: bool = True, ffn_layers: int = 0
    ):
        """
        Freeze parts of the model for transfer learning or fine-tuning.

        Parameters
        ----------
        message_passing : bool, optional
            If True, freeze the message passing layers. Default is True.
        batch_norm : bool, optional
            If True, freeze the batch normalization layers. Default is True.
        ffn_layers : int, optional
            Number of feed-forward network (FFN) layers to freeze. Default is 0.

        Notes
        -----
        This method sets the `requires_grad` attribute of the specified layers to False,
        preventing their weights from being updated during training. It also sets these
        layers to evaluation mode.

        """
        # Check number of layers
        if ffn_layers > self.ffn_num_layers:
            raise ValueError(
                f"Requested to freeze {ffn_layers} feedforward network layer(s), "
                f"but only {self.ffn_num_layers} available."
            )

        # Freeze message passing
        if message_passing:
            # No gradient updates
            self.estimator.message_passing.apply(
                lambda module: module.requires_grad_(False)
            )
            # Set to evaluation mode
            self.estimator.message_passing.eval()

            # Log for message passing
            logger.info(f"Model weights for message passing frozen.")

        # Freeze batch norm
        if batch_norm:
            # No gradient updates
            self.estimator.bn.apply(lambda module: module.requires_grad_(False))
            # Evaluation mode
            self.estimator.bn.eval()
            # Log for batch normalization
            logger.info(f"Model weights for batch normalization frozen.")

        # Freeze feedforward network
        if ffn_layers > 0:
            for idx in range(ffn_layers):
                # No gradient updates
                self.estimator.predictor.ffn[idx].requires_grad_(False)
                # Evaluation mode (same layer as the gradient freeze)
                self.estimator.predictor.ffn[idx].eval()

            # Log for feedforward network
            logger.info(
                f"Model weights for {ffn_layers} feedforward network layer(s) frozen."
            )
