"""Monroe foundation model embedding featurizer."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import torch
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator, model_validator

from openadmet.models.architecture.chemprop import _resolve_device
from openadmet.models.features.feature_base import FeaturizerBase, featurizers

# The monroe package and the checkpoint are pinned to the same commit, because the
# embedding is a function of both the weights and the graph-construction code
_MONROE_CKPT_COMMIT = "3e138d9bb947e89c9f8faf7db86210385d6657c2"

_MONROE_INSTALL_HINT = (
    "MonroeFeaturizer requires the monroe package, which is not published on PyPI. "
    "Install it with: pip install "
    f"git+https://github.com/blazejba/monroe.git@{_MONROE_CKPT_COMMIT}"
)

_MONROE_CONFIG_URL = (
    f"https://raw.githubusercontent.com/blazejba/monroe/{_MONROE_CKPT_COMMIT}"
    "/checkpoint/config.json"
)

# weights.pt is Git LFS tracked, so raw.githubusercontent.com serves a pointer file
# instead of the weights; the media endpoint serves the real bytes
_MONROE_WEIGHTS_URL = (
    f"https://media.githubusercontent.com/media/blazejba/monroe/{_MONROE_CKPT_COMMIT}"
    "/checkpoint/weights.pt"
)
_MONROE_WEIGHTS_SHA256 = (
    "df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a"
)
_MONROE_WEIGHTS_BYTES = 300057823

# Width of the pinned checkpoint's graph-level embedding, which is its encoder
# hidden_dim; lets an empty input be shaped without downloading anything
_MONROE_EMBEDDING_DIM = 720

# Written last, once both files are promoted, so its presence and contents are what
# distinguish a complete cache entry from a partial or tampered one
_VERIFIED_MARKER = "weights.sha256"

# Cap on how many failed SMILES the drop warning names before it elides the rest
_MAX_REPORTED_DROPS = 10


def _import_monroe() -> tuple[
    Callable[..., torch.nn.Module], Callable[..., dict[str, np.ndarray]]
]:
    """
    Import monroe's checkpoint loader and embedder, deferring the cost to first use.

    Importing lazily is what lets this module be imported, and therefore
    registered, in an environment where monroe is not installed.

    Returns
    -------
    tuple
        The ``(load_ckpt, embed_smiles)`` callables from monroe.

    Raises
    ------
    ImportError
        If monroe is not installed, re-raised with installation instructions.

    """
    try:
        from monroe.eval.embed import embed_smiles
        from monroe.model.ckpt import load_ckpt
    except ImportError as e:
        raise ImportError(_MONROE_INSTALL_HINT) from e
    return load_ckpt, embed_smiles


def _sha256(path: Path) -> str:
    """
    Return the hex sha256 digest of a file, read in chunks.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    str
        Lowercase hex digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified_cache(ckpt_dir: Path) -> bool:
    """
    Report whether a cache directory holds a complete, digest-matched checkpoint.

    The marker records the digest of the weights that were promoted, so a cache
    entry that was truncated, half-written, or replaced on disk after the fact
    fails this check rather than being trusted forever on its size alone.

    Parameters
    ----------
    ckpt_dir : Path
        Candidate cache directory.

    Returns
    -------
    bool
        True when the directory can be handed to monroe as-is.

    """
    marker = ckpt_dir / _VERIFIED_MARKER
    weights = ckpt_dir / "weights.pt"

    if not (
        marker.is_file() and weights.is_file() and (ckpt_dir / "config.json").is_file()
    ):
        return False

    # Size is the cheap guard against a weights file replaced since promotion;
    # the marker is what ties the entry to the pinned digest
    if weights.stat().st_size != _MONROE_WEIGHTS_BYTES:
        return False

    return marker.read_text().strip() == _MONROE_WEIGHTS_SHA256


def _fetch_to(url: str, destination: Path) -> None:
    """
    Download a URL into place through a process-private scratch file.

    Promoting by rename means a concurrent reader sees either the previous
    contents or the complete new ones, never a partial transfer.

    Parameters
    ----------
    url : str
        Source URL.
    destination : Path
        Final path to promote the download to.

    """
    scratch = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    try:
        urlretrieve(url, scratch)
        scratch.replace(destination)
    finally:
        scratch.unlink(missing_ok=True)


def _download_monroe_checkpoint(cache_root: Path | None = None) -> Path:
    """
    Fetch the published Monroe checkpoint, returning a directory monroe can load.

    Parameters
    ----------
    cache_root : Path, optional
        Root under which checkpoints are cached, by default ``~/.openadmet``.

    Returns
    -------
    Path
        Directory holding ``config.json`` and ``weights.pt``.

    Raises
    ------
    RuntimeError
        If the downloaded weights do not match the expected sha256 digest.

    """
    root = cache_root if cache_root is not None else Path.home() / ".openadmet"
    ckpt_dir = root / "monroe" / _MONROE_CKPT_COMMIT
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _is_verified_cache(ckpt_dir):
        logger.info("Loading cached Monroe checkpoint from {}", ckpt_dir)
        return ckpt_dir

    logger.info(
        "Downloading Monroe checkpoint ({:.0f} MB) to {}",
        _MONROE_WEIGHTS_BYTES / 1e6,
        ckpt_dir,
    )

    # The marker is removed first, so an interrupted re-download cannot leave the
    # previous marker vouching for the new weights
    marker_path = ckpt_dir / _VERIFIED_MARKER
    marker_path.unlink(missing_ok=True)

    _fetch_to(_MONROE_CONFIG_URL, ckpt_dir / "config.json")

    weights_path = ckpt_dir / "weights.pt"
    scratch = weights_path.with_name(f"weights.pt.{os.getpid()}.part")
    try:
        urlretrieve(_MONROE_WEIGHTS_URL, scratch)

        digest = _sha256(scratch)
        if digest != _MONROE_WEIGHTS_SHA256:
            raise RuntimeError(
                f"Monroe checkpoint download from {_MONROE_WEIGHTS_URL} is corrupt: "
                f"expected sha256 {_MONROE_WEIGHTS_SHA256}, got {digest}. "
                "The partial download has been removed; retry the featurization."
            )

        scratch.replace(weights_path)
    finally:
        scratch.unlink(missing_ok=True)

    marker_path.write_text(_MONROE_WEIGHTS_SHA256)
    return ckpt_dir


def _load_encoder(checkpoint_dir: Path, use_ema: bool) -> torch.nn.Module:
    """
    Build the frozen Monroe encoder from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory holding ``config.json`` and ``weights.pt``.
    use_ema : bool
        Whether to load the exponential-moving-average weights.

    Returns
    -------
    torch.nn.Module
        The encoder, as returned by monroe's ``load_ckpt``.

    """
    load_ckpt, _ = _import_monroe()
    return load_ckpt(str(checkpoint_dir), use_ema=use_ema)


def _embed_smiles(
    smiles: list[str],
    encoder: torch.nn.Module,
    device: str,
    batch_size: int,
    n_workers: int,
) -> dict[str, np.ndarray]:
    """
    Embed SMILES with monroe, returning its SMILES-keyed mapping unchanged.

    Parameters
    ----------
    smiles : list of str
        SMILES strings to embed.
    encoder : torch.nn.Module
        Frozen Monroe encoder.
    device : str
        Torch device name to run the forward pass on.
    batch_size : int
        Molecules per forward pass.
    n_workers : int
        Worker processes used for graph construction.

    Returns
    -------
    dict
        Mapping of SMILES to embedding vector, keyed by the verbatim input
        string. Molecules monroe could not featurize are absent from the mapping.

    """
    _, embed_smiles = _import_monroe()
    return embed_smiles(
        smiles,
        encoder,
        device=device,
        batch_size=batch_size,
        n_workers=n_workers,
    )


@featurizers.register("MonroeFeaturizer")
class MonroeFeaturizer(FeaturizerBase):
    """
    Return Monroe graph-level embeddings for SMILES.

    Monroe is a GRIT transformer pretrained jointly on PM6 quantum-chemical
    properties and PCBA bioassay labels. This featurizer runs its encoder frozen:
    the pretrained weights are used as-is and no training is performed.

    The published checkpoint is downloaded and cached under ``~/.openadmet`` on
    first use, verified against a pinned sha256. Set ``checkpoint_path`` to use a
    locally trained encoder instead.

    Monroe drops molecules whose graph it cannot build rather than raising. Those
    positions are reported through the returned index array, which a
    ``FeatureConcatenator`` intersects away. Callers that use ``featurize``
    directly must apply the indices themselves to keep features aligned with
    their targets.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.
    checkpoint_path : Path or None
        Directory holding ``config.json`` and ``weights.pt``, by default None,
        which downloads and caches the published checkpoint.
    accelerator : str
        Device to use for inference, by default "auto", which resolves like
        Lightning's auto accelerator: TPU, MPS, or CUDA where available,
        otherwise CPU.
    batch_size : int
        Number of molecules per forward pass, by default 32.
    n_workers : int or None
        Worker processes used for graph construction, by default None, which
        uses ``os.cpu_count()``.
    use_ema : bool
        Whether to load the checkpoint's exponential-moving-average weights, by
        default False. The published checkpoint ships no EMA weights, so this
        requires ``checkpoint_path``.

    Notes
    -----
    **Embeddings are not reproducible across runs.** Monroe builds a conformer
    per molecule with ETKDGv3 and never sets ``randomSeed``, and the pinned
    checkpoint has ``zero_vn_edge_rbf`` disabled, so the arbitrary coordinate
    frame of that conformer reaches the features. Featurizing the same molecule
    twice yields two different vectors, and monroe exposes no seed to change
    that. Materialize features once and reuse them rather than re-featurizing:
    in particular, features used for training and features used at inference
    must come from the same featurization pass.

    Monroe forks its graph-building pool after the encoder has been moved to the
    device. On a CUDA host this forks a process with an initialized CUDA
    context. Set ``n_workers=1`` to avoid the pool, or featurize with
    ``accelerator="cpu"``, if you hit fork-related hangs.

    Monroe identifies molecules by the verbatim SMILES string it was handed, and
    routes structures through an InChI round trip, which normalizes tautomers and
    discards enhanced (AND/OR) stereo. Defined tetrahedral and double-bond stereo
    survive. As elsewhere in this package, inputs are expected to be canonical,
    salt-stripped SMILES; standardization belongs upstream of the featurizer.

    A ``checkpoint_path`` whose ``config.json`` disagrees with its ``weights.pt``
    builds a partially randomly-initialized encoder, because monroe loads state
    with ``strict=False``. Point at a config and weights from the same run.

    """

    type: ClassVar[str] = "MonroeFeaturizer"

    checkpoint_path: Path | None = Field(
        default=None,
        description="Local Monroe checkpoint directory; None downloads the published one",
    )
    accelerator: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    n_workers: int | None = Field(default=None, ge=1)
    use_ema: bool = False

    # Cached after a successful build, so featurizing several partitions loads once
    _encoder: torch.nn.Module | None = PrivateAttr(default=None)

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """
        Validate that the accelerator resolves to a torch-recognized device.

        Checking eagerly here means a bad accelerator fails at construction time
        rather than part-way through a featurization run.

        Parameters
        ----------
        value : str
            Accelerator value to validate.

        Returns
        -------
        str
            The validated accelerator value.

        Raises
        ------
        ValueError
            If the value does not resolve to a torch device.

        """
        try:
            torch.device(_resolve_device(value))
        except RuntimeError as e:
            raise ValueError(f"Invalid accelerator {value!r}: {e}") from e
        return value

    @field_validator("checkpoint_path")
    @classmethod
    def validate_checkpoint_path(cls, value: Path | None) -> Path | None:
        """
        Check the path holds both checkpoint files without loading the weights.

        Parameters
        ----------
        value : Path or None
            The configured checkpoint directory.

        Returns
        -------
        Path or None
            The validated directory.

        Raises
        ------
        ValueError
            If the directory, its config.json, or its weights.pt is missing.

        """
        if value is None:
            return None

        value = Path(value)

        if not value.is_dir():
            raise ValueError(f"Monroe checkpoint directory {value} does not exist.")

        for filename in ("config.json", "weights.pt"):
            if not (value / filename).is_file():
                raise ValueError(
                    f"Monroe checkpoint directory {value} has no {filename}, so it "
                    "is not a Monroe checkpoint."
                )

        return value

    @model_validator(mode="after")
    def check_ema_is_available(self):
        """
        Check EMA weights can exist for the configured checkpoint.

        The published checkpoint ships only config.json and weights.pt, so
        requesting EMA weights from it downloads 300 MB before failing inside
        monroe. The recipe answers this without touching the network.

        Raises
        ------
        ValueError
            If EMA weights are requested from the published checkpoint.

        """
        if self.use_ema and self.checkpoint_path is None:
            raise ValueError(
                "use_ema is set, but the published Monroe checkpoint ships no EMA "
                "weights. Leave use_ema unset, or set checkpoint_path to a "
                "checkpoint containing ema_weights.pt."
            )

        return self

    @property
    def embedding_dim(self) -> int:
        """
        Return the width of the emitted embeddings, without loading any weights.

        Returns
        -------
        int
            The encoder hidden dimension.

        Raises
        ------
        ValueError
            If a configured checkpoint's config.json cannot be parsed or does
            not declare an encoder hidden dimension.

        """
        if self.checkpoint_path is None:
            return _MONROE_EMBEDDING_DIM

        config_path = self.checkpoint_path / "config.json"
        try:
            config = json.loads(config_path.read_text())
            return config["encoder"]["hidden_dim"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Monroe checkpoint config {config_path} does not declare "
                f"encoder.hidden_dim: {e}"
            ) from e

    @property
    def encoder(self) -> torch.nn.Module:
        """Return the frozen Monroe encoder, building it on first access."""
        if self._encoder is None:
            checkpoint_dir = self.checkpoint_path or _download_monroe_checkpoint()

            # Cache only after the load succeeds, so a failure is not memoized
            self._encoder = _load_encoder(checkpoint_dir, self.use_ema)

        return self._encoder

    def _empty_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the zero-row result, shaped so concatenation stays well defined."""
        return (
            np.empty((0, self.embedding_dim), dtype=np.float32),
            np.empty(0, dtype=int),
        )

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize a list of SMILES strings.

        Parameters
        ----------
        smiles : Iterable[str]
            Canonical, salt-stripped SMILES strings to featurize. Entries Monroe
            cannot build a graph for are dropped rather than raising, and are
            reported through the returned indices.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 2D numpy array of shape
            (n_featurized, embedding_dim) and indices is a 1D numpy array giving
            the input position of each feature row, in input order.

        Raises
        ------
        RuntimeError
            If monroe returns vectors of an unexpected rank or width.

        """
        smiles_list = list(smiles)

        # Shaped from the config, so an empty input never downloads or builds
        if not smiles_list:
            return self._empty_features()

        # monroe falls back to os.sched_getaffinity when this is None, which does
        # not exist on macOS, so always hand it an explicit count
        workers = os.cpu_count() or 1 if self.n_workers is None else self.n_workers

        # Conformer generation dominates the cost and monroe collapses repeats in
        # its result anyway, so pay for each distinct structure once
        unique_smiles = list(dict.fromkeys(smiles_list))

        embedded = _embed_smiles(
            unique_smiles,
            self.encoder,
            device=_resolve_device(self.accelerator),
            batch_size=self.batch_size,
            n_workers=workers,
        )

        # monroe keys its result by the verbatim input string, so walking the
        # input restores the caller's order and repeats resolve to one vector
        kept = [i for i, smi in enumerate(smiles_list) if smi in embedded]

        if len(kept) < len(smiles_list):
            self._warn_dropped(smiles_list, embedded)

        if not kept:
            return self._empty_features()

        features = np.stack([embedded[smiles_list[i]] for i in kept])
        self._check_shape(features)

        return features.astype(np.float32), np.asarray(kept, dtype=int)

    def _warn_dropped(
        self, smiles_list: list[str], embedded: dict[str, np.ndarray]
    ) -> None:
        """
        Log which distinct structures Monroe could not featurize.

        Parameters
        ----------
        smiles_list : list of str
            The full input, in order.
        embedded : dict
            Monroe's result, keyed by the SMILES it succeeded on.

        """
        dropped = [smi for smi in dict.fromkeys(smiles_list) if smi not in embedded]

        shown = ", ".join(dropped[:_MAX_REPORTED_DROPS])
        if len(dropped) > _MAX_REPORTED_DROPS:
            shown += f", ... ({len(dropped) - _MAX_REPORTED_DROPS} more)"

        logger.warning(
            "Monroe could not featurize {} of {} distinct structures; the "
            "affected rows are excluded from the returned indices: {}",
            len(dropped),
            len(dict.fromkeys(smiles_list)),
            shown,
        )

    def _check_shape(self, features: np.ndarray) -> None:
        """
        Check monroe's vectors match the width this featurizer advertises.

        ``embedding_dim`` shapes the empty result, so a disagreement would give a
        partition with no valid molecules a different column count from a normal
        one, silently.

        Parameters
        ----------
        features : np.ndarray
            The stacked embedding matrix.

        Raises
        ------
        RuntimeError
            If the matrix is not 2D or its width is not ``embedding_dim``.

        """
        expected = self.embedding_dim

        if features.ndim != 2:
            raise RuntimeError(
                f"Monroe returned rank-{features.ndim - 1} vectors; expected one "
                "vector per molecule."
            )

        if features.shape[1] != expected:
            raise RuntimeError(
                f"Monroe returned {features.shape[1]}-wide embeddings, but this "
                f"checkpoint declares {expected}."
            )
