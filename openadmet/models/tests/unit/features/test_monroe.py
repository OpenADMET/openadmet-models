"""Tests for MonroeFeaturizer."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openadmet.models.features import monroe as monroe_module
from openadmet.models.features.combine import FeatureConcatenator
from openadmet.models.features.feature_base import get_featurizer_class
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.monroe import (
    _VERIFIED_MARKER,
    MonroeFeaturizer,
    _download_monroe_checkpoint,
)


@pytest.fixture
def smiles():
    return ["CCO", "CCN", "c1ccccc1"]


@pytest.fixture
def featurizer_factory(mocker):
    """
    Build a featurizer with a seeded encoder and monroe's embedder patched out.

    monroe's embed_smiles spawns a process pool to build conformers, which is the
    external boundary here; everything this module owns (order reconstruction,
    drop reporting, dtype, the index contract) sits on this side of it. The
    upstream contract this stand-in assumes is pinned separately by
    test_real_embed_smiles_contract.
    """

    def _make(vectors, dim=720, **kwargs):
        def _embed(smiles, encoder, device, batch_size, n_workers):
            return {
                smi: np.full(dim, value, dtype=np.float64)
                for smi, value in vectors.items()
            }

        embed_mock = mocker.patch.object(
            monroe_module, "_embed_smiles", autospec=True, side_effect=_embed
        )

        featurizer = MonroeFeaturizer(**kwargs)

        # Any non-None object satisfies the lazy encoder property; the patched
        # embedder ignores it
        featurizer._encoder = object()

        return featurizer, embed_mock

    return _make


@pytest.fixture
def local_checkpoint(tmp_path):
    """A checkpoint directory whose config declares a narrow, non-default width."""
    ckpt_dir = tmp_path / "monroe_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps({"encoder": {"hidden_dim": 8}}))
    (ckpt_dir / "weights.pt").write_bytes(b"")
    return ckpt_dir


@pytest.fixture
def pin_artifact(mocker):
    """Shrink the pinned artifact constants so download tests move bytes, not 300 MB."""

    def _apply(payload: bytes):
        mocker.patch.object(
            monroe_module, "_MONROE_WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest()
        )
        mocker.patch.object(monroe_module, "_MONROE_WEIGHTS_BYTES", len(payload))

    return _apply


@pytest.fixture
def fake_download(mocker):
    """Patch urlretrieve so it writes canned bytes instead of reaching the network."""

    def _apply(config: bytes, weights: bytes):
        def _write(url, destination):
            destination = Path(destination)

            # The real urlretrieve writes to the scratch name the caller chose,
            # so match on the artifact rather than the process-private suffix
            key = "config" if "config" in destination.name else "weights"
            destination.write_bytes(config if key == "config" else weights)
            return destination, None

        return mocker.patch.object(
            monroe_module, "urlretrieve", autospec=True, side_effect=_write
        )

    return _apply


def _cache_dir(root: Path) -> Path:
    return root / "monroe" / monroe_module._MONROE_CKPT_COMMIT


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "mps", "cuda:0"])
def test_accelerator_validator_accepts_known_values(accelerator):
    """cpu, gpu, and unaliased torch device spellings must all construct cleanly."""
    assert MonroeFeaturizer(accelerator=accelerator).accelerator == accelerator


def test_accelerator_validator_rejects_bad_value():
    """An accelerator torch.device() cannot parse must fail at construction time."""
    with pytest.raises(ValueError, match="Invalid accelerator"):
        MonroeFeaturizer(accelerator="not_a_real_device")


def test_checkpoint_path_must_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        MonroeFeaturizer(checkpoint_path=tmp_path / "nowhere")


def test_checkpoint_path_without_config_rejected(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"")

    with pytest.raises(ValueError, match="no config.json"):
        MonroeFeaturizer(checkpoint_path=tmp_path)


def test_checkpoint_path_without_weights_rejected(tmp_path):
    """A directory monroe would fail to load must be rejected at construction."""
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="no weights.pt"):
        MonroeFeaturizer(checkpoint_path=tmp_path)


def test_use_ema_without_checkpoint_rejected():
    """The published checkpoint ships no EMA weights, so this must fail before downloading."""
    with pytest.raises(ValueError, match="no EMA"):
        MonroeFeaturizer(use_ema=True)


@pytest.mark.parametrize(("field", "value"), [("n_workers", 0), ("batch_size", 0)])
def test_non_positive_worker_and_batch_counts_rejected(field, value):
    """Zero is a silently degenerate value for both, so reject it at the boundary."""
    with pytest.raises(ValueError):
        MonroeFeaturizer(**{field: value})


def test_featurize_shape_dtype_indices(smiles, featurizer_factory):
    featurizer, _ = featurizer_factory({"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0})

    features, indices = featurizer.featurize(smiles)

    assert features.shape == (3, 720)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(indices, np.arange(3))


def test_featurize_rows_follow_input_order(smiles, featurizer_factory):
    """monroe returns a SMILES-keyed dict, so row order must come from the input."""
    featurizer, _ = featurizer_factory({"c1ccccc1": 3.0, "CCO": 1.0, "CCN": 2.0})

    features, _ = featurizer.featurize(smiles)

    np.testing.assert_allclose(features[:, 0], [1.0, 2.0, 3.0])


def test_featurize_reports_dropped_molecules(smiles, featurizer_factory):
    """A molecule monroe omits must be absent from both the rows and the indices."""
    featurizer, _ = featurizer_factory({"CCO": 1.0, "c1ccccc1": 3.0})

    features, indices = featurizer.featurize(smiles)

    assert features.shape == (2, 720)
    np.testing.assert_array_equal(indices, np.array([0, 2]))
    np.testing.assert_allclose(features[:, 0], [1.0, 3.0])


def test_drop_warning_names_the_failed_smiles(smiles, featurizer_factory):
    """A bare count leaves the user unable to find the offending structure."""
    from loguru import logger

    featurizer, _ = featurizer_factory({"CCO": 1.0, "c1ccccc1": 3.0})

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        featurizer.featurize(smiles)
    finally:
        logger.remove(handler_id)

    assert any("CCN" in message for message in captured)


def test_featurize_all_dropped_returns_empty(smiles, featurizer_factory):
    featurizer, _ = featurizer_factory({})

    features, indices = featurizer.featurize(smiles)

    assert features.shape == (0, 720)
    assert indices.shape == (0,)


def test_featurize_duplicate_smiles_repeat_vector(featurizer_factory):
    """Duplicates collapse in monroe's dict, so each position must get its vector back."""
    featurizer, _ = featurizer_factory({"CCO": 1.0, "CCN": 2.0})

    features, indices = featurizer.featurize(["CCO", "CCN", "CCO"])

    assert features.shape == (3, 720)
    np.testing.assert_array_equal(indices, np.arange(3))
    np.testing.assert_allclose(features[:, 0], [1.0, 2.0, 1.0])


def test_duplicates_are_embedded_once(featurizer_factory):
    """Conformer generation dominates the cost, so repeats must not be sent twice."""
    featurizer, embed_mock = featurizer_factory({"CCO": 1.0, "CCN": 2.0})

    featurizer.featurize(["CCO", "CCN", "CCO", "CCO"])

    assert embed_mock.call_args.args[0] == ["CCO", "CCN"]


def test_featurize_always_passes_explicit_n_workers(smiles, featurizer_factory):
    """monroe falls back to os.sched_getaffinity when n_workers is None, which macOS lacks."""
    featurizer, embed_mock = featurizer_factory(
        {"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0}
    )

    featurizer.featurize(smiles)

    assert embed_mock.call_args.kwargs["n_workers"] >= 1


def test_featurize_resolves_accelerator_to_a_device(smiles, featurizer_factory):
    """The accelerator must reach monroe resolved, not passed through verbatim."""
    featurizer, embed_mock = featurizer_factory(
        {"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0}, accelerator="gpu"
    )

    featurizer.featurize(smiles)

    # "gpu" is a Lightning spelling torch.device() cannot parse, so monroe must
    # be handed the resolved torch name instead
    assert embed_mock.call_args.kwargs["device"] == "cuda"


def test_featurize_forwards_batch_size(smiles, featurizer_factory):
    featurizer, embed_mock = featurizer_factory(
        {"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0}, batch_size=7
    )

    featurizer.featurize(smiles)

    assert embed_mock.call_args.kwargs["batch_size"] == 7


def test_featurize_rejects_unexpected_width(smiles, featurizer_factory):
    """A width disagreeing with the checkpoint would silently reshape empty partitions."""
    featurizer, _ = featurizer_factory(
        {"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0}, dim=64
    )

    with pytest.raises(RuntimeError, match="64-wide"):
        featurizer.featurize(smiles)


def test_featurize_empty_input_returns_empty_without_encoder():
    """An empty input returns an empty matrix without downloading or building anything."""
    featurizer = MonroeFeaturizer(accelerator="cpu")

    features, indices = featurizer.featurize([])

    assert features.shape == (0, 720)
    np.testing.assert_array_equal(indices, np.empty(0, dtype=int))
    assert featurizer._encoder is None


def test_embedding_dim_read_from_local_checkpoint(local_checkpoint):
    """A local checkpoint's width comes from its config, not the published constant."""
    featurizer = MonroeFeaturizer(checkpoint_path=local_checkpoint)

    features, _ = featurizer.featurize([])

    assert featurizer.embedding_dim == 8
    assert features.shape == (0, 8)


def test_embedding_dim_rejects_unparseable_config(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    (tmp_path / "weights.pt").write_bytes(b"")

    with pytest.raises(ValueError, match="hidden_dim"):
        MonroeFeaturizer(checkpoint_path=tmp_path).embedding_dim


def test_encoder_uses_configured_checkpoint_and_ema(local_checkpoint, mocker):
    """Both documented options must reach monroe's loader."""
    load_mock = mocker.patch.object(
        monroe_module, "_load_encoder", autospec=True, return_value=object()
    )

    MonroeFeaturizer(checkpoint_path=local_checkpoint, use_ema=True).encoder

    load_mock.assert_called_once_with(local_checkpoint, True)


def test_encoder_downloads_when_no_checkpoint_configured(mocker, tmp_path):
    download_mock = mocker.patch.object(
        monroe_module,
        "_download_monroe_checkpoint",
        autospec=True,
        return_value=tmp_path / "downloaded",
    )
    load_mock = mocker.patch.object(
        monroe_module, "_load_encoder", autospec=True, return_value=object()
    )

    MonroeFeaturizer().encoder

    download_mock.assert_called_once()
    assert load_mock.call_args.args[0] == tmp_path / "downloaded"


def test_download_verifies_and_marks_the_cache(tmp_path, pin_artifact, fake_download):
    payload = b"monroe weights"
    pin_artifact(payload)
    fake_download(config=b"{}", weights=payload)

    ckpt_dir = _download_monroe_checkpoint(cache_root=tmp_path)

    assert (ckpt_dir / "weights.pt").read_bytes() == payload
    assert (ckpt_dir / _VERIFIED_MARKER).read_text() == hashlib.sha256(
        payload
    ).hexdigest()
    assert not list(ckpt_dir.glob("*.part"))


def test_download_reuses_verified_cache(tmp_path, mocker, pin_artifact):
    payload = b"monroe weights"
    pin_artifact(payload)

    ckpt_dir = _cache_dir(tmp_path)
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "weights.pt").write_bytes(payload)
    (ckpt_dir / _VERIFIED_MARKER).write_text(hashlib.sha256(payload).hexdigest())

    urlretrieve_mock = mocker.patch.object(monroe_module, "urlretrieve", autospec=True)

    assert _download_monroe_checkpoint(cache_root=tmp_path) == ckpt_dir
    urlretrieve_mock.assert_not_called()


def test_download_redownloads_unverified_cache(tmp_path, pin_artifact, fake_download):
    """Right-sized weights with no marker are not evidence of a good download."""
    payload = b"monroe weights"
    pin_artifact(payload)

    ckpt_dir = _cache_dir(tmp_path)
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "weights.pt").write_bytes(b"stale but righ")

    fake_download(config=b"{}", weights=payload)

    _download_monroe_checkpoint(cache_root=tmp_path)

    assert (ckpt_dir / "weights.pt").read_bytes() == payload


def test_download_redownloads_when_weights_replaced_after_marking(
    tmp_path, pin_artifact, fake_download
):
    """A marker must not vouch for weights that were swapped on disk afterwards."""
    payload = b"monroe weights"
    pin_artifact(payload)

    ckpt_dir = _cache_dir(tmp_path)
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "weights.pt").write_bytes(b"a different length payload entirely")
    (ckpt_dir / _VERIFIED_MARKER).write_text(hashlib.sha256(payload).hexdigest())

    fake_download(config=b"{}", weights=payload)

    _download_monroe_checkpoint(cache_root=tmp_path)

    assert (ckpt_dir / "weights.pt").read_bytes() == payload


def test_download_rejects_corrupt_weights(tmp_path, pin_artifact, fake_download):
    """A digest mismatch must raise, leave no scratch file, and mark nothing verified."""
    pin_artifact(b"the real weights")
    fake_download(config=b"{}", weights=b"not the weights")

    with pytest.raises(RuntimeError, match="is corrupt"):
        _download_monroe_checkpoint(cache_root=tmp_path)

    ckpt_dir = _cache_dir(tmp_path)
    assert not list(ckpt_dir.glob("*.part"))
    assert not (ckpt_dir / "weights.pt").exists()
    assert not (ckpt_dir / _VERIFIED_MARKER).exists()


def test_missing_monroe_raises_pinned_install_hint(mocker):
    """Without monroe installed, the error must name the pinned git install command."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("monroe"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    mocker.patch.object(builtins, "__import__", side_effect=_blocked)

    with pytest.raises(ImportError, match=monroe_module._MONROE_CKPT_COMMIT):
        monroe_module._import_monroe()


def test_featurizer_is_registered():
    assert get_featurizer_class("MonroeFeaturizer") is MonroeFeaturizer


def test_featurizer_compatible_with_concatenator(smiles, featurizer_factory):
    monroe_featurizer, _ = featurizer_factory({"CCO": 1.0, "CCN": 2.0, "c1ccccc1": 3.0})
    concat = FeatureConcatenator(
        featurizers=[
            monroe_featurizer,
            FingerprintFeaturizer(fp_type="ecfp:4", n_jobs=1),
        ]
    )

    X, idx = concat.featurize(smiles)

    # ECFP 2000 + Monroe 720, in the class-name order the concatenator sorts into
    assert X.shape == (3, 2720)
    np.testing.assert_array_equal(idx, np.arange(3))


def test_concatenator_intersects_monroe_drops(smiles, featurizer_factory):
    """A molecule Monroe drops must disappear from the concatenated matrix too."""
    monroe_featurizer, _ = featurizer_factory({"CCO": 1.0, "c1ccccc1": 3.0})
    concat = FeatureConcatenator(
        featurizers=[
            monroe_featurizer,
            FingerprintFeaturizer(fp_type="ecfp:4", n_jobs=1),
        ]
    )

    X, idx = concat.featurize(smiles)

    assert X.shape == (2, 2720)
    np.testing.assert_array_equal(idx, np.array([0, 2]))


@pytest.fixture
def tiny_monroe_checkpoint(tmp_path):
    """A real, randomly initialized Monroe encoder small enough to run in a unit test."""
    pytest.importorskip("monroe")

    import torch
    from monroe.model.constants import (
        EDGE_FEAT_LIST_ONE_HOT,
        NODE_FEAT_LIST_FLOAT,
        NODE_FEAT_LIST_ONE_HOT,
    )
    from monroe.model.grit import GritTransformer

    encoder_cfg = {
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "emb_dim": 8,
        "walk_len": 4,
        "rbf_dim": 4,
        "dropout": 0.0,
    }
    encoder = GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **encoder_cfg,
    )

    ckpt_dir = tmp_path / "tiny_monroe"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps({"encoder": encoder_cfg}))
    torch.save(
        {f"encoder.{k}": v for k, v in encoder.state_dict().items()},
        ckpt_dir / "weights.pt",
    )

    return ckpt_dir


def test_real_load_ckpt_builds_encoder(tiny_monroe_checkpoint):
    """The real monroe loader must accept the checkpoint layout this featurizer hands it."""
    from monroe.model.grit import GritTransformer

    loaded = monroe_module._load_encoder(tiny_monroe_checkpoint, use_ema=False)

    assert isinstance(loaded, GritTransformer)
    assert MonroeFeaturizer(checkpoint_path=tiny_monroe_checkpoint).embedding_dim == 16


def test_real_embed_smiles_contract(tiny_monroe_checkpoint):
    """
    Pin the upstream contract the patched tests above assume.

    The wrapper identifies molecules by raw input string, so a change in monroe's
    key normalization would silently empty every result. This is the only test
    that can catch it.
    """
    featurizer = MonroeFeaturizer(
        checkpoint_path=tiny_monroe_checkpoint, accelerator="cpu", n_workers=1
    )

    embedded = monroe_module._embed_smiles(
        ["CCO", "c1ccccc1"],
        featurizer.encoder,
        device="cpu",
        batch_size=2,
        n_workers=1,
    )

    # Keyed by the verbatim input string, one 1D vector of the config width each
    assert set(embedded) == {"CCO", "c1ccccc1"}
    for vector in embedded.values():
        assert vector.shape == (16,)


def test_real_featurize_end_to_end(tiny_monroe_checkpoint):
    """The full path must produce the documented (features, indices) shapes."""
    featurizer = MonroeFeaturizer(
        checkpoint_path=tiny_monroe_checkpoint, accelerator="cpu", n_workers=1
    )

    features, indices = featurizer.featurize(["CCO", "c1ccccc1"])

    assert features.shape == (2, 16)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(indices, np.arange(2))
