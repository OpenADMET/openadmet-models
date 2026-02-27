from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from openadmet.models.architecture.kermt import KERMTRegressorModel


@pytest.fixture
def smiles_y():
    smiles = np.asarray(["CCO", "CCN", "CCC"], dtype=object)
    y = np.asarray([0.1, 0.2, 0.3], dtype=float)
    return smiles, y


def test_kermt_model_type():
    model = KERMTRegressorModel()
    assert model.type == "KERMTRegressorModel"


def test_kermt_split_size_validation():
    with pytest.raises(ValueError, match="split_sizes must sum to 1.0"):
        KERMTRegressorModel(split_sizes=(0.7, 0.2, 0.2))


def test_kermt_train_predict_and_serialize(monkeypatch, tmp_path, smiles_y):
    smiles, y = smiles_y
    expected = np.asarray([0.11, 0.22, 0.33])

    repo = tmp_path / "KERMT"
    repo.mkdir()
    (repo / "main.py").write_text("print('kermt')\n")

    def fake_run(args, cwd, env, check, text, capture_output):
        if "finetune" in args:
            save_dir = Path(args[args.index("--save_dir") + 1])
            ckpt = save_dir / "fold_0" / "model_0"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "model.pt").write_bytes(b"checkpoint-bytes")
        elif "predict" in args:
            output_csv = Path(args[args.index("--output_path") + 1])
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"target": expected}, index=smiles).to_csv(output_csv)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "openadmet.models.architecture.kermt.subprocess.run",
        fake_run,
    )

    model = KERMTRegressorModel(
        kermt_repo_path=repo.as_posix(),
        epochs=1,
        batch_size=2,
        no_cuda=True,
    )
    model.train(smiles, y)
    preds = model.predict(smiles)
    assert preds.shape == (3, 1)
    assert_allclose(preds[:, 0], expected)

    model.serialize(tmp_path / "kermt.json", tmp_path / "kermt.pkl")
    reloaded = KERMTRegressorModel.deserialize(
        tmp_path / "kermt.json", tmp_path / "kermt.pkl"
    )
    preds_reloaded = reloaded.predict(smiles)
    assert_allclose(preds_reloaded[:, 0], expected)
