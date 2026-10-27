"""Smoke tests that torch and Lightning run on every accelerator the install supports."""

import os
from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ACCELERATORS = ["cpu", "mps", "cuda"]


def _is_available(accelerator: str) -> bool:
    if accelerator == "cpu":
        return True
    if accelerator == "mps":
        return torch.backends.mps.is_available()
    return torch.cuda.is_available()


class _TinyRegressor(L.LightningModule):
    """Linear regressor that records the device type seen by each training step."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.seen_device_types: list[str] = []

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features, targets = batch
        self.seen_device_types.append(features.device.type)
        return torch.nn.functional.mse_loss(self.layer(features), targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.fixture(params=ACCELERATORS)
def accelerator(request: pytest.FixtureRequest) -> str:
    """Accelerator under test, skipped when unavailable unless the environment requires it."""
    name = request.param
    required = os.environ.get("OPENADMET_EXPECT_DEVICE", "")
    if _is_available(name):
        return name
    if name == required:
        pytest.fail(f"{name} is required by OPENADMET_EXPECT_DEVICE but torch cannot use it")
    pytest.skip(f"{name} is not available in this environment")


def test_tensor_matmul_on_device_matches_cpu(accelerator: str):
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(8, 4, generator=generator)
    right = torch.randn(4, 3, generator=generator)

    on_device = (left.to(accelerator) @ right.to(accelerator)).cpu()

    torch.testing.assert_close(on_device, left @ right, rtol=1e-4, atol=1e-5)


def test_lightning_fit_runs_on_requested_accelerator(accelerator: str, tmp_path: Path):
    L.seed_everything(0)
    features = torch.randn(32, 4)
    targets = features.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(features, targets), batch_size=16)
    model = _TinyRegressor()
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, loader)

    assert set(model.seen_device_types) == {accelerator}
