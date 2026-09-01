from pathlib import Path

import torch

from src.compressors.frappe.experiment import (
    EarlyStopping,
    KBestCheckpointManager,
    TensorBoardTracker,
    atomic_torch_save,
)


def test_atomic_torch_save(tmp_path: Path) -> None:
    target = tmp_path / "nested/checkpoint.pth.tar"
    atomic_torch_save({"value": torch.tensor([3])}, target)
    assert torch.load(target, weights_only=True)["value"].item() == 3
    assert not list(target.parent.glob(".checkpoint.pth.tar.*"))


def test_kbest_max_retains_only_k_and_survives_restart(tmp_path: Path) -> None:
    manager = KBestCheckpointManager(tmp_path / "best", k=2, mode="max")
    assert manager.consider(1.0, 1, {"step": 1})
    assert manager.consider(3.0, 2, {"step": 2})
    assert not manager.consider(0.5, 3, {"step": 3})
    assert manager.consider(2.0, 4, {"step": 4})

    assert [entry["score"] for entry in manager.entries] == [3.0, 2.0]
    assert len(list((tmp_path / "best").glob("*.pth.tar"))) == 2

    resumed = KBestCheckpointManager(tmp_path / "best", k=2, mode="max")
    assert [entry["step"] for entry in resumed.entries] == [2, 4]


def test_kbest_min_and_newer_tie_wins(tmp_path: Path) -> None:
    manager = KBestCheckpointManager(tmp_path / "best", k=1, mode="min")
    manager.consider(2.0, 1, {"step": 1})
    manager.consider(1.0, 2, {"step": 2})
    manager.consider(1.0, 3, {"step": 3})
    assert manager.entries == [{
        "score": 1.0,
        "step": 3,
        "path": "best_step0003_score1.000000.pth.tar",
    }]


def test_tensorboard_tracker_writes_an_event(tmp_path: Path) -> None:
    tracker = TensorBoardTracker(tmp_path / "tensorboard")
    tracker.scalar("validation/psnr", 31.5, step=1)
    tracker.close()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


def test_early_stopping_restores_the_best_model_and_ema() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    from src.compressors.frappe.experiment import ModelEMA
    ema = ModelEMA(model, decay=0.9)
    stopper = EarlyStopping(patience=1, min_delta=0.0, min_epochs=2)

    assert not stopper.step(30.0, 0, model, ema)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)
    assert stopper.step(29.0, 1, model, ema)

    assert stopper.restore(model, ema)
    assert stopper.best_epoch == 0
    assert model.weight.item() == 1.0
    assert ema.shadow["weight"].item() == 1.0


def test_early_stopping_threshold_does_not_restore_subthreshold_weights() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    stopper = EarlyStopping(patience=1, min_delta=0.0, min_epochs=1, min_score=40.0)

    assert not stopper.step(39.9, 0, model)
    assert stopper.best_epoch is None
    assert not stopper.restore(model)

    with torch.no_grad():
        model.weight.fill_(2.0)
    assert not stopper.step(40.0, 1, model)
    assert stopper.threshold_reached
    assert stopper.best_epoch == 1

    with torch.no_grad():
        model.weight.fill_(3.0)
    assert not stopper.step(39.9, 2, model)
    assert stopper.restore(model)
    assert model.weight.item() == 2.0
