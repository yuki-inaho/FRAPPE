from types import SimpleNamespace

import torch

from train_rae_progressive import train_one


def test_iteration_mode_has_exact_update_and_validation_cadence() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model._target = lambda _input, target: target
    loader = torch.utils.data.DataLoader(torch.ones(3, 1), batch_size=1, shuffle=False)
    config = SimpleNamespace(
        min_lr=1.0e-8,
        optimizer="adan",
        ema_decay=0.0,
        grad_clip=1.5,
        max_batches_per_epoch=None,
        linear_input=False,
    )
    monitor_steps: list[int] = []
    losses = train_one(
        model, list(model.parameters()), loader, "cpu", config,
        total_steps=5, epochs=1, label="toy", max_lr=1.0e-3,
        iterations=5, validation_every_iterations=2,
        epoch_evaluator=lambda step, _model: monitor_steps.append(step) or 1.0,
    )
    assert len(losses) == 5
    assert monitor_steps == [2, 4, 5]
