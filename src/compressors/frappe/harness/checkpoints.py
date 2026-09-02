"""Loading a joint-prefix checkpoint back into a model.

A checkpoint carries the configuration that built the model as well as its
weights, so reconstructing it is mechanical -- and every tool was doing that
mechanical thing slightly differently, some discarding the config, some the
iteration number. One loader returns all three.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import torch

from ..prefix import JointPrefixFRAPPE


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A model together with the provenance needed to report on it."""

    model: JointPrefixFRAPPE
    config: Namespace
    iteration: int | None
    path: Path
    state: dict

    @property
    def channels(self) -> int:
        return self.model.n_channels

    def describe(self) -> str:
        return (f"{self.path} (iteration {self.iteration}), "
                f"{self.channels} channels, ps={list(self.config.ps)}")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                    eval_mode: bool = True) -> LoadedCheckpoint:
    """Rebuild the model a checkpoint describes and load its weights."""
    path = Path(path)
    state = torch.load(path, map_location=device, weights_only=False)
    config = Namespace(**state["config"])
    model = JointPrefixFRAPPE(config).to(device)
    model.load_state_dict(state["model"])
    if eval_mode:
        model.eval()
    return LoadedCheckpoint(model=model, config=config,
                            iteration=state.get("iteration"), path=path, state=state)
