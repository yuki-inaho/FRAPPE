"""Small, dependency-light experiment tracking helpers for FRAPPE training."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: object, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: object, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".json", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class KBestCheckpointManager:
    """Retain the best K checkpoint payloads and a resumable JSON index."""

    def __init__(self, directory: str | Path, k: int = 3, mode: str = "max") -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = Path(directory)
        self.k = k
        self.mode = mode
        self.index_path = self.directory / "index.json"
        self.entries: list[dict[str, Any]] = []
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            if raw.get("mode") != mode:
                raise ValueError("existing K-best index uses a different mode")
            self.entries = [entry for entry in raw.get("entries", [])
                            if (self.directory / entry["path"]).is_file()]
            self._sort()

    def _key(self, entry: dict[str, Any]) -> tuple[float, int]:
        score = float(entry["score"])
        step = int(entry["step"])
        return (score, step) if self.mode == "max" else (-score, step)

    def _sort(self) -> None:
        self.entries.sort(key=self._key, reverse=True)

    def _write_index(self) -> None:
        atomic_json_dump({"mode": self.mode, "keep_best_k": self.k,
                          "entries": self.entries}, self.index_path)

    def consider(self, score: float, step: int, payload: object) -> bool:
        if self.k == 0:
            return False
        score = float(score)
        if not math.isfinite(score):
            return False
        candidate = {"score": score, "step": int(step)}
        if len(self.entries) >= self.k and self._key(candidate) <= self._key(self.entries[-1]):
            return False

        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"best_step{step:04d}_score{score:.6f}.pth.tar"
        candidate["path"] = filename
        atomic_torch_save(payload, self.directory / filename)
        self.entries.append(candidate)
        self._sort()

        while len(self.entries) > self.k:
            removed = self.entries.pop()
            target = (self.directory / removed["path"]).resolve()
            if target.parent != self.directory.resolve():
                raise RuntimeError("refusing to remove a checkpoint outside the K-best directory")
            target.unlink(missing_ok=True)
        self._write_index()
        return True


class TensorBoardTracker:
    """Rank-zero style TensorBoard writer with a no-op disabled mode."""

    def __init__(self, log_dir: str | Path | None, enabled: bool = True) -> None:
        self.writer = None
        self.global_step = 0
        if enabled and log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def scalar(self, name: str, value: float, step: int | None = None) -> None:
        if self.writer is not None and math.isfinite(float(value)):
            self.writer.add_scalar(name, float(value), self.global_step if step is None else step)

    def next_step(self) -> int:
        current = self.global_step
        self.global_step += 1
        return current

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class ModelEMA:
    """Parameter EMA kept on the active device for inexpensive training updates."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in the open interval (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone(memory_format=torch.preserve_format)
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(value):
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            value.copy_(self.shadow[name])

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}

    @torch.no_grad()
    def restore(self, snapshot: dict[str, torch.Tensor]) -> None:
        for name, value in self.shadow.items():
            value.copy_(snapshot[name].to(device=value.device, dtype=value.dtype))


class EarlyStopping:
    """Maximize a validation metric and restore the best epoch's weights."""

    def __init__(self, patience: int, min_delta: float = 0.0, min_epochs: int = 1) -> None:
        if patience < 1:
            raise ValueError("early-stopping patience must be at least one")
        if min_delta < 0.0:
            raise ValueError("early-stopping min_delta must be non-negative")
        if min_epochs < 1:
            raise ValueError("early-stopping min_epochs must be at least one")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.min_epochs = min_epochs
        self.best_score = float("-inf")
        self.best_epoch: int | None = None
        self.bad_epochs = 0
        self._model_snapshot: dict[str, torch.Tensor] | None = None
        self._ema_snapshot: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _snapshot_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def step(self, score: float, epoch: int, model: torch.nn.Module,
             ema: ModelEMA | None = None) -> bool:
        if not math.isfinite(float(score)):
            self.bad_epochs += 1
            return epoch + 1 >= self.min_epochs and self.bad_epochs >= self.patience
        if float(score) > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = epoch
            self.bad_epochs = 0
            self._model_snapshot = self._snapshot_model(model)
            self._ema_snapshot = ema.snapshot() if ema is not None else None
            return False
        self.bad_epochs += 1
        return epoch + 1 >= self.min_epochs and self.bad_epochs >= self.patience

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, ema: ModelEMA | None = None) -> bool:
        if self._model_snapshot is None:
            return False
        model.load_state_dict(self._model_snapshot)
        if ema is not None and self._ema_snapshot is not None:
            ema.restore(self._ema_snapshot)
        return True
