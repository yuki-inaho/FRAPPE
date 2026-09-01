# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/kjeiun/amuse at
# 48922743b32f33f919ab54edde3dbad0d0ce2dc7 (src/optim/AMUSE.py).
# The upstream repository is Apache-2.0 licensed. This file is kept local
# because upstream is not published as an installable Python package.

import torch
import torch.optim


UPDATE_TYPES = {"muon", "adamw", "sgd"}
AUX_UPDATE_TYPES = {"adamw", "sgd"}


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.mT if transposed else X


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    aux_update_type: str = "adamw",
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    if aux_update_type == "adamw":
        update *= 0.2 * max(update.size(0), update.size(1)) ** 0.5
    elif aux_update_type == "sgd":
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    else:
        raise ValueError("AMUSE aux_update_type must be 'adamw' or 'sgd'")
    return update


class AMUSE(torch.optim.Optimizer):
    """Anytime Muon with Stable gradient Evaluation.

    The implementation and state convention are from the pinned upstream
    source. Parameters store the gradient-evaluation sequence while in
    ``train`` mode and the averaged sequence while in ``eval`` mode.
    """

    def __init__(
        self,
        param_groups,
        *,
        weight_decay_at_y: float = 0.0,
        beta1: float = 0.9,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        rho: float = 1.0,
        r: float = 0.0,
    ):
        if warmup_steps <= 0:
            raise ValueError("AMUSE requires warmup_steps > 0.")
        self.weight_decay_at_y = weight_decay_at_y
        self.beta1_init = float(beta1)
        self.weight_lr_power = weight_lr_power
        self.warmup_steps = int(warmup_steps)
        self.rho = float(rho)
        self.r = r
        self.train_mode = False

        super().__init__(param_groups, defaults={})
        for group in self.param_groups:
            group.setdefault("warmup_steps", self.warmup_steps)
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("use_muon", False)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("beta1", self.beta1_init)
            update_type = "muon" if group["use_muon"] else group.get("update_type", "adamw")
            if update_type not in UPDATE_TYPES:
                raise ValueError("Invalid AMUSE update_type")
            if update_type == "muon" and not group["use_muon"]:
                raise ValueError('AMUSE update_type="muon" requires use_muon=True.')
            group["update_type"] = update_type

            if update_type == "muon":
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("aux_update_type", "adamw")
                if group["aux_update_type"] not in AUX_UPDATE_TYPES:
                    raise ValueError("Invalid AMUSE aux_update_type")
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                for p in group["params"]:
                    self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
            elif update_type == "adamw":
                group.setdefault("lr", 3e-4)
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-10)
                for p in group["params"]:
                    self.state[p].setdefault("exp_avg_sq", torch.zeros_like(p))
            elif update_type == "sgd":
                group.setdefault("lr", 1.0)
            group["base_lr"] = group["lr"]

    def _compute_beta1(self, group, t, ckp1, warmup_steps):
        if t <= warmup_steps:
            if t == warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init
        c_warmup = group.get("c_warmup", 1.0 / warmup_steps)
        s_t = (ckp1 * (1.0 - c_warmup)) / (c_warmup * (1.0 - ckp1))
        return 1.0 - (s_t ** self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p):
        state = self.state[p]
        if "z" not in state:
            state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        return state["z"]

    def _apply_weight_decay_at_y(self, p, z, lr, beta1):
        if self.weight_decay_at_y != 0.0:
            z.sub_(p, alpha=lr * self.weight_decay_at_y)
            p.sub_(p, alpha=lr * self.weight_decay_at_y * (1.0 - beta1))

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(end=self.state[p]["z"], weight=1.0 - 1.0 / beta1)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(end=self.state[p]["z"], weight=1.0 - beta1)
        self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise RuntimeError("AMUSE must be switched to train() before step().")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            k = group["k"]
            warmup_steps = group.get("warmup_steps", self.warmup_steps)
            if warmup_steps <= 0:
                raise ValueError("AMUSE requires warmup_steps > 0.")
            t = k + 1
            lr = group["base_lr"] * min(1.0, t / warmup_steps)
            group["lr"] = lr
            weight = (t ** self.r) * (lr ** self.weight_lr_power)
            future_weight_sum = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / future_weight_sum if future_weight_sum > 0 else 1.0
            group["ckp1"] = ckp1
            group["weight_sum"] = future_weight_sum
            beta1 = self._compute_beta1(group, t, ckp1, warmup_steps)
            group["beta1"] = beta1
            group["r_t"] = ckp1 / ((1.0 - beta1) + beta1 * ckp1 + 1e-12)
            self.beta1 = beta1
            wd = group.get("weight_decay", 0.0)

            if group["update_type"] == "muon":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    momentum = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    update = muon_update(
                        p.grad, momentum, beta=group["momentum"],
                        aux_update_type=group.get("aux_update_type", "adamw"), nesterov=True)
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(update.reshape(p.shape), alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)
            elif group["update_type"] == "adamw":
                beta2 = group.get("beta2", 0.999)
                eps = group.get("eps", 1e-10)
                bias_correction2 = 1.0 - beta2 ** t
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    v = state.setdefault("exp_avg_sq", torch.zeros_like(p))
                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
                    update = p.grad / v.div(bias_correction2).sqrt_().add_(eps)
                    if wd != 0.0:
                        update = update.add(z, alpha=wd)
                    z.add_(update, alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)
            elif group["update_type"] == "sgd":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(p.grad, alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)
            else:
                raise ValueError("Invalid AMUSE update_type")
            group["k"] = k + 1
        return loss
