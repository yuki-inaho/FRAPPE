import torch

from src.compressors.frappe.experiment import ModelEMA
from src.compressors.frappe.third_party.amuse import AMUSE
from train_rae_progressive import make_amuse_optimizer


def test_amuse_official_train_eval_transition() -> None:
    matrix = torch.nn.Parameter(torch.eye(4))
    bias = torch.nn.Parameter(torch.zeros(4))
    optimizer = AMUSE([
        {
            'params': [matrix], 'use_muon': True, 'lr': 0.01,
            'momentum': 0.95, 'aux_update_type': 'adamw',
        },
        {
            'params': [bias], 'use_muon': False, 'update_type': 'adamw',
            'lr': 0.01,
        },
    ], beta1=0.8, warmup_steps=1, rho=0.3)
    optimizer.train()
    loss = (matrix.square().sum() + bias.square().sum())
    loss.backward()
    optimizer.step()
    assert optimizer.train_mode
    assert 'z' in optimizer.state[matrix]
    optimizer.eval()
    assert not optimizer.train_mode
    optimizer.train()


def test_frappes_amuse_grouping_splits_matrix_and_bias() -> None:
    model = torch.nn.Linear(4, 3)

    class Config:
        amuse_warmup_ratio = 0.1
        amuse_muon_min_ndim = 2
        amuse_weight_decay = 0.0
        amuse_momentum = 0.95
        amuse_aux_update_type = 'adamw'
        amuse_beta2 = 0.999
        amuse_eps = 1e-10
        amuse_beta = 0.8
        amuse_rho = 0.3
        amuse_r = 0.0
        amuse_weight_lr_power = 2.0
        amuse_weight_decay_at_y = 0.0

    optimizer = make_amuse_optimizer([
        {'params': list(model.parameters()), 'lr': 0.5},
    ], max_lr=0.02, total_steps=10, config=Config())
    assert {group['update_type'] for group in optimizer.param_groups} == {'muon', 'adamw'}
    assert {group['base_lr'] for group in optimizer.param_groups} == {0.01}


def test_frappes_amuse_accepts_a_bare_parameter_list() -> None:
    model = torch.nn.Linear(4, 3)

    class Config:
        amuse_warmup_ratio = 0.1
        amuse_muon_min_ndim = 2
        amuse_weight_decay = 0.0
        amuse_momentum = 0.95
        amuse_aux_update_type = 'adamw'
        amuse_beta2 = 0.999
        amuse_eps = 1e-10
        amuse_beta = 0.8
        amuse_rho = 0.3
        amuse_r = 0.0
        amuse_weight_lr_power = 2.0
        amuse_weight_decay_at_y = 0.0

    optimizer = make_amuse_optimizer(
        list(model.parameters()), max_lr=0.02, total_steps=10, config=Config())
    assert len(optimizer.param_groups) == 2


def test_ema_tracks_floating_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, 0.99)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    ema.copy_to(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 1.02), atol=1e-6)
