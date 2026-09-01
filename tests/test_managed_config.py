import json

from hydra import compose, initialize_config_dir
from pathlib import Path

from train_managed import build_argv


def test_hydra_config_builds_backward_compatible_cli() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            "run.id=test_run",
            "training.max_batches_per_epoch=1",
            "logging.tensorboard=false",
        ])
    argv = build_argv(cfg)
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--train_ds") + 1].endswith("data/frappe_rgb_640x480/imagefolder")
    assert argv[argv.index("--keep_best_k") + 1] == "3"
    assert argv[argv.index("--max_batches_per_epoch") + 1] == "1"
    assert "--ps" in argv
    assert argv[argv.index("--optimizer") + 1] == "amuse"
    assert argv[argv.index("--min_width") + 1] == "640"
    assert argv[argv.index("--validation_height") + 1] == "480"
    assert argv[argv.index("--iterations_single") + 1] == "106"
    assert argv[argv.index("--validation_every_iterations") + 1] == "53"
    augmentation = json.loads(argv[argv.index("--augmentation_config") + 1])
    assert augmentation["transforms"]["channel_shuffle"]["name"] == "ChannelShuffle"


def test_amuse_ema_experiment_preset() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=amuse_ema"])
    assert cfg.optimization.optimizer == "amuse"
    assert cfg.optimization.ema_decay == 0.99


def test_augmentation_profile_can_be_selected_modularly() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["augmentation=rgb_strong"])
    assert cfg.augmentation.transforms.channel_shuffle.p == 0.25
    assert cfg.augmentation.transforms.to_gray.name == "ToGray"


def test_iteration_trial_preset_is_fixed_update_based() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=iteration_trial_30m"])
    assert cfg.model.ps == [32]
    assert cfg.training.iterations_single == [200]
    assert cfg.training.iterations_merged == [500]
    assert cfg.validation.every_iterations == 125


def test_21ch_eight_hour_preset_selects_its_model_data_and_geometry() -> None:
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["experiment=iteration_21ch_8h"])
    assert len(cfg.model.ps) == 21
    assert cfg.model.recommended_batch_size == 12
    assert cfg.data.root.endswith("data/frappe_rgb_800x608/imagefolder")
    assert (cfg.augmentation.validation_width, cfg.augmentation.validation_height) == (800, 608)
    assert cfg.training.iterations_single == [100]
    assert cfg.training.iterations_merged == [220]
