import pytest

from train_rae_progressive import parse_args


def test_original_required_cli_remains_supported() -> None:
    args = parse_args(["--device", "cpu", "--ps", "32", "16"])
    assert args.device == "cpu"
    assert args.ps == [32, 16]
    assert args.keep_best_k == 0
    assert args.tensorboard is False


def test_managed_options_parse() -> None:
    args = parse_args([
        "--device", "cuda:0", "--ps", "32", "--run_dir", "runs/test",
        "--tensorboard", "true", "--keep_best_k", "3", "--seed", "7",
        "--validation_samples", "2", "--max_batches_per_epoch", "1",
        "--optimizer", "amuse", "--ema_decay", "0.99",
        "--min_width", "640", "--max_width", "640",
        "--validation_height", "480", "--validation_width", "640",
        "--iterations_single", "10", "--iterations_merged", "20",
        "--validation_every_iterations", "5",
    ])
    assert args.tensorboard is True
    assert args.keep_best_k == 3
    assert args.validation_samples == 2
    assert args.max_batches_per_epoch == 1
    assert args.optimizer == "amuse"
    assert args.ema_decay == 0.99
    assert (args.min_width, args.max_width) == (640, 640)
    assert (args.validation_height, args.validation_width) == (480, 640)
    assert args.iterations_single == [10]
    assert args.iterations_merged == [20]
    assert args.validation_every_iterations == 5


def test_ps_is_still_required() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--device", "cpu"])
