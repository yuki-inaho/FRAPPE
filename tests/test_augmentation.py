import numpy as np
from PIL import Image

from src.compressors.frappe.augmentation import RGBTrainingAugmentation


def test_configured_flips_are_applied_after_geometry() -> None:
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    augmentation = RGBTrainingAugmentation({
        "transforms": [
            {"name": "HorizontalFlip", "p": 1.0},
            {"name": "VerticalFlip", "p": 1.0},
        ],
    })
    actual = augmentation(Image.fromarray(source), height=4, width=5, scale=1.0, max_size=5)
    assert np.array_equal(actual, source[::-1, ::-1])


def test_colour_space_and_channel_augmentations_keep_rgb_uint8_shape() -> None:
    source = np.full((8, 8, 3), (40, 100, 180), dtype=np.uint8)
    augmentation = RGBTrainingAugmentation({
        "transforms": [
            {"name": "HueSaturationValue", "hue_shift_limit": 10,
             "sat_shift_limit": 20, "val_shift_limit": 10, "p": 1.0},
            {"name": "ChannelShuffle", "p": 1.0},
        ],
    })
    actual = augmentation(Image.fromarray(source), height=8, width=8, scale=1.0, max_size=8)
    assert actual.shape == (8, 8, 3)
    assert actual.dtype == np.uint8


def test_landscape_640_by_480_crop_is_supported() -> None:
    source = np.full((600, 800, 3), 127, dtype=np.uint8)
    augmentation = RGBTrainingAugmentation({"transforms": []})
    actual = augmentation(Image.fromarray(source), height=480, width=640, scale=1.0, max_size=960)
    assert actual.shape == (480, 640, 3)


def test_invalid_transform_is_rejected_before_training() -> None:
    try:
        RGBTrainingAugmentation({"transforms": [{"name": "Tyop", "p": 1.0}]})
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("invalid transform should fail configuration validation")
