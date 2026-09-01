from types import SimpleNamespace

import pytest
from PIL import Image

from tools.export_local_reconstruction import select_median_psnr_index, validated_image


def test_validated_image_uses_checkpoint_validation_dimensions() -> None:
    config = SimpleNamespace(validation_height=480, validation_width=640, ps=[32, 16])

    result = validated_image(Image.new("RGB", (800, 600)), config)

    assert result.mode == "RGB"
    assert result.size == (640, 480)


def test_validated_image_rejects_unaligned_unresized_inputs() -> None:
    config = SimpleNamespace(validation_height=None, validation_width=None, ps=[32])

    with pytest.raises(ValueError, match="max patch size 32"):
        validated_image(Image.new("RGB", (63, 32)), config)


def test_representative_selection_uses_median_and_stable_tie_breaking() -> None:
    index, median = select_median_psnr_index([5.0, 10.0, 12.0, 19.0])

    assert median == 11.0
    assert index == 1
