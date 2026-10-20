from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from src.preprocessing import load_image, preprocess_image, validate_preprocessing


def test_rgb_resize_and_pixel_contract():
    pixels = preprocess_image(Image.new("L", (40, 80), 128))
    assert pixels.shape == (224, 224, 3)
    assert pixels.dtype == np.float32
    assert np.all(pixels == 128)  # No external /255 for EfficientNet.


def test_alpha_composites_on_white():
    image = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    assert np.all(preprocess_image(image) == 255)


@pytest.mark.parametrize("raw", [b"not an image", b"", b"\xff\xd8\xffbroken"])
def test_invalid_bytes(raw):
    with pytest.raises(ValueError, match="corrupted"):
        load_image(raw)


def test_real_format_is_checked():
    output = BytesIO()
    Image.new("RGB", (32, 32)).save(output, format="GIF")
    with pytest.raises(ValueError, match="JPEG and PNG"):
        load_image(output.getvalue())


def test_too_small():
    with pytest.raises(ValueError, match="too small"):
        load_image(Image.new("RGB", (4, 4)))


def test_bad_preprocessing_metadata():
    with pytest.raises(ValueError):
        validate_preprocessing({"pixel_range": [0, 1]})
