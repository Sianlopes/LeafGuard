"""One RGB/resize contract for both training and uploaded images.

Both networks accept float32 RGB pixels in [0,255]. Rescaling is INSIDE
the saved model (EfficientNet's own preprocessing, CNN Rescaling layer).
"""
from io import BytesIO
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
FORMATS = {"JPEG", "PNG"}


def load_image(source):
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            if isinstance(source, (str, Path)):
                if Path(source).stat().st_size > MAX_BYTES:
                    raise ValueError("Image exceeds the 10 MB limit.")
                raw = Path(source).read_bytes()
            elif isinstance(source, bytes):
                raw = source
            elif hasattr(source, "getvalue"):
                raw = source.getvalue()
            else:
                raw = source.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Image exceeds the 10 MB limit.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(BytesIO(raw))
                if image.format not in FORMATS:
                    raise ValueError("Only JPEG and PNG images are supported.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 25 megapixel limit.")
                image.load()
        if image.width < 16 or image.height < 16:
            raise ValueError("Image is too small; use at least 16 by 16 pixels.")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds the 25 megapixel limit.")
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Unreadable or corrupted image. Upload a valid JPEG or PNG.") from exc


def preprocess_image(source, size=224):
    image = load_image(source)
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def validate_preprocessing(settings):
    if settings.get("pixel_range") != [0, 255] or settings.get("color_mode") != "RGB" or settings.get("resize") != "PIL_BILINEAR":
        raise ValueError("Unsupported preprocessing contract; install matching LeafGuard artifacts.")
    if not isinstance(settings.get("image_size"), int) or settings["image_size"] < 32:
        raise ValueError("Invalid saved image size.")
