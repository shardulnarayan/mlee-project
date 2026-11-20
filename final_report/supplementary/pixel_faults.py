"""Pixel-space faults, separate from model loading and tensor-space noise."""
import io
from PIL import Image, ImageEnhance, ImageFilter


def corruption_cases():
    # Fixed fault severities used by the supplementary evaluation protocol.
    return ([('clean', 0)] + [('gaussian_blur_radius', x) for x in (1, 3, 5)] +
            [('jpeg_quality', x) for x in (95, 50, 20, 5)] +
            [('brightness_factor', x) for x in (0.25, 0.5, 1.5, 2.0)] +
            [('rotation_degrees', x) for x in (-20, 20)] +
            [('normalized_gaussian_sigma', x) for x in (0.1, 0.4, 0.8)])


def corrupt_pixels(image, kind, severity):
    """Corrupt the clean 224x224 representation before ImageNet normalization."""
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    if kind == 'gaussian_blur_radius':
        return image.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if kind == 'jpeg_quality':
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=int(severity), subsampling=2)
        buf.seek(0)
        with Image.open(buf) as decoded:
            return decoded.convert('RGB')
    if kind == 'brightness_factor':
        return ImageEnhance.Brightness(image).enhance(float(severity))
    if kind == 'rotation_degrees':
        return image.rotate(float(severity), resample=Image.Resampling.BILINEAR,
                            expand=False, fillcolor=(0, 0, 0))
    if kind in ('clean', 'normalized_gaussian_sigma'):
        return image
    raise ValueError(f'Unknown corruption {kind}')
