"""Finding and loading images."""

import os
from functools import cached_property

import numpy as np
from PIL import Image
from skimage import color

Image.MAX_IMAGE_PIXELS = None  # allow very large artworks

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.webp', '.gif'}


def find_images(folder):
    """Paths of all images below ``folder``, relative to it, sorted, with '/' separators.

    Hidden files and folders (starting with '.') are skipped.
    """
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            if not name.startswith('.') and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                found.append(os.path.relpath(os.path.join(root, name), folder).replace(os.sep, '/'))
    return sorted(found)


class LoadedImage:
    """An image in the representations used by the QIPs.

    ``rgb`` and ``gray`` are 8-bit arrays obtained like in the Aesthetics
    Toolbox (Pillow ``convert('RGB')`` / ``convert('L')``); ``lab`` and ``hsv``
    are computed with scikit-image on first use. ``original_size`` is the
    (width, height) of the file before any resizing.
    """

    def __init__(self, rgb, gray, original_size):
        self.rgb = rgb
        self.gray = gray
        self.original_size = original_size

    @cached_property
    def lab(self):
        return color.rgb2lab(self.rgb)

    @cached_property
    def hsv(self):
        return color.rgb2hsv(self.rgb)


def load_image(source, short_side=None):
    """Load an image file (or a PIL image) as a :class:`LoadedImage`.

    If ``short_side`` is given and the shorter side of the image is longer than
    that, the image is first downscaled (Lanczos) so that its shorter side is
    ``short_side`` pixels. Images are never upscaled.
    """
    img = Image.open(source) if not isinstance(source, Image.Image) else source
    try:
        width, height = img.size
        if short_side and min(width, height) > short_side:
            scale = short_side / min(width, height)
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            rgb = img.convert('RGB').resize(new_size, Image.Resampling.LANCZOS)
            gray = rgb.convert('L')
        else:
            rgb = img.convert('RGB')
            gray = img.convert('L')
        return LoadedImage(np.asarray(rgb), np.asarray(gray), (width, height))
    finally:
        if img is not source:
            img.close()
