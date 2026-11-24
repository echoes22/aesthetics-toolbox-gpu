"""Image size and colour QIPs (port of ``AT/color_and_simple_qips.py``)."""

import numpy as np
from scipy import stats
from skimage.measure import shannon_entropy


def image_size(width, height):
    """Image size = width + height in pixels (Datta et al., 2006)."""
    return width + height


def aspect_ratio(width, height):
    """Aspect ratio = width / height."""
    return width / height


def mean_channels(img):
    """Mean of each colour channel."""
    return np.mean(img, axis=(0, 1))


def std_channels(img):
    """Standard deviation of each colour channel. std(L*) is the RMS contrast."""
    return np.std(img, axis=(0, 1))


def circ_stats(img_hsv):
    """Circular mean and circular standard deviation of the HSV hue channel."""
    hue = img_hsv[:, :, 0].astype(float)
    return stats.circmean(hue, high=1, low=0), stats.circstd(hue, high=1, low=0)


def shannon_entropy_channel(channel):
    """Shannon entropy of a single channel after rescaling it to 256 levels.

    Used for colour entropy (HSV hue) and luminance entropy (L*a*b* lightness).
    """
    if np.max(channel) > 0:
        channel = channel / np.max(channel)
    channel = np.round(channel * 255)
    return shannon_entropy(channel)
