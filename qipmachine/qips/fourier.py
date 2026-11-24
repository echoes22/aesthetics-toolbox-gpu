"""Fourier spectral slopes and Fourier sigma (port of ``AT/fourier_qips.py``).

Three published methods are implemented, each with its own preprocessing:

* Redies et al. (2007): slope and sigma of the rotationally averaged power
  spectrum of the image padded to 1024 x 1024 pixels;
* Spehar & Isherwood: slope of the amplitude spectrum after removing outliers
  with Cook's distance;
* Mather (2014): slope of the L* power spectrum of the central power-of-2 crop.

The 2D FFT and the radial averaging run in float64 on the selected torch
device; the binning and line fits are the original NumPy code.
"""

from functools import lru_cache

import numpy as np
import PIL.Image
import torch
from skimage import color


@lru_cache(maxsize=8)
def _radius_bins(n, start, device):
    """Rounded distance to the spectrum centre for an n x n fftshift-ed spectrum.

    ``start`` is the coordinate of the first row/column (``-n // 2`` in the
    Redies code, ``-n / 2`` in the Spehar and Mather code). Pixels farther than
    n / 2 are mapped to an overflow bin. Returns (bin index per pixel, pixel count per bin).
    """
    coords = np.arange(start, start + n)
    x, y = np.meshgrid(coords, coords)
    rho = np.round(np.sqrt(x ** 2 + y ** 2)).astype(np.int64)
    n_r = n // 2 + 1
    rho[rho > n // 2] = n_r
    counts = np.bincount(rho.ravel(), minlength=n_r + 1)[:n_r]
    return torch.from_numpy(rho.ravel()).to(device), counts


def _radial_mean(spectrum, start):
    """Mean of ``spectrum`` (a square torch tensor) over rings of integer radius 0 .. n // 2."""
    n = spectrum.shape[0]
    rho, counts = _radius_bins(n, start, spectrum.device)
    sums = torch.bincount(rho, weights=spectrum.reshape(-1), minlength=n // 2 + 2)[:n // 2 + 1]
    return sums.cpu().numpy() / counts


def _require_finite(*arrays):
    """Fail clearly instead of passing log(0) to the line fit (LAPACK would print errors)."""
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("zero power at some frequencies (uniform image area); slope undefined")


def _shifted_spectrum(img, device):
    """``fftshift(fft2(img))`` computed in complex128 on ``device``."""
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64)).to(device)
    return torch.fft.fftshift(torch.fft.fft2(x))


################################ Redies #################################


def calc_bin(val, binsize, x_min, x_max):
    """Mean of ``val`` in logarithmically spaced frequency bins."""
    x_max = x_max + 1
    x_min = x_min + 1

    bin_num = np.ceil((x_max - x_min) / binsize).astype(int)
    bin_log_size = (np.log10(x_max) - np.log10(x_min)) / bin_num
    bin_mean_bar = np.zeros((2, bin_num))

    for i in range(bin_num):
        a_x = np.log10(x_min) + i * bin_log_size
        b_x = np.log10(x_min) + (i + 1) * bin_log_size
        from_val = np.floor(10 ** a_x).astype(int)
        to_val = np.floor(10 ** b_x).astype(int)
        if to_val > len(val):
            to_val = len(val)
        bin_mean_bar[0, i] = np.mean(np.arange(from_val, to_val + 1))  # x
        bin_mean_bar[1, i] = np.mean(val[np.arange(from_val - 1, to_val)])  # y

    return bin_mean_bar


def var_eigen(log_m, log_n, log_x, log_y):
    """Variance of the residuals around the fitted line (Fourier sigma)."""
    return np.var(log_m * log_x + log_n - log_y)


def padding_and_resizing_to_square_1024_pixel(img):
    """Resize the longer side to 1024 pixels and pad to a square with the mean grey value."""
    mean = np.round(np.mean(img)).astype(np.uint8)

    img = PIL.Image.fromarray(img)
    a = 1024 / float(max(img.size))
    img = img.resize((int(img.size[0] * a), int(img.size[1] * a)), PIL.Image.Resampling.LANCZOS)

    img = np.asarray(img)
    h, w = img.shape
    w_c = int(w / 2)
    h_c = int(h / 2)
    if h > w:
        img_pad = np.full([h, h], mean)
        img_pad[:, h_c - w_c:h_c + w_c + (w % 2)] = img
        img = img_pad
    elif h < w:
        img_pad = np.full([w, w], mean)
        img_pad[w_c - h_c:w_c + h_c + (h % 2), :] = img
        img = img_pad
    return img


def fourier_redies(img_gray, device, bin_size=2, cycles_min=10, cycles_max=256):
    """Return (Fourier sigma, Fourier slope) following Redies et al. (2007)."""
    img = padding_and_resizing_to_square_1024_pixel(img_gray)

    power = _shifted_spectrum(img, device).abs() ** 2
    a = _radial_mean(power, -img.shape[0] // 2)  # (-n) // 2, as in the original

    if cycles_max > len(a):
        cycles_max = len(a)

    rang = np.arange(cycles_min, cycles_max + 1) - 1
    log_rang = np.log10(rang + 1)
    log_using_range = np.log10(a[rang])

    bin_mean_bar = calc_bin(a, bin_size, rang[0], rang[-1])
    design = np.vstack([np.ones(len(bin_mean_bar[1])), np.log10(bin_mean_bar[0])]).T
    log_bin_means = np.log10(bin_mean_bar[1])
    _require_finite(log_bin_means, log_using_range)
    intercept, slope = np.linalg.lstsq(design, log_bin_means, rcond=None)[0]

    sigma = var_eigen(slope, intercept, log_rang, log_using_range)
    return sigma, slope


################################ Spehar & Isherwood #################################
# Translated from Zoey Isherwood's Matlab code (https://github.com/zoeyisherwood/pp-spatiotemp),
# rotational average by Bruno Olshausen. Outliers (mostly the lowest frequencies) are removed
# from the linear fit with Cook's distance > 4 / n before fitting the log-log slope.


def cooks_distance(x, y):
    """Cook's distance of each point of the least-squares fit ``y ~ 1 + x``.

    Closed form of ``statsmodels.OLS(y, add_constant(x)).fit().get_influence().cooks_distance[0]``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    n, k = design.shape
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    mse = resid @ resid / (n - k)
    xc = x - x.mean()
    leverage = 1.0 / n + xc ** 2 / (xc @ xc)
    return resid ** 2 / (k * mse) * leverage / (1.0 - leverage) ** 2


def fourier_slope_spehar(img_gray, device):
    """Fourier slope following Spehar & Isherwood (Isherwood et al., 2021)."""
    img_gray = np.asarray(img_gray)

    # Make dimensions even
    if img_gray.shape[0] % 2 == 1:
        img_gray = img_gray[:-1, :]
    if img_gray.shape[1] % 2 == 1:
        img_gray = img_gray[:, :-1]

    # Centre crop to the largest power-of-2 square (only if not square already)
    if img_gray.shape[0] != img_gray.shape[1]:
        s = img_gray.shape
        s_trim = min(2 ** int(np.log2(s[0])), 2 ** int(np.log2(s[1])))
        cx, cy = s[0] // 2, s[1] // 2
        img_gray = img_gray[cx - s_trim // 2:cx + s_trim // 2, cy - s_trim // 2:cy + s_trim // 2]

    xsize, ysize = img_gray.shape
    amplitude = _shifted_spectrum(img_gray, device).abs()
    pf = _radial_mean(amplitude, -xsize / 2)

    x_vec = np.arange(1, xsize // 2 + 1)
    y_vec = pf[1:ysize // 2 + 1]

    outliers = cooks_distance(x_vec, y_vec) > 4 / len(x_vec)
    x_vec = x_vec[~outliers]
    y_vec = y_vec[~outliers]

    _require_finite(np.log(y_vec))
    return np.polyfit(np.log(x_vec), np.log(y_vec), 1)[0]


################################ Mather #################################


def _mather_crop_box(height, width):
    """(top, bottom, left, right) of the largest centred power-of-2 square."""
    npow = np.floor(np.log2(height if width > height else width))
    c_new = int(2 ** (npow - 1))
    c_width = int(np.floor(width / 2))
    c_height = int(np.floor(height / 2))
    return c_height - c_new, c_height + c_new, c_width - c_new, c_width + c_new


def fourier_slope_mather(img_rgb, device, img_lab=None):
    """Fourier slope following Mather (2014), computed on the L* channel.

    ``img_lab`` is an optional precomputed ``rgb2lab(img_rgb)``; L* is computed
    per pixel, so cropping it is identical to converting the crop.
    """
    top, bottom, left, right = _mather_crop_box(*img_rgb.shape[:2])
    nr = bottom - top
    if img_lab is None:
        lightness = color.rgb2lab(img_rgb[top:bottom, left:right])[:, :, 0]
    else:
        lightness = img_lab[top:bottom, left:right, 0]
    cg = ((lightness / 100.0) * 255.0).astype(np.uint8)

    minf = int(np.round((nr / 2) * (1 / 25)))
    maxf = int(np.round((nr / 2) * 0.5))

    power = _shifted_spectrum(cg.astype(float) / 255, device).abs() ** 2
    pf = _radial_mean(power, -nr / 2)
    f1 = np.arange(nr // 2 + 1)

    log_f, log_pf = np.log(f1[minf - 1:maxf]), np.log(pf[minf - 1:maxf])
    _require_finite(log_f, log_pf)
    return np.polyfit(log_f, log_pf, 1)[0] / 2
