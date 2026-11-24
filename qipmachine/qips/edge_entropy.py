"""Edge density and 1st/2nd-order edge-orientation entropy (port of ``AT/edge_entropy_qips.py``).

Redies et al. (2017). The two expensive steps run in float64 on the selected
torch device:

* the 24-filter Gabor filterbank is a single ``conv2d`` call with symmetric
  padding, which reproduces ``scipy.ndimage.convolve`` (mode ``'reflect'``);
* the pairwise histogram over the ~10,000 strongest edge pixels (~10^8 pairs)
  is accumulated chunk by chunk with ``index_add_`` instead of a Python loop.
  Distance and direction bins are computed from the same NumPy lookup tables
  and float64 expressions as the original, so every pair lands in the same bin.
"""

import warnings
from functools import lru_cache

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

FILTER_SIZE = 31
NUM_FILTERS = 24
CIRC_BINS = 48
GABOR_BINS = 24
MAX_DIAGONAL = 500
N_STRONGEST = 10000
PAIRS_PER_CHUNK = 2 ** 20


def create_gabor(size, theta=0, octave=3):
    amplitude = 1.0
    phase = np.pi / 2.0
    frequency = 0.5 ** octave
    hrsf = 4  # half response spatial frequency bandwidth
    sigma = 1 / (np.pi * frequency) * np.sqrt(np.log(2) / 2) * (2.0 ** hrsf + 1) / (2.0 ** hrsf - 1)
    valsy = np.linspace(-size // 2 + 1, size // 2, size)
    valsx = np.linspace(-size // 2 + 1, size // 2, size)
    xgr, ygr = np.meshgrid(valsx, valsy)

    omega = 2 * np.pi * frequency
    gaussian = np.exp(-(xgr * xgr + ygr * ygr) / (2 * sigma * sigma))
    slant = xgr * (omega * np.sin(theta)) + ygr * (omega * np.cos(theta))

    gabor = np.round(gaussian, decimals=4) * amplitude * np.cos(slant + phase)
    return np.round(gabor, decimals=4)


def create_filterbank(flt_size=FILTER_SIZE, num_filters=NUM_FILTERS):
    flt_raw = np.zeros([num_filters, flt_size, flt_size])
    thetas = np.linspace(0, 2 * np.pi, num_filters + 1)[:-1]
    for i in range(num_filters):
        flt_raw[i, :, :] = create_gabor(flt_size, theta=thetas[i], octave=3)
    return flt_raw


@lru_cache(maxsize=None)
def _filterbank_kernel(device):
    # scipy.ndimage.convolve flips the kernel, conv2d (cross-correlation) does not
    flt = np.ascontiguousarray(create_filterbank()[:, ::-1, ::-1])
    return torch.from_numpy(flt).unsqueeze(1).to(device)  # (24, 1, 31, 31) float64


def run_filterbank(img, device):
    """Return (index, value) of the strongest Gabor response at every pixel."""
    pad = FILTER_SIZE // 2
    h, w = img.shape
    # scipy's 'reflect' boundary mode is numpy's 'symmetric' padding (edge pixel repeated)
    rows = torch.from_numpy(np.pad(np.arange(h), pad, mode='symmetric')).to(device)
    cols = torch.from_numpy(np.pad(np.arange(w), pad, mode='symmetric')).to(device)
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64)).to(device)
    x = x[rows][:, cols]

    responses = F.conv2d(x[None, None], _filterbank_kernel(device))[0]  # (24, h, w)
    # The kernels have 4 decimals and the resized image is integer-valued, so the exact
    # responses are multiples of 1e-4. Rounding removes the floating-point noise, so that
    # equal responses (frequent in images with uniform areas or straight edges) are exactly
    # equal and ties are resolved the same way on every device: first filter in argmax, all
    # tied pixels kept at the cutoff of do_counting.
    responses = torch.round(responses * 1e4)  # exact integers
    resp_bin = torch.argmax(responses, dim=0)  # first maximum, like np.argmax
    resp_val = torch.amax(responses, dim=0)
    return resp_bin.cpu().numpy(), resp_val.cpu().numpy() / 1e4  # NumPy division: same values on every device


def edge_density(resp_val):
    return np.sum(resp_val) / float(resp_val.shape[0] * resp_val.shape[1])


@lru_cache(maxsize=4)
def _lookup_tables(h, w, device):
    """Distance bin and (unrounded) direction of every displacement (dy, dx), indexed by
    (dy + h) * (2w + 1) + (dx + w), using the expressions of the original code."""
    xx, yy = np.meshgrid(np.linspace(-w, w, 2 * w + 1), np.linspace(-h, h, 2 * h + 1))
    distance = np.round(np.sqrt(xx ** 2 + yy ** 2))
    distance = np.minimum(distance, MAX_DIAGONAL - 1).astype(np.int64)
    direction = np.arctan2(yy, xx) / (2.0 * np.pi) * CIRC_BINS
    return (torch.from_numpy(distance.ravel()).to(device),
            torch.from_numpy(direction.ravel()).to(device))


def do_counting(resp_val, resp_bin, device, n_strongest=N_STRONGEST):
    """Histogram over pairs of strong edge pixels: (distance, direction, relative orientation).

    Each pair (i, j) adds strength_i * strength_j to the bin given by their
    distance, the direction of i seen from j relative to j's orientation, and
    the orientation of i relative to j. ``resp_val`` is modified in place
    (responses below the ``n_strongest``-th largest are set to 0), as in the original.
    """
    cutoff = np.sort(resp_val.flatten())[-n_strongest]
    resp_val[resp_val < cutoff] = 0
    ey, ex = resp_val.nonzero()
    h, w = resp_val.shape

    orientations = resp_bin[ey, ex]
    distance_lut, direction_lut = _lookup_tables(h, w, device)

    def to_device(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    ey_t = to_device(ey.astype(np.int64))
    ex_t = to_device(ex.astype(np.int64))
    ori_t = to_device(orientations.astype(np.int64))
    val_t = to_device(resp_val[ey, ex].astype(np.float64))
    # Direction offset of each reference pixel, same float expression as the original
    offset_t = to_device(orientations / float(GABOR_BINS) * CIRC_BINS)

    counts = torch.zeros(MAX_DIAGONAL * CIRC_BINS * GABOR_BINS, dtype=torch.float64, device=device)
    n = ey.size
    step = max(1, PAIRS_PER_CHUNK // max(n, 1))
    for start in range(0, n, step):
        cp = slice(start, start + step)  # reference pixels of this chunk
        lut_idx = (ey_t[None, :] - ey_t[cp, None] + h) * (2 * w + 1) + (ex_t[None, :] - ex_t[cp, None] + w)
        distance = distance_lut[lut_idx]
        # round half to even + modulo, as np.round(...).astype(uint32) followed by np.mod in the original
        direction = torch.round(direction_lut[lut_idx] + offset_t[cp, None]).long().remainder(CIRC_BINS)
        orientation = (ori_t[None, :] - ori_t[cp, None]).remainder(GABOR_BINS)
        flat = (distance * CIRC_BINS + direction) * GABOR_BINS + orientation
        counts.index_add_(0, flat.reshape(-1), (val_t[None, :] * val_t[cp, None]).reshape(-1))

    return counts.reshape(MAX_DIAGONAL, CIRC_BINS, GABOR_BINS).cpu().numpy(), resp_val


def entropy(a):
    if np.sum(a) != 1.0 and np.sum(a) > 0:
        a = a / np.sum(a)
    v = a > 0.0
    return -np.sum(a[v] * np.log2(a[v]))


def do_statistics(counts):
    """Shannon entropy of the relative-orientation histogram of every (distance, direction) bin.

    Bins with a total weight <= 1 are NaN. Vectorised version of the original loop.
    """
    counts_sum = np.sum(counts, axis=2) + 0.00001
    normalized = counts / counts_sum[:, :, np.newaxis]
    # entropy() renormalises each histogram by its sum before taking -sum(p log2 p)
    total = np.sum(normalized, axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where((total != 1.0) & (total > 0), normalized / total, normalized)
        shannon = -np.sum(np.where(p > 0.0, p * np.log2(p), 0.0), axis=2)
    return np.where(counts_sum > 1, shannon, np.nan)


def edge_resize(img_gray, max_pixels=300 * 400):
    img = PIL.Image.fromarray(img_gray)
    s0, s1 = img.size
    a = np.sqrt(max_pixels / float(s0 * s1))
    img = img.resize((int(s0 * a), int(s1 * a)), PIL.Image.Resampling.LANCZOS)
    return np.asarray(img, dtype='float')


def eoe_qips(img_gray, device):
    """Return (1st-order EOE, 2nd-order EOE, edge density) of an 8-bit grayscale image."""
    img = edge_resize(img_gray)
    resp_bin, resp_val = run_filterbank(img, device)

    # Edge density must be computed before do_counting zeroes the weak responses
    edge_d = edge_density(resp_val)
    counts, resp_val = do_counting(resp_val, resp_bin, device)

    first_order_bin = np.zeros(GABOR_BINS)
    for b in range(GABOR_BINS):
        first_order_bin[b] = np.sum(resp_val[resp_bin == b])
    first_order = entropy(first_order_bin)

    shannon_nan = do_statistics(counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # mean of empty slice
        second_order = np.nanmean(np.nanmean(shannon_nan, axis=1)[20:240])

    return first_order, second_order, edge_d
