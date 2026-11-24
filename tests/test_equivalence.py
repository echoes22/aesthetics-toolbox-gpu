"""The vectorised / GPU building blocks against the loop implementations of the original toolbox.

The reference functions below are condensed copies of the original
``AT/*_qips.py`` code. For a full end-to-end comparison with the original
toolbox, see ``scripts/compare_with_original.py``.
"""

import numpy as np
import pytest
import torch
from scipy import ndimage
from skimage.transform import resize

from qipmachine.qips import balance, cnn, edge_entropy, fourier, fractal, phog

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize('device', DEVICES)
def test_filterbank_matches_scipy_convolve(rng, device):
    img = rng.integers(0, 256, (57, 83)).astype(float)  # edge_resize returns integer-valued images
    resp_bin, resp_val = edge_entropy.run_filterbank(img, torch.device(device))
    ref = np.stack([ndimage.convolve(img, f) for f in edge_entropy.create_filterbank()])
    np.testing.assert_allclose(resp_val, ref.max(axis=0), rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(resp_bin, ref.argmax(axis=0))


def test_filterbank_ties_are_device_independent():
    # Uniform areas and straight edges produce many exactly equal responses
    img = np.zeros((120, 160))
    img[30:90, 40:120] = 200
    img[60:, 100:] = 90
    results = [edge_entropy.run_filterbank(img, torch.device(d)) for d in DEVICES]
    for resp_bin, resp_val in results:
        assert np.all(np.round(resp_val * 1e4) / 1e4 == resp_val)
        np.testing.assert_array_equal(resp_bin, results[0][0])
        np.testing.assert_array_equal(resp_val, results[0][1])


def reference_counting(resp_val, resp_bin, n_strongest, circ_bins=48, gabor_bins=24, max_diagonal=500):
    cutoff = np.sort(resp_val.flatten())[-n_strongest]
    resp_val[resp_val < cutoff] = 0
    ey, ex = resp_val.nonzero()
    h, w = resp_val.shape
    xx, yy = np.meshgrid(np.linspace(-w, w, 2 * w + 1), np.linspace(-h, h, 2 * h + 1))
    dist = np.sqrt(xx ** 2 + yy ** 2)
    orientations = resp_bin[ey, ex]
    counts = np.zeros([max_diagonal, circ_bins, gabor_bins])
    for cp in range(ex.size):
        orientations_rel = np.mod(orientations - orientations[cp] + gabor_bins, gabor_bins)
        distance_rel = np.round(dist[(ey - ey[cp]) + h, (ex - ex[cp]) + w]).astype(np.int64)
        distance_rel[distance_rel >= max_diagonal] = max_diagonal - 1
        direction = np.round(np.arctan2(ey - ey[cp], ex - ex[cp]) / (2.0 * np.pi) * circ_bins
                             + (orientations[cp] / float(gabor_bins) * circ_bins)).astype(np.int64)
        # The original casts to uint32 before the modulo; on x86-64 the unsigned
        # wrap-around makes this equal to the mathematical modulo used here.
        direction = np.mod(direction + circ_bins, circ_bins)
        np.add.at(counts, (distance_rel, direction, orientations_rel),
                  resp_val[ey, ex] * resp_val[ey[cp], ex[cp]])
    return counts


@pytest.mark.parametrize('device', DEVICES)
def test_pair_histogram_matches_loop(rng, device):
    resp_val = rng.uniform(0, 100, (40, 50))
    resp_bin = rng.integers(0, 24, (40, 50))
    expected = reference_counting(resp_val.copy(), resp_bin, n_strongest=400)
    counts, _ = edge_entropy.do_counting(resp_val.copy(), resp_bin, torch.device(device), n_strongest=400)
    np.testing.assert_allclose(counts, expected, rtol=1e-12, atol=1e-9)


def test_statistics_matches_loop(rng):
    counts = rng.uniform(0, 3, (60, 48, 24)) * (rng.uniform(size=(60, 48, 24)) > 0.5)
    counts[:5] = 0  # empty bins -> NaN
    counts_sum = np.sum(counts, axis=2) + 0.00001
    normalized = counts / counts_sum[:, :, np.newaxis]
    expected = np.full(counts_sum.shape, np.nan)
    for d in range(counts.shape[0]):
        for a in range(counts.shape[1]):
            if counts_sum[d, a] > 1:
                expected[d, a] = edge_entropy.entropy(normalized[d, a])
    np.testing.assert_allclose(edge_entropy.do_statistics(counts), expected, rtol=1e-12)


def reference_max_pooling(resp, patches):
    filters, ih, iw = resp.shape
    out = np.zeros((patches, patches, filters))
    patch_h, patch_w = ih / float(patches), iw / float(patches)
    for h in range(patches):
        for w in range(patches):
            ph, pw = h * patch_h, w * patch_w
            patch = resp[:, int(ph):int(ph + patch_h), int(pw):int(pw + patch_w)]
            out[h, w] = patch.reshape(filters, -1).max(axis=1)
    return out


@pytest.mark.parametrize('patches', [1, 8, 12, 17, 22])
def test_max_pooling_matches_loop(rng, patches):
    resp = rng.uniform(0, 1, (6, 126, 126))
    pooled = cnn.max_pooling(torch.from_numpy(resp)[None], patches)[0]
    np.testing.assert_array_equal(pooled, reference_max_pooling(resp, patches))


def test_flipping_commutes_with_cnn_resize(rng):
    img = rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)
    for flip in (np.fliplr, np.flipud):
        np.testing.assert_allclose(cnn.preprocess(flip(img)), flip(cnn.preprocess(img)), atol=1e-4)


def test_cnn_preprocessing_is_the_original():
    img = np.random.default_rng(1).integers(0, 256, (300, 410, 3), dtype=np.uint8)
    expected = resize(img[:, :, (2, 1, 0)].astype(np.float32), [512, 512], order=1) - cnn.IMAGENET_MEAN_BGR
    np.testing.assert_array_equal(cnn.preprocess(img), expected)


def test_cooks_distance_matches_statsmodels(rng):
    sm = pytest.importorskip('statsmodels.api')
    x = np.arange(1, 129, dtype=float)
    y = 1e4 * x ** -1.2 * rng.uniform(0.8, 1.2, x.size)
    expected = sm.OLS(y, sm.add_constant(x.reshape(-1, 1))).fit().get_influence().cooks_distance[0]
    np.testing.assert_allclose(fourier.cooks_distance(x, y), expected, rtol=1e-8)


@pytest.mark.parametrize('n,start', [(16, -8), (15, -15 // 2), (16, -8.0)])
@pytest.mark.parametrize('device', DEVICES)
def test_radial_mean_matches_loop(rng, n, start, device):
    spectrum = rng.uniform(0, 10, (n, n))
    coords = np.arange(start, start + n)
    x, y = np.meshgrid(coords, coords)
    rho = np.round(np.sqrt(x ** 2 + y ** 2)).astype(int)
    expected = np.array([spectrum[rho == r].mean() for r in range(n // 2 + 1)])
    result = fourier._radial_mean(torch.from_numpy(spectrum).to(device), start)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def reference_descriptor(value, angle_map, bins, angle, levels, section):
    """Unnormalised PHOG descriptor, original loop implementation."""
    interval = angle / bins
    half = interval / 2

    def hist(v, a):
        out = [np.sum(v[(a >= angle - half) | (a < half)])]
        for b in range(1, bins):
            out.append(np.sum(v[(a >= b * interval - half) & (a < (b + 1) * interval - half)]))
        return out

    descriptor = hist(value, angle_map)
    for level in range(1, levels + 1):
        cell_w = angle_map.shape[1] / (section ** level)
        cell_h = angle_map.shape[0] / (section ** level)
        for j in range(1, section ** level + 1):
            left, right = 1 + np.round((j - 1) * cell_w).astype(int), np.round(j * cell_w).astype(int)
            for i in range(1, section ** level + 1):
                top, bottom = 1 + np.round((i - 1) * cell_h).astype(int), np.round(i * cell_h).astype(int)
                descriptor.extend(hist(value[top - 1:bottom, left - 1:right], angle_map[top - 1:bottom, left - 1:right]))
    return descriptor


def test_phog_descriptor_matches_loop(rng):
    value = rng.uniform(0, 50, (101, 77))
    angle_map = rng.uniform(0, 360, (101, 77))
    angle_map[0, :16] = np.arange(16) * 22.5 + 11.25  # values on bin edges
    result = phog.compute_descriptor(value, angle_map, bins=16, angle=360, levels=3, section=2)
    np.testing.assert_allclose(result, reference_descriptor(value, angle_map, 16, 360, 3, 2), rtol=1e-12)


def reference_mirror_symmetry(img_gray):
    from skimage.filters import threshold_otsu
    from skimage.transform import rotate
    bw = img_gray <= threshold_otsu(img_gray)
    height, width = bw.shape
    h2 = height // 2
    sym = sum(bw[j, i] * bw[height - 1 - j, i] * (1 + j / (h2 - 1)) for i in range(width) for j in range(h2))
    sh = sym * (2 / (3 * width * h2))
    w2 = width // 2
    sym = sum(bw[i, j] * bw[i, width - 1 - j] * (1 + j / (w2 - 1)) for i in range(height) for j in range(w2))
    sv = sym * (2 / (3 * height * w2))
    if width != height:
        return ((sh + sv) / 2) * 100
    diagonals = []
    for b in (bw, rotate(bw, 90)):
        sym = sum(b[i, j] * b[j, i] * (1 + (j + 1) / i) for i in range(1, height) for j in range(i))
        diagonals.append(sym * (2 / (3 * height * (width - 1) / 2)))
    return ((sh + sv + sum(diagonals)) / 4) * 100


@pytest.mark.parametrize('shape', [(40, 40), (41, 37), (30, 52)])
def test_mirror_symmetry_matches_loop(rng, shape):
    img = (ndimage.gaussian_filter(rng.uniform(0, 255, shape), 3) * 3 % 256).astype(np.uint8)
    assert balance.mirror_symmetry(img) == pytest.approx(reference_mirror_symmetry(img), rel=1e-12)


def test_fractal_2d_box_merging_matches_loop(rng):
    img = (ndimage.gaussian_filter(rng.uniform(0, 255, (128, 128)), 2) * 4 % 256).astype(np.uint8)
    b = (img < np.mean(img)).astype(np.uint8)
    x, y = [], []
    while b.shape[0] > 6:
        x.append(b.shape[0])
        y.append(np.sum(b))
        c = np.zeros((b.shape[0] // 2, b.shape[1] // 2))
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i, j] = np.sum(b[i * 2:i * 2 + 2, j * 2:j * 2 + 2])
        b = (c > 0) & (c < 4)
    expected = np.polyfit(np.log2(x[1:]), np.log2(y[1:]), 1)[0]
    assert fractal.fractal_dimension_2d(img) == expected
