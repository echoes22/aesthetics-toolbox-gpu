"""QIPs based on first-layer AlexNet features (port of ``AT/CNN_qips.py``).

Symmetry (Brachmann & Redies, 2016), self-similarity, sparseness and
variability (Brachmann et al., 2017) are computed from the max-pooled responses
of the 96 conv1 filters of the Caffe AlexNet (``data/bvlc_alexnet_conv1.npy``).

The image is resized on the CPU with the original preprocessing; the
convolution of the original image and its three flipped versions runs as one
float64 batch on the selected torch device. Flipping the resized image is
equivalent to resizing the flipped image (the resize is mirror-symmetric).
"""

from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize

IMAGENET_MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])


def load_alexnet_conv1():
    """Return (kernel (11, 11, 3, 96), bias (96,)) of the Caffe AlexNet conv1 layer."""
    with files("qipmachine").joinpath("data/bvlc_alexnet_conv1.npy").open("rb") as f:
        kernel, bias = np.load(f, encoding="latin1", allow_pickle=True)
    return kernel, bias


@lru_cache(maxsize=None)
def _conv1_weights(device):
    kernel, bias = load_alexnet_conv1()
    kernel = torch.from_numpy(kernel.astype(np.float64)).permute(3, 2, 0, 1).contiguous()  # (96, 3, 11, 11)
    bias = torch.from_numpy(bias.astype(np.float64))
    return kernel.to(device), bias.to(device)


def preprocess(img_rgb):
    """Caffe channel order (BGR), bilinear resize to 512 x 512 and ImageNet mean subtraction."""
    img = img_rgb[:, :, (2, 1, 0)].astype(np.float32)
    img = resize(img, [512, 512], order=1)
    return img - IMAGENET_MEAN_BGR


def conv1_responses(imgs, device):
    """ReLU(conv1) responses, stride 4, no padding: (B, 512, 512, 3) -> (B, 96, 126, 126)."""
    kernel, bias = _conv1_weights(device)
    return F.relu(F.conv2d(imgs.permute(0, 3, 1, 2), kernel, bias, stride=4))


def max_pooling(resp, patches):
    """Max of each filter response over a ``patches x patches`` grid.

    ``resp`` is a (B, C, H, W) tensor; returns a (B, patches, patches, C) NumPy
    array. Cell boundaries are those of the original implementation.
    """
    _, _, ih, iw = resp.shape
    patch_h = ih / float(patches)
    patch_w = iw / float(patches)
    # The maximum over a rectangle is the maximum over its rows of the column maxima
    rows = torch.stack([resp[:, :, int(h * patch_h):int(h * patch_h + patch_h), :].amax(dim=2)
                        for h in range(patches)], dim=2)
    pooled = torch.stack([rows[:, :, :, int(w * patch_w):int(w * patch_w + patch_w)].amax(dim=3)
                          for w in range(patches)], dim=3)
    return pooled.permute(0, 2, 3, 1).cpu().numpy()


def normalize(max_pool_map):
    """Normalise each cell's filter responses to sum to one."""
    return max_pool_map / np.sum(max_pool_map, axis=2)[:, :, np.newaxis]


def get_differences(max_pool_map_orig, max_pool_map_flip):
    sum_abs = np.sum(np.abs(max_pool_map_orig - max_pool_map_flip))
    sum_max = np.sum(np.maximum(max_pool_map_orig, max_pool_map_flip))
    return 1.0 - sum_abs / sum_max


def cnn_variance(normalized_max_pool_map, kind):
    """'sparseness': variance of all values; 'variability': median over filters of the spatial variance."""
    if kind == 'sparseness':
        return np.var(normalized_max_pool_map)
    if kind == 'variability':
        return np.median(np.var(normalized_max_pool_map, axis=(0, 1)))
    raise ValueError("kind must be 'sparseness' or 'variability'")


def cnn_selfsimilarity(histogram_ground, histogram_level):
    """Median histogram intersection between the whole-image histogram and each cell's histogram."""
    ph, pw, _ = histogram_level.shape
    hiks = [np.sum(np.minimum(histogram_ground, histogram_level[ih, iw]))
            for ih in range(ph) for iw in range(pw)]
    return np.median(hiks)


def cnn_qips(img_rgb, device, symmetry=True, sparseness=True, variability=True, self_similarity=True):
    """Compute the requested CNN QIPs of an 8-bit RGB image.

    Returns a dict with keys among 'symmetry_lr', 'symmetry_ud', 'symmetry_lrud',
    'sparseness', 'variability' and 'self_similarity'.
    """
    x = torch.from_numpy(preprocess(img_rgb)).to(device).unsqueeze(0)  # (1, 512, 512, 3)
    if symmetry:
        x = torch.cat([x, x.flip(2), x.flip(1), x.flip(1).flip(2)])  # original, left-right, up-down, both
    resp = conv1_responses(x, device)

    out = {}
    if symmetry:
        pooled = max_pooling(resp, patches=17)
        out['symmetry_lr'] = get_differences(pooled[0], pooled[1])
        out['symmetry_ud'] = get_differences(pooled[0], pooled[2])
        out['symmetry_lrud'] = get_differences(pooled[0], pooled[3])

    resp = resp[:1]
    if sparseness:
        out['sparseness'] = cnn_variance(normalize(max_pooling(resp, patches=22)[0]), kind='sparseness')
    if variability:
        out['variability'] = cnn_variance(normalize(max_pooling(resp, patches=12)[0]), kind='variability')
    if self_similarity:
        ground = normalize(max_pooling(resp, patches=1)[0])
        level = normalize(max_pooling(resp, patches=8)[0])
        out['self_similarity'] = cnn_selfsimilarity(ground, level)
    return out
