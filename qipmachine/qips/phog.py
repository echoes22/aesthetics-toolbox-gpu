"""PHOG-based QIPs: self-similarity, complexity and anisotropy (port of ``AT/PHOG_qips.py``).

Based on the Matlab code of Redies & Groß (2013) / Braun et al. (2013). The
histogram computation in :func:`compute_descriptor` is vectorised with
``np.bincount``; it produces the same descriptor as the original cell loops.
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.transform import resize


def resize_img(img, re):
    """Resize to ``re`` pixels (keeping the aspect ratio); ``re <= 1`` disables resizing."""
    if re > 1:
        s = img.shape
        d = np.sqrt(re / (s[0] * s[1]))
        s_new = np.round([s[0] * d, s[1] * d]).astype(np.int32)
        img = resize(img, s_new, order=3, anti_aliasing=True) * 255
        return img.astype(np.uint8)
    return img


def absmax_nd(a, axis=2):
    """Value with the largest magnitude along ``axis`` (sign preserved)."""
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def max_gradient(img):
    """Per-pixel x/y gradients of the colour channel with the strongest response."""
    grad_y, grad_x = np.gradient(img, axis=[0, 1], edge_order=1)
    return absmax_nd(grad_x), absmax_nd(grad_y)


def compute_descriptor(gradient_value, gradient_angle, bins, angle, levels, section, is_global=False):
    """Pyramid of gradient-orientation histograms.

    Level ``l`` splits the image into ``section**l x section**l`` cells; cells are
    ordered column-major (x outer, y inner) as in the Matlab original.
    """
    interval = angle / bins
    half_interval = interval / 2
    height, width = gradient_angle.shape

    values = gradient_value.ravel()
    # Bin b covers [b*interval - half, (b+1)*interval - half); bin 0 wraps around
    angle_bin = np.floor((gradient_angle.ravel() + half_interval) / interval).astype(np.int64) % bins

    descriptor = []
    for level in range(levels + 1):
        ncells = section ** level
        if ncells == 1:
            hist = np.bincount(angle_bin, weights=values, minlength=bins).tolist()
            descriptor.extend(normalize_descriptor(hist, bins) if is_global else hist)
            continue

        cell_w = width / ncells
        cell_h = height / ncells
        if cell_w < 1 or cell_h < 1:
            raise ValueError("Cell size is less than 1. Adjust the number of levels.")

        col_cell = np.empty(width, dtype=np.int64)
        row_cell = np.empty(height, dtype=np.int64)
        for c in range(ncells):
            col_cell[int(np.round(c * cell_w)):int(np.round((c + 1) * cell_w))] = c
            row_cell[int(np.round(c * cell_h)):int(np.round((c + 1) * cell_h))] = c

        cell_index = (np.repeat(row_cell, width) * ncells + np.tile(col_cell, height)) * bins + angle_bin
        hists = np.bincount(cell_index, weights=values, minlength=ncells * ncells * bins)
        hists = hists.reshape(ncells, ncells, bins).transpose(1, 0, 2)  # (x cell, y cell, bin)

        if is_global:
            for hist in hists.reshape(-1, bins):
                descriptor.extend(normalize_descriptor(hist.tolist(), bins))
        else:
            descriptor.extend(hists.ravel().tolist())

    if is_global:
        return normalize_descriptor_global(descriptor)
    return descriptor


def compute_phog_lab(img, angle, bins, levels, section):
    gradient_x, gradient_y = max_gradient(img)
    gradient_value = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Avoid division by zero
    gradient_x[np.where(gradient_x == 0)] = 1e-5

    if angle == 180:
        gradient_angle = ((np.arctan(gradient_y / gradient_x) + (np.pi / 2)) * 180) / np.pi
    elif angle == 360:
        gradient_angle = ((np.arctan2(gradient_y, gradient_x) + np.pi) * 180) / np.pi
    else:
        raise ValueError("Invalid angle value. Use 180 or 360.")

    descriptor = compute_descriptor(gradient_value, gradient_angle, bins, angle, levels, section)
    return descriptor, gradient_value, gradient_angle


def convert_to_matlab_lab(img_rgb, img_lab=None):
    """L*a*b* scaled to Matlab's uint ranges, as used by the original PHOG code.

    ``img_lab`` (``skimage.color.rgb2lab(img_rgb)``) can be passed to skip the
    colour conversion.
    """
    img = rgb2lab(img_rgb) if img_lab is None else img_lab.copy()
    img[:, :, 0] = np.round(np.array((img[:, :, 0] / 100) * 255)).astype(np.int32)
    # a* and b* are truncated (astype before round), exactly as in the original
    img[:, :, 1] = np.round(np.array(img[:, :, 1] + 128).astype(np.int32))
    img[:, :, 2] = np.round(np.array(img[:, :, 2] + 128).astype(np.int32))
    return img.astype(np.uint16)


def normalize_descriptor(descriptor, bins):
    b = np.reshape(descriptor, (bins, len(descriptor) // bins), order='F')
    c = np.sum(b, axis=0)
    temp = np.zeros(b.shape)
    for i in range(b.shape[1]):
        temp[:, i] = b[:, i] / c[i] if c[i] != 0 else b[:, i]
    return list(np.reshape(temp, len(descriptor), order='F'))


def normalize_descriptor_global(descriptor):
    if np.sum(descriptor) != 0:
        return descriptor / np.sum(descriptor)
    return list(descriptor)


def compute_weighted_distances(descriptor, bins, levels, section, descriptor_nn):
    distances = []
    comparison_global = descriptor[:bins]

    temp = np.zeros((levels, 2), dtype=int)
    temp[0, 0] = bins + 1
    temp[0, 1] = section ** 2 * bins + temp[0, 0] - 1
    for i in range(1, levels):
        temp[i, 0] = temp[i - 1, 1] + 1
        temp[i, 1] = section ** ((i + 1) * 2) * bins + temp[i, 0] - 1

    distances.append(np.sum(comparison_global))

    for i in range(levels):
        for j in range(temp[i, 0], temp[i, 1] + 1, bins):
            j = j - 1
            part = descriptor[j:j + bins]

            if (np.max(comparison_global) > 1e-8) and (np.max(part) > 1e-8):
                dist = np.sum(np.minimum(comparison_global, part))

                m1 = np.mean(descriptor_nn[:bins])
                m2 = np.mean(descriptor_nn[j:j + bins]) * section ** ((i + 1) * 2)

                if m1 < 1e-8 or m2 < 1e-8:
                    strength_similarity = 0
                elif m1 > m2:
                    strength_similarity = m2 / m1
                else:
                    strength_similarity = m1 / m2

                distances.append(dist * strength_similarity)
            else:
                distances.append(0)

    return distances


def compute_sd(descriptor_global, bins, levels, section):
    temp = np.zeros((levels, 2), dtype=int)
    temp[0, 0] = bins + 1
    temp[0, 1] = section ** 2 * bins + temp[0, 0] - 1
    for i in range(1, levels):
        temp[i, 0] = temp[i - 1, 1] + 1
        temp[i, 1] = section ** ((i + 1) * 2) * bins + temp[i, 0] - 1
    return np.std(descriptor_global[temp[levels - 1, 0] - 1:temp[levels - 1, 1]])


def distance_at_level(distances, levels, section):
    temp = np.zeros([levels + 1, 2], dtype=int)
    temp[0, 0] = 1
    temp[0, 1] = 1
    for i in range(levels):
        temp[i + 1, 0] = section ** (2 * (i + 1))
        temp[i + 1, 1] = temp[i + 1, 0] + temp[i, 1]
    return np.median(distances[temp[levels - 1, 1]:temp[levels, 1]])


def phog_qips(img_rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=(1, 1, 1), img_lab=None):
    """Return (self-similarity, complexity, anisotropy) from the PHOG descriptor.

    ``img_lab`` is an optional precomputed ``rgb2lab(img_rgb)`` (ignored when resizing).
    """
    img = resize_img(img_rgb, re)
    img = convert_to_matlab_lab(img, img_lab if re <= 1 else None)

    descriptor, gradient_value, gradient_angle = compute_phog_lab(img, angle, bins, levels, section)
    descriptor_nn = descriptor
    descriptor = normalize_descriptor(descriptor, bins)
    descriptor_global = compute_descriptor(gradient_value, gradient_angle, bins, angle, levels, section,
                                           is_global=True)
    distances = compute_weighted_distances(descriptor, bins, levels, section, descriptor_nn)

    anisotropy = compute_sd(descriptor_global, bins, levels, section)
    complexity = np.mean(gradient_value)

    self_sim = 0
    for i in range(levels):
        self_sim = self_sim + distance_at_level(distances, i + 1, section) * sesfweight[i]
    self_sim = self_sim / sum(sesfweight)

    return self_sim, complexity, anisotropy
