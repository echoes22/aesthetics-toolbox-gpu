"""Balance, DCM, mirror symmetry and homogeneity (port of ``AT/balance_qips.py``).

Translated from the Matlab code of Ronald Hübner's group
(Hübner & Fillinger, 2016). Mirror symmetry is vectorised; everything else is
the original code.
"""

import numpy as np
from skimage.filters import threshold_otsu


def balance(img_gray):
    """Balance QIP (0-100, lower values = better balanced)."""
    height, width = img_gray.shape

    counts = np.histogram(img_gray, bins=256, range=(0, 256))[0]
    thres = 128
    sum1 = sum(counts[:thres])
    # The Matlab original counts the threshold bin twice; kept for compatibility.
    sum2 = sum(counts[thres - 1:])

    if sum1 <= sum2:
        im_comp = 255 - img_gray
    else:
        im_comp = img_gray

    nall = np.sum(im_comp)
    if nall == 0:
        nall = 1

    # Horizontal balance
    w = width // 2
    s1 = np.sum(im_comp[:, :w], dtype=int)
    s2 = np.sum(im_comp[:, -w:], dtype=int)
    bh = (abs(s1 - s2) / nall) * 100

    w2 = width // 4
    s1 = np.sum(im_comp[:, :w2], dtype=int)
    s2 = np.sum(im_comp[:, -w2:], dtype=int)
    bioh = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Vertical balance
    h = height // 2
    s1 = np.sum(im_comp[:h, :], dtype=int)
    s2 = np.sum(im_comp[-h:, :], dtype=int)
    bv = (abs(s1 - s2) / nall) * 100

    h2 = height // 4
    s1 = np.sum(im_comp[:h2, :], dtype=int)
    s2 = np.sum(im_comp[-h2:, :], dtype=int)
    biov = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Main diagonal and inner-outer
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bmd = (abs(s1 - s2) / nall) * 100

    prop = 1 / np.sqrt(2)
    b1 = height - int(height * prop)
    b2 = width - int(width * prop)
    s1 = np.sum(np.tril(im_comp, -b1), dtype=int)
    s2 = np.sum(np.triu(im_comp, b2), dtype=int)
    biomd = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Anti-diagonal and inner-outer
    im_comp = np.rot90(im_comp)
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bad = (abs(s1 - s2) / nall) * 100

    s1 = np.sum(np.tril(im_comp, -b2), dtype=int)
    s2 = np.sum(np.triu(im_comp, b1), dtype=int)
    bioad = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    return (bh + bv + bioh + biov + bmd + biomd + bad + bioad) / 8


def dcm(img_gray):
    """Deviation of the centre of mass.

    Returns (distance from the image centre in % of half the image size,
    horizontal offset, vertical offset), offsets normalised by image size.
    """
    height, width = img_gray.shape

    counts = np.histogram(img_gray, bins=256, range=(0, 256))[0]
    thres = 128
    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres:])

    if sum1 <= sum2:
        im_comp = 255 - img_gray
    else:
        im_comp = img_gray

    nall = np.sum(im_comp)

    # Horizontal balance point
    r = 0
    for i in range(width):
        r += np.sum(im_comp[:, i], dtype=float) * i
    rh = np.round(r / nall) + 1
    rh_norm = rh / width

    # Vertical balance point
    r = 0
    for i in range(height):
        r += np.sum(im_comp[i, :], dtype=float) * i
    rv = np.round(r / nall) + 1
    rv_norm = rv / height

    htmp = 0.5 - rh_norm
    vtmp = 0.5 - rv_norm
    dist = np.sqrt(htmp ** 2 + vtmp ** 2)
    return (dist / 0.5) * 100, htmp, vtmp


def mirror_symmetry(img_gray):
    """Mirror symmetry of the Otsu-binarised image (0-100).

    Vectorised version of the original pixel loops (identical result).
    Diagonal axes are only evaluated for square images.
    """
    level = threshold_otsu(img_gray)
    bw = (img_gray <= level).astype(np.float64)
    height, width = bw.shape

    # Horizontal axis of reflection: pixel j from the top vs pixel j from the bottom
    h2 = height // 2
    weights = (1.0 + np.arange(h2) / max(h2 - 1, 1))[:, None]
    sh = np.sum(bw[:h2] * bw[::-1][:h2] * weights) * (2 / (3 * width * h2))

    # Vertical axis of reflection
    w2 = width // 2
    weights = (1.0 + np.arange(w2) / max(w2 - 1, 1))[None, :]
    sv = np.sum(bw[:, :w2] * bw[:, ::-1][:, :w2] * weights) * (2 / (3 * height * w2))

    if width == height:
        # Major and minor diagonals, pixel pairs (i, j) with j < i weighted by 1 + (j+1)/i
        i_idx, j_idx = np.tril_indices(height, k=-1)
        weights = 1.0 + (j_idx + 1) / i_idx
        norm = 2 / (3 * height * (width - 1) / 2)
        smd = np.sum(bw[i_idx, j_idx] * bw[j_idx, i_idx] * weights) * norm
        bw_rot = np.rot90(bw)
        sad = np.sum(bw_rot[i_idx, j_idx] * bw_rot[j_idx, i_idx] * weights) * norm
        return ((sh + sv + smd + sad) / 4) * 100
    return ((sh + sv) / 2) * 100


def homogeneity(img_gray):
    """Homogeneity: normalised entropy of the distribution of dark pixels over a 10x10 grid (0-100)."""
    hbins = 10
    vbins = 10
    height, width = img_gray.shape

    counts = np.histogram(img_gray, bins=256, range=(0, 256))[0]
    thres = 128
    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres - 1:])
    if sum1 <= sum2:
        im = 255 - img_gray
    else:
        im = img_gray

    bw = im > threshold_otsu(im)

    hinc = width // hbins
    vinc = height // vbins
    x = np.zeros((vbins, hbins))
    # Sum pixels per cell; leftover rows/columns are added to the last cells
    for i in range(hbins + 1):
        for j in range(vbins + 1):
            if (i != hbins) and (j != vbins):
                x[j, i] = np.sum(bw[vinc * j:(j + 1) * vinc, hinc * i:hinc * (i + 1)])
            elif (i < hbins) and (j == vbins):
                x[j - 1, i] += np.sum(bw[vinc * j:, hinc * i:hinc * (i + 1)])
            elif (i == hbins) and (j < vbins):
                x[j, i - 1] += np.sum(bw[vinc * j:(j + 1) * vinc, hinc * i:])
            elif (i == hbins) and (j == vbins):
                x[j - 1, i - 1] += np.sum(bw[vinc * j:, hinc * i:])

    all_sum = np.sum(x)

    y = np.sum(x, axis=0) / all_sum
    nz = np.nonzero(y)
    en_hori = (-np.sum(y[nz] * np.log2(y[nz])) / np.log2(hbins)) * 100

    y = np.sum(x.T, axis=0) / all_sum
    nz = np.nonzero(y)
    en_vert = (-np.sum(y[nz] * np.log2(y[nz])) / np.log2(vbins)) * 100

    return (en_hori + en_vert) / 2
