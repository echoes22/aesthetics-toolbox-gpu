"""Box-counting fractal dimensions (port of ``AT/fractal_dimension_qips.py``)."""

import numpy as np
import PIL.Image


def fractal_dimension_2d(img_gray):
    """2D fractal dimension of the binarised image (Branka Spehar's box count).

    The box-merging loop of the original is vectorised (identical result).
    """
    nr, nc = img_gray.shape
    # Scale to a square image
    if nr > nc:
        img_gray = np.asarray(PIL.Image.fromarray(img_gray).resize((nc, nc)))
    elif nr < nc:
        img_gray = np.asarray(PIL.Image.fromarray(img_gray).resize((nr, nr)))

    b = (img_gray < np.mean(img_gray)).astype(np.uint8)

    x, y = [], []
    while b.shape[0] > 6:
        x.append(b.shape[0])
        y.append(int(np.sum(b)))
        # Merge 2x2 boxes: a box counts if it is neither empty nor full
        h2, w2 = b.shape[0] // 2, b.shape[1] // 2
        block_sum = b[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        b = ((block_sum > 0) & (block_sum < 4)).astype(np.uint8)

    return np.polyfit(np.log2(x[1:]), np.log2(y[1:]), 1)[0]


def fractal_dimension_3d(img_gray):
    """3D fractal dimension using grey levels as the third dimension (George Mather's box count)."""
    # Centre crop to the largest square with a power-of-2 side
    nr, nc = img_gray.shape
    if nr < nc:
        nk = 2 ** np.ceil(np.log2(nr))
        nnr = 2 ** (np.ceil(np.log2(nr)) - 1) if nr < nk else 2 ** np.ceil(np.log2(nr))
        dc = nc - nnr
        dr = nr - nnr
        nnc = nnr
    else:
        nk = 2 ** np.ceil(np.log2(nc))
        nnc = 2 ** (np.ceil(np.log2(nc)) - 1) if nc < nk else 2 ** np.ceil(np.log2(nc))
        dr = nr - nnc
        dc = nc - nnc
        nnr = nnc
    nr, nc = int(nnr), int(nnc)
    img = img_gray[int(round(dr / 2)):int(round(dr / 2 + nr)), int(round(dc / 2)):int(round(dc / 2 + nc))]

    # Range of box sizes
    nr, nc = img.shape
    bmin = 2 ** int(np.ceil(np.log2(nr ** (1 / 3))))
    bmax = bmin
    while np.ceil(nr / bmax + 1) <= np.ceil(nr / (bmax - 1)):
        bmax = bmax + 1

    boxes = np.arange(bmin, bmax + 1, 2)
    box_heights = boxes * (1 / nr)  # box size in grey levels

    box_counts = np.zeros(len(boxes))
    box_sizes = np.zeros(len(boxes))
    for b in range(len(boxes)):
        bs = boxes[b]
        bh = box_heights[b]
        # Grid of bs x bs boxes; columns overlap by one pixel
        box_count = 0
        for by in range(1, nc - bs, bs):
            for bx in range(1, nr - bs + 1, bs - 1):
                submat = img[by - 1:by + bs - 1, bx - 1:bx + bs - 1]
                hi = np.max(submat)
                lo = np.min(submat)
                box_count = box_count + (1 if hi == lo else np.ceil((hi - lo) / bh))
        box_counts[b] = box_count
        box_sizes[b] = 1.0 / bs

    return np.polyfit(np.log(box_sizes), np.log(box_counts), 1)[0]
