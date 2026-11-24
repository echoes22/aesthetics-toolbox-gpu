"""Compare qipmachine with the original Aesthetics Toolbox implementation.

Runs the original CPU code (``AT/*_qips.py`` of a clone of
https://github.com/RBartho/Aesthetics-Toolbox, called exactly like its
``QIP_machine_script.py``) and qipmachine on the same images at full
resolution, then reports the largest absolute and relative difference of every
QIP column and the time per image of both implementations.

    git clone https://github.com/RBartho/Aesthetics-Toolbox
    pip install statsmodels   # needed by the original code
    python scripts/compare_with_original.py --original Aesthetics-Toolbox --images my_images --n 10

The original implementation is slow (about a minute per image), so use --jobs
to compute the reference values in parallel.
"""

import argparse
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qipmachine.image_io import find_images  # noqa: E402  (does not import PyTorch)


def reference_qips(path, original_dir):
    """QIPs from the original toolbox code, keyed by qipmachine column names."""
    if original_dir not in sys.path:
        sys.path.insert(0, original_dir)
    from skimage import color
    from AT import (CNN_qips, PHOG_qips, balance_qips, color_and_simple_qips, edge_entropy_qips,
                    fourier_qips, fractal_dimension_qips)

    start = time.perf_counter()
    img_rgb = np.asarray(Image.open(path).convert('RGB'))
    img_lab = color.rgb2lab(img_rgb)
    img_hsv = color.rgb2hsv(img_rgb)
    img_gray = np.asarray(Image.open(path).convert('L'))
    with open(os.path.join(original_dir, 'AT', 'bvlc_alexnet_conv1.npy'), 'rb') as f:
        kernel, bias = np.load(f, encoding='latin1', allow_pickle=True)

    def cnn_rest():
        resp = CNN_qips.conv2d(img_rgb, kernel, bias)
        _, sparse = CNN_qips.max_pooling(resp, patches=22)
        _, var = CNN_qips.max_pooling(resp, patches=12)
        _, map8 = CNN_qips.max_pooling(resp, patches=8)
        _, map1 = CNN_qips.max_pooling(resp, patches=1)
        return (CNN_qips.CNN_Variance(sparse, kind='sparseness'), CNN_qips.CNN_Variance(var, kind='variability'),
                CNN_qips.CNN_selfsimilarity(map1, map8))

    groups = {
        ('image_size',): lambda: [color_and_simple_qips.image_size(img_rgb)],
        ('aspect_ratio',): lambda: [color_and_simple_qips.aspect_ratio(img_rgb)],
        ('rms_contrast',): lambda: [color_and_simple_qips.std_channels(img_lab)[0]],
        ('luminance_entropy',): lambda: [color_and_simple_qips.shannonentropy_channels(img_lab[:, :, 0])],
        ('color_entropy',): lambda: [color_and_simple_qips.shannonentropy_channels(img_hsv[:, :, 0])],
        ('mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b'): lambda: color_and_simple_qips.mean_channels(img_rgb),
        ('mean_lab_l', 'mean_lab_a', 'mean_lab_b'): lambda: color_and_simple_qips.mean_channels(img_lab),
        ('mean_hsv_h', 'mean_hsv_s', 'mean_hsv_v'): lambda: [color_and_simple_qips.circ_stats(img_hsv)[0],
                                                             *color_and_simple_qips.mean_channels(img_hsv)[1:]],
        ('std_rgb_r', 'std_rgb_g', 'std_rgb_b'): lambda: color_and_simple_qips.std_channels(img_rgb),
        ('std_lab_l', 'std_lab_a', 'std_lab_b'): lambda: color_and_simple_qips.std_channels(img_lab),
        ('std_hsv_h', 'std_hsv_s', 'std_hsv_v'): lambda: [color_and_simple_qips.circ_stats(img_hsv)[1],
                                                          *color_and_simple_qips.std_channels(img_hsv)[1:]],
        ('mirror_symmetry',): lambda: [balance_qips.Mirror_symmetry(img_gray)],
        ('dcm_distance', 'dcm_x', 'dcm_y'): lambda: balance_qips.DCM(img_gray),
        ('balance',): lambda: [balance_qips.Balance(img_gray)],
        ('homogeneity',): lambda: [balance_qips.Homogeneity(img_gray)],
        ('cnn_symmetry_lr', 'cnn_symmetry_ud', 'cnn_symmetry_lrud'):
            lambda: CNN_qips.CNN_symmetry(img_rgb, kernel, bias),
        ('sparseness', 'variability', 'self_similarity_cnn'): cnn_rest,
        ('fourier_sigma', 'fourier_slope_redies'):
            lambda: fourier_qips.fourier_redies(img_gray, bin_size=2, cycles_min=10, cycles_max=256),
        ('fourier_slope_spehar',): lambda: [fourier_qips.fourier_slope_branka_Spehar_Isherwood(img_gray)],
        ('fourier_slope_mather',): lambda: [fourier_qips.fourier_slope_mather(img_rgb)],
        ('fractal_dimension_2d',): lambda: [fractal_dimension_qips.fractal_dimension_2d(img_gray)],
        ('fractal_dimension_3d',): lambda: [fractal_dimension_qips.fractal_dimension_3d(img_gray)],
        ('self_similarity_phog', 'complexity', 'anisotropy'): lambda: PHOG_qips.PHOGfromImage(
            img_rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=[1, 1, 1]),
        ('eoe_1st_order', 'eoe_2nd_order', 'edge_density'):
            lambda: edge_entropy_qips.do_first_and_second_order_entropy_and_edge_density(img_gray),
    }

    values = {}
    for columns, compute in groups.items():
        try:
            result = list(compute())
        except Exception:
            result = [math.nan] * len(columns)
        values.update({c: float(v) for c, v in zip(columns, result)})
    return values, time.perf_counter() - start


def _reference_task(args):
    return reference_qips(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--original', required=True, help="path to a clone of the Aesthetics Toolbox repository")
    parser.add_argument('--images', required=True, help="folder with test images")
    parser.add_argument('--n', type=int, default=None, help="use n images spread evenly over the folder")
    parser.add_argument('--device', default='auto', help="device for qipmachine (default: auto)")
    parser.add_argument('--jobs', type=int, default=1, help="parallel processes for the original code")
    parser.add_argument('--save', metavar='PREFIX', help="also save PREFIX_original.csv and PREFIX_qipmachine.csv")
    args = parser.parse_args()

    from qipmachine import QIPExtractor  # PyTorch is only needed in this process, not in the reference workers

    original = os.path.abspath(args.original)
    files = find_images(args.images)
    if args.n and args.n < len(files):
        files = [files[i] for i in np.linspace(0, len(files) - 1, args.n).round().astype(int)]
    paths = [os.path.join(args.images, f) for f in files]
    print(f"{len(paths)} images")

    extractor = QIPExtractor(device=args.device)
    print(f"qipmachine on {extractor.device}")
    extractor.compute(paths[0])  # warm-up (CUDA initialisation)
    ours, our_times = [], []
    for p in paths:
        start = time.perf_counter()
        ours.append(extractor.compute(p))
        our_times.append(time.perf_counter() - start)

    print(f"original code on {args.jobs} process(es) ...")
    tasks = [(p, original) for p in paths]
    if args.jobs > 1:
        with multiprocessing.get_context('spawn').Pool(args.jobs) as pool:
            results = pool.map(_reference_task, tasks, chunksize=1)
    else:
        results = [_reference_task(t) for t in tasks]
    refs = [r[0] for r in results]
    ref_times = [r[1] for r in results]

    if args.save:
        import csv
        for name, rows in (('original', refs), ('qipmachine', ours)):
            with open(f'{args.save}_{name}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['file'] + extractor.columns)
                for file, row in zip(files, rows):
                    writer.writerow([file] + [repr(row[c]) for c in extractor.columns])

    print()
    print(f"| QIP | max abs. difference | max rel. difference | NaN mismatches |")
    print(f"|---|---|---|---|")
    for column in extractor.columns:
        a = np.array([r[column] for r in refs])
        b = np.array([o[column] for o in ours])
        both = np.isfinite(a) & np.isfinite(b)
        nan_mismatch = int(np.sum(np.isfinite(a) != np.isfinite(b)))
        diff = np.abs(a[both] - b[both])
        rel = diff / np.maximum(np.abs(a[both]), 1e-300)
        max_abs = f"{diff.max():.2g}" if diff.size else "-"
        max_rel = f"{rel.max():.2g}" if rel.size else "-"
        print(f"| {column} | {max_abs} | {max_rel} | {nan_mismatch} |")

    print()
    print(f"Time per image: original {np.median(ref_times):.1f} s (median), "
          f"qipmachine {np.median(our_times):.2f} s (median, 1 process on {extractor.device})")


if __name__ == '__main__':
    main()
