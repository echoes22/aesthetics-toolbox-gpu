# QIP Machine (GPU)

Fast, GPU-accelerated computation of the **quantitative image properties (QIPs)** of the
[Aesthetics Toolbox](https://github.com/RBartho/Aesthetics-Toolbox) (Redies et al., 2025) for whole folders of images.

Point it at a folder, get one CSV row per image with 47 QIP columns: colour statistics, contrast, entropy,
balance, symmetry, Fourier spectral slopes, fractal dimensions, PHOG self-similarity and complexity,
edge-orientation entropy, CNN-feature QIPs and more.

```bash
qip-machine paintings/ -o paintings_qips.csv
```

The QIP algorithms are those of the Aesthetics Toolbox. The expensive steps (Gabor filter bank and
second-order edge-orientation entropy, CNN features, Fourier spectra) run on the GPU in double precision,
the rest is vectorised, and images are processed in parallel. The results match those of the original
implementation (see [Validation](#validation)), about 30 times faster per image, and a dataset like WikiArt
(81k images) takes hours instead of days.

## Installation

Requires Python ≥ 3.9. Install [PyTorch](https://pytorch.org/get-started/locally/) for your platform first
(choose a CUDA build to use an NVIDIA GPU), then:

```bash
pip install git+https://github.com/echoes22/aesthetics-toolbox-gpu
```

or, from a clone of this repository, `pip install .`

A CUDA GPU is optional: without one everything runs on the CPU (slower, but still much faster than the
original toolbox). Apple-silicon GPUs (MPS) are not supported because the QIPs are computed in float64.

## Usage

```bash
qip-machine IMAGE_FOLDER [-o OUTPUT.csv] [options]
```

The folder is searched recursively for `.jpg .jpeg .png .tif .tiff .bmp .webp .gif` files. Results are
written as soon as each image is done, so an interrupted run can be resumed by running the same command again.
A log file (`OUTPUT.log`) records the settings of each run and any image or QIP that could not be computed.

| Option | Description |
|---|---|
| `-o, --output CSV` | Output file (default: `<folder name>_qips.csv` in the current directory) |
| `--qips NAME ...` | Compute only these QIPs (names from `--list-qips`), e.g. `--qips fourier_slope_redies fourier_sigma` |
| `--exclude NAME ...` | Compute all QIPs except these |
| `--short-side PIXELS` | Downscale images whose shorter side is longer than `PIXELS` before computing the QIPs (see below) |
| `--device DEVICE` | `auto` (default: CUDA GPU if available), `cuda`, `cuda:1`, `cpu` |
| `--workers N` | Number of images processed in parallel (default: chosen from the CPU count and GPU memory) |
| `--shard I/N` | Process only part `I` of `N` of the images (`I` = 0 … N−1), to split a dataset over several jobs |
| `--toolbox-names` | Use the column names of the Aesthetics Toolbox QIP machine (e.g. `Slope Redies`) |
| `--digits N` | Significant digits written to the CSV (default 6) |
| `--overwrite` | Start a new output file instead of resuming |
| `--list-qips` | List the available QIPs |

### Image size

By default QIPs are computed at the full resolution of each image, as in the Aesthetics Toolbox. Several QIPs
depend on resolution (e.g. the Spehar and Mather Fourier slopes, fractal dimensions, PHOG, symmetry and
balance measures), so compare values only between images of similar size, or resize your images
consistently. `--short-side 1000` downscales (Lanczos) every image whose shorter side is longer than
1000 pixels before computing the QIPs, which is equivalent to resizing the files beforehand and is much
faster for large images. `image_size` and `aspect_ratio` always describe the original file.

### Output

One row per image. The `file` column holds the path relative to the input folder; the other columns are:

| Column(s) | QIP |
|---|---|
| `image_size` | Width + height of the original image, in pixels |
| `aspect_ratio` | Width / height of the original image |
| `rms_contrast` | Standard deviation of L* (CIELAB lightness) |
| `luminance_entropy` | Shannon entropy of L* |
| `complexity` | Mean gradient strength (PHOG) |
| `edge_density` | Mean response of a Gabor filter bank |
| `color_entropy` | Shannon entropy of the HSV hue channel |
| `mean_rgb_r`, `mean_rgb_g`, `mean_rgb_b` | Mean of R, G, B (0-255) |
| `mean_lab_l`, `mean_lab_a`, `mean_lab_b` | Mean of L*, a*, b* |
| `mean_hsv_h`, `mean_hsv_s`, `mean_hsv_v` | Mean of H (circular), S, V (0-1) |
| `std_rgb_r`, `std_rgb_g`, `std_rgb_b` | Standard deviation of R, G, B |
| `std_lab_l`, `std_lab_a`, `std_lab_b` | Standard deviation of L*, a*, b* |
| `std_hsv_h`, `std_hsv_s`, `std_hsv_v` | Standard deviation of H (circular), S, V |
| `mirror_symmetry` | Mirror symmetry of the binarised image (%) |
| `dcm_distance`, `dcm_x`, `dcm_y` | Deviation of the centre of mass: distance (%), x and y offsets |
| `balance` | Balance of the binarised image (%) |
| `cnn_symmetry_lr` | Left-right symmetry of CNN (AlexNet conv1) features |
| `cnn_symmetry_ud` | Up-down symmetry of CNN features |
| `cnn_symmetry_lrud` | Left-right & up-down symmetry of CNN features |
| `fourier_slope_redies` | Fourier spectral slope (Redies method) |
| `fourier_slope_spehar` | Fourier spectral slope (Spehar method) |
| `fourier_slope_mather` | Fourier spectral slope (Mather method) |
| `fourier_sigma` | Deviation of the spectrum from a straight line |
| `fractal_dimension_2d` | Box-counting dimension of the binarised image |
| `fractal_dimension_3d` | Box-counting dimension of the grey-level surface |
| `self_similarity_phog` | Self-similarity of PHOG histograms |
| `self_similarity_cnn` | Self-similarity of CNN feature histograms |
| `anisotropy` | Anisotropy of gradient orientations (PHOG) |
| `homogeneity` | Homogeneity of the binarised image (%) |
| `eoe_1st_order` | First-order edge-orientation entropy |
| `eoe_2nd_order` | Second-order edge-orientation entropy |
| `sparseness` | Sparseness of CNN features |
| `variability` | Variability of CNN features |

Empty cells mean that a QIP could not be computed for that image (for example the Mather slope of an image
with large uniform areas); the reason is written to the log file. Use `--toolbox-names` for the column names
of the original toolbox (`img_file`, `Image size (pixels)`, …, `Variability`).

### Python API

```python
from qipmachine import QIPExtractor

extractor = QIPExtractor()                      # all QIPs, GPU if available
qips = extractor.compute("painting.jpg")        # {'image_size': 2200.0, 'aspect_ratio': 1.39, ...}

slopes = QIPExtractor(qips=["fourier_slope_redies", "fourier_sigma"], device="cpu")
slopes.compute("painting.jpg")
```

`QIPExtractor.compute` also accepts a PIL image. To process many images, use the command-line tool (or
`qipmachine.run(...)`), which handles parallelism, resuming and errors.

### Large datasets and clusters

Each worker process uses one CPU core; the GPU work is small, so throughput grows with the number of
CPU cores. To spread a large dataset over several jobs, give each job a different `--shard`:

```bash
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
qip-machine /data/wikiart --shard ${SLURM_ARRAY_TASK_ID}/4 -o wikiart_qips_${SLURM_ARRAY_TASK_ID}.csv
```

and merge the results afterwards:

```bash
python -c "import pandas as pd, glob; pd.concat(map(pd.read_csv, sorted(glob.glob('wikiart_qips_*.csv')))).sort_values('file').to_csv('wikiart_qips.csv', index=False)"
```

On network file systems, copying the images to a local disk first can speed things up considerably.

## Speed

Measured on 200 WikiArt paintings (median 0.6 megapixels, largest 44 megapixels) with 8 CPU cores and a
slice of an NVIDIA H100 GPU (MIG 2g.20gb, about 2/7 of the GPU), at full resolution unless noted:

| | Images per second | Time for WikiArt (81,449 images) |
|---|---|---|
| Original toolbox code, 8 processes in parallel (22.8 s per image each) | 0.35 | ~2.7 days |
| qip-machine, CPU only (`--device cpu`, 8 workers) | 1.2 | ~19 hours |
| qip-machine, GPU (default, 8 workers) | 2.5 | ~9 hours |
| qip-machine, GPU, `--short-side 1000` | 7.1 | ~3 hours |

A single worker takes 0.76 s for a typical image, against 22.8 s for the original code. Each worker uses one
CPU core and about 0.7 GB of GPU memory. Most of the remaining time is CPU work (colour conversions, PHOG,
fractal dimensions), so throughput grows with the number of cores; one GPU is enough.

## Validation

`scripts/compare_with_original.py` runs the original Aesthetics Toolbox code (called exactly like its
`QIP_machine_script.py`) and this package on the same images and reports the largest differences.
On 22 test images (WikiArt and VAPS paintings, plus a square image, a small 300 × 200 image, a grayscale
JPEG, an RGBA PNG, a 21-megapixel image and a size that makes the Redies spectrum 1023 × 1023 pixels), against
the original toolbox at commit `05bc132`:

| QIPs | Largest relative difference |
|---|---|
| Image size, aspect ratio, colour means and standard deviations, RMS contrast, luminance and colour entropy, balance, DCM, homogeneity, mirror symmetry, fractal dimensions, Fourier slopes and sigma, PHOG complexity, anisotropy and self-similarity, edge density | < 10⁻¹² |
| CNN symmetry, CNN self-similarity, sparseness, variability | < 3 · 10⁻⁷ |
| 1st- and 2nd-order edge-orientation entropy | < 3 · 10⁻⁵ (see below) |

No QIP was missing in one implementation but not the other.

The edge-orientation entropies agree to about 10⁻¹⁵ on most images. In images with uniform areas or
straight edges many Gabor filter responses are exactly equal, and which of them the original code ranks
higher depends on floating-point rounding in its convolution. qip-machine recovers the exact responses
(multiples of 10⁻⁴), so these ties are resolved the same way on every device; for such images (7 of 200 WikiArt
paintings we tested) the entropies differ from the original by up to 10⁻⁴ relative, less than the rounding of the
toolbox's own output.

`tests/` checks every vectorised or GPU-ported function against the loop implementation of the original
(`pip install ".[dev]"`, then `pytest tests`).

## Differences from the Aesthetics Toolbox QIP machine

- **Same algorithms and parameters.** The filter bank, histogram, FFT and CNN computations are ports of the
  original code; the remaining differences are floating-point rounding (see above).
- **Failures are isolated.** If one QIP fails for an image, its cells are left empty and the other QIPs are
  still written; images that cannot be read are reported in the log.
- **Output.** Values are written with 6 significant digits (the toolbox rounds to about 3), column names are
  snake_case unless `--toolbox-names` is given, and the `file` column holds the path relative to the input
  folder, so images with the same name in different subfolders are kept apart.
- **Image size and aspect ratio** are those of the original file even when `--short-side` is used.

## Citation

If you use this tool, please cite the Aesthetics Toolbox paper and the original publications of the QIPs you
use (listed in the paper):

> Redies, C., Bartho, R., Koßmann, L., Spehar, B., Hübner, R., Wagemans, J., & Hayn-Leichsenring, G. U. (2025).
> A toolbox for calculating objective image properties in aesthetics research. *Behavior Research Methods*.
> https://doi.org/10.3758/s13428-025-02632-3

## License

MIT. The QIP implementations are derived from the Aesthetics Toolbox (MIT license); see [LICENSE](LICENSE).
