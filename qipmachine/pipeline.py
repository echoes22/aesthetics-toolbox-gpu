"""QIP registry and the per-image extractor."""

import math
import traceback
import warnings
from dataclasses import dataclass

import torch

from .image_io import load_image
from .qips import balance, cnn, color, edge_entropy, fourier, fractal, phog


@dataclass(frozen=True)
class QIP:
    name: str
    group: str  # QIPs of the same group are computed together
    columns: tuple
    toolbox_columns: tuple  # column names used by the Aesthetics Toolbox QIP machine
    description: str


def _qip(name, group, toolbox, description, columns=None):
    toolbox = (toolbox,) if isinstance(toolbox, str) else tuple(toolbox)
    return QIP(name, group, tuple(columns or (name,)), toolbox, description)


# Same order as the QIP machine of the Aesthetics Toolbox
QIPS = [
    _qip('image_size', 'size', 'Image size (pixels)', 'Width + height of the original image, in pixels'),
    _qip('aspect_ratio', 'size', 'Aspect ratio', 'Width / height of the original image'),
    _qip('rms_contrast', 'rms_contrast', 'RMS contrast', 'Standard deviation of L* (CIELAB lightness)'),
    _qip('luminance_entropy', 'luminance_entropy', 'Luminance entropy', 'Shannon entropy of L*'),
    _qip('complexity', 'phog', 'Complexity', 'Mean gradient strength (PHOG)'),
    _qip('edge_density', 'eoe', 'Edge density', 'Mean response of a Gabor filter bank'),
    _qip('color_entropy', 'color_entropy', 'Color entropy', 'Shannon entropy of the HSV hue channel'),
    _qip('mean_rgb', 'mean_rgb', ['mean R channel', 'mean G channel', 'mean B channel (RGB)'],
         'Mean of R, G, B (0-255)', ['mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b']),
    _qip('mean_lab', 'mean_lab', ['mean L channel', 'mean a channel', 'mean b channel (Lab)'],
         'Mean of L*, a*, b*', ['mean_lab_l', 'mean_lab_a', 'mean_lab_b']),
    _qip('mean_hsv', 'mean_hsv', ['mean H channel', 'mean S channel', 'mean V channel'],
         'Mean of H (circular), S, V (0-1)', ['mean_hsv_h', 'mean_hsv_s', 'mean_hsv_v']),
    _qip('std_rgb', 'std_rgb', ['std R channel', 'std G channel', 'std B channel'],
         'Standard deviation of R, G, B', ['std_rgb_r', 'std_rgb_g', 'std_rgb_b']),
    _qip('std_lab', 'std_lab', ['std L channel', 'std a channel', 'std b channel (Lab)'],
         'Standard deviation of L*, a*, b*', ['std_lab_l', 'std_lab_a', 'std_lab_b']),
    _qip('std_hsv', 'std_hsv', ['std H channel', 'std S channel', 'std V channel'],
         'Standard deviation of H (circular), S, V', ['std_hsv_h', 'std_hsv_s', 'std_hsv_v']),
    _qip('mirror_symmetry', 'mirror_symmetry', 'Mirror symmetry', 'Mirror symmetry of the binarised image (%)'),
    _qip('dcm', 'dcm', ['DCM distance', 'DCM x position', 'DCM y position'],
         'Deviation of the centre of mass: distance (%), x and y offsets',
         ['dcm_distance', 'dcm_x', 'dcm_y']),
    _qip('balance', 'balance', 'Balance', 'Balance of the binarised image (%)'),
    _qip('cnn_symmetry_lr', 'cnn', 'CNN symmetry left-right', 'Left-right symmetry of CNN (AlexNet conv1) features'),
    _qip('cnn_symmetry_ud', 'cnn', 'CNN symmetry up-down', 'Up-down symmetry of CNN features'),
    _qip('cnn_symmetry_lrud', 'cnn', 'CNN symmetry left-right & up-down',
         'Left-right & up-down symmetry of CNN features'),
    _qip('fourier_slope_redies', 'fourier_redies', 'Slope Redies', 'Fourier spectral slope (Redies method)'),
    _qip('fourier_slope_spehar', 'fourier_spehar', 'Slope Spehar', 'Fourier spectral slope (Spehar method)'),
    _qip('fourier_slope_mather', 'fourier_mather', 'Slope Mather', 'Fourier spectral slope (Mather method)'),
    _qip('fourier_sigma', 'fourier_redies', 'Fourier sigma', 'Deviation of the spectrum from a straight line'),
    _qip('fractal_dimension_2d', 'fractal_2d', '2D Fractal dimension', 'Box-counting dimension of the binarised image'),
    _qip('fractal_dimension_3d', 'fractal_3d', '3D Fractal dimension', 'Box-counting dimension of the grey-level surface'),
    _qip('self_similarity_phog', 'phog', 'Self-similarity (PHOG)', 'Self-similarity of PHOG histograms'),
    _qip('self_similarity_cnn', 'cnn', 'Self-similarity (CNN)', 'Self-similarity of CNN feature histograms'),
    _qip('anisotropy', 'phog', 'Anisotropy', 'Anisotropy of gradient orientations (PHOG)'),
    _qip('homogeneity', 'homogeneity', 'Homogeneity', 'Homogeneity of the binarised image (%)'),
    _qip('eoe_1st_order', 'eoe', '1st-order EOE', 'First-order edge-orientation entropy'),
    _qip('eoe_2nd_order', 'eoe', '2nd-order EOE', 'Second-order edge-orientation entropy'),
    _qip('sparseness', 'cnn', 'Sparseness', 'Sparseness of CNN features'),
    _qip('variability', 'cnn', 'Variability', 'Variability of CNN features'),
]
QIP_BY_NAME = {q.name: q for q in QIPS}


def select_qips(names=None, exclude=None):
    """QIP definitions for the given names (all by default), in canonical order."""
    for name in list(names or []) + list(exclude or []):
        if name not in QIP_BY_NAME:
            raise ValueError(f"Unknown QIP '{name}'. Valid names: {', '.join(QIP_BY_NAME)}")
    selected = set(names) if names else set(QIP_BY_NAME)
    selected -= set(exclude or [])
    return [q for q in QIPS if q.name in selected]


def resolve_device(device='auto'):
    """'auto' (CUDA if available, else CPU), 'cpu', 'cuda' or 'cuda:N' -> torch.device."""
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but PyTorch cannot find a CUDA GPU.")
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
    elif device.type != 'cpu':
        raise ValueError(f"Unsupported device '{device}': QIPs are computed in float64, "
                         "which requires a CUDA GPU or the CPU.")
    return device


################################ QIP groups #################################
# Each function returns {qip name: value or tuple of values} for one group.

def _size(img, device, wanted):
    width, height = img.original_size
    return {'image_size': color.image_size(width, height), 'aspect_ratio': color.aspect_ratio(width, height)}


def _mean_hsv(img, device, wanted):
    circ_mean, _ = color.circ_stats(img.hsv)
    means = color.mean_channels(img.hsv)
    return {'mean_hsv': (circ_mean, means[1], means[2])}


def _std_hsv(img, device, wanted):
    _, circ_std = color.circ_stats(img.hsv)
    stds = color.std_channels(img.hsv)
    return {'std_hsv': (circ_std, stds[1], stds[2])}


def _phog(img, device, wanted):
    self_sim, complexity, anisotropy = phog.phog_qips(img.rgb, img_lab=img.lab)
    return {'self_similarity_phog': self_sim, 'complexity': complexity, 'anisotropy': anisotropy}


def _eoe(img, device, wanted):
    first, second, density = edge_entropy.eoe_qips(img.gray, device)
    return {'eoe_1st_order': first, 'eoe_2nd_order': second, 'edge_density': density}


def _cnn(img, device, wanted):
    out = cnn.cnn_qips(
        img.rgb, device,
        symmetry=bool(wanted & {'cnn_symmetry_lr', 'cnn_symmetry_ud', 'cnn_symmetry_lrud'}),
        sparseness='sparseness' in wanted,
        variability='variability' in wanted,
        self_similarity='self_similarity_cnn' in wanted,
    )
    names = {'symmetry_lr': 'cnn_symmetry_lr', 'symmetry_ud': 'cnn_symmetry_ud',
             'symmetry_lrud': 'cnn_symmetry_lrud', 'self_similarity': 'self_similarity_cnn'}
    return {names.get(k, k): v for k, v in out.items()}


def _fourier_redies(img, device, wanted):
    sigma, slope = fourier.fourier_redies(img.gray, device, bin_size=2, cycles_min=10, cycles_max=256)
    return {'fourier_sigma': sigma, 'fourier_slope_redies': slope}


GROUPS = {
    'size': _size,
    'rms_contrast': lambda img, device, wanted: {'rms_contrast': color.std_channels(img.lab)[0]},
    'luminance_entropy': lambda img, device, wanted: {
        'luminance_entropy': color.shannon_entropy_channel(img.lab[:, :, 0])},
    'color_entropy': lambda img, device, wanted: {
        'color_entropy': color.shannon_entropy_channel(img.hsv[:, :, 0])},
    'mean_rgb': lambda img, device, wanted: {'mean_rgb': tuple(color.mean_channels(img.rgb))},
    'mean_lab': lambda img, device, wanted: {'mean_lab': tuple(color.mean_channels(img.lab))},
    'mean_hsv': _mean_hsv,
    'std_rgb': lambda img, device, wanted: {'std_rgb': tuple(color.std_channels(img.rgb))},
    'std_lab': lambda img, device, wanted: {'std_lab': tuple(color.std_channels(img.lab))},
    'std_hsv': _std_hsv,
    'mirror_symmetry': lambda img, device, wanted: {'mirror_symmetry': balance.mirror_symmetry(img.gray)},
    'dcm': lambda img, device, wanted: {'dcm': balance.dcm(img.gray)},
    'balance': lambda img, device, wanted: {'balance': balance.balance(img.gray)},
    'homogeneity': lambda img, device, wanted: {'homogeneity': balance.homogeneity(img.gray)},
    'cnn': _cnn,
    'fourier_redies': _fourier_redies,
    'fourier_spehar': lambda img, device, wanted: {
        'fourier_slope_spehar': fourier.fourier_slope_spehar(img.gray, device)},
    'fourier_mather': lambda img, device, wanted: {
        'fourier_slope_mather': fourier.fourier_slope_mather(img.rgb, device, img_lab=img.lab)},
    'fractal_2d': lambda img, device, wanted: {'fractal_dimension_2d': fractal.fractal_dimension_2d(img.gray)},
    'fractal_3d': lambda img, device, wanted: {'fractal_dimension_3d': fractal.fractal_dimension_3d(img.gray)},
    'phog': _phog,
    'eoe': _eoe,
}


class QIPExtractor:
    """Computes quantitative image properties (QIPs) of single images.

    Parameters
    ----------
    qips : list of str, optional
        QIP names to compute (see ``QIPS``); all by default.
    device : str
        'auto' (default: CUDA GPU if available, else CPU), 'cuda', 'cuda:N' or 'cpu'.
    short_side : int, optional
        Downscale images whose shorter side is longer than this many pixels
        before computing QIPs. ``image_size`` and ``aspect_ratio`` always
        describe the original image.

    While computing, PyTorch is limited to one CPU thread (the setting is
    restored afterwards). To use several CPU cores, process several images in
    parallel, as the ``qip-machine`` command does.

    Example
    -------
    >>> extractor = QIPExtractor()
    >>> extractor.compute('painting.jpg')['fourier_slope_redies']
    """

    def __init__(self, qips=None, device='auto', short_side=None):
        self.qips = select_qips(qips)
        self.device = resolve_device(device)
        self.short_side = short_side
        self.columns = [c for q in self.qips for c in q.columns]
        self.toolbox_columns = [c for q in self.qips for c in q.toolbox_columns]
        self._wanted = {q.name for q in self.qips}
        self._groups = list(dict.fromkeys(q.group for q in self.qips))

    def compute(self, image):
        """QIPs of one image (path or PIL image) as an ordered {column: float} dict.

        QIPs that cannot be computed for this image are NaN. Raises if the
        image cannot be read.
        """
        return self.compute_with_errors(image)[0]

    def compute_with_errors(self, image):
        """Like :meth:`compute`, also returning [(group, error message, traceback)] for failed QIPs."""
        img = load_image(image, self.short_side)
        results, errors = {}, []
        # With more than one torch CPU thread, NumPy's tiny BLAS calls (e.g. in rgb2lab) can become
        # ~10^4 times slower when NumPy and PyTorch share an OpenMP BLAS (common on HPC clusters).
        # Parallelism comes from processing several images at once instead (see runner.py).
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for group in self._groups:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        results.update(GROUPS[group](img, self.device, self._wanted))
                except Exception as exc:  # one failing QIP should not lose the others
                    errors.append((group, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        finally:
            torch.set_num_threads(torch_threads)

        values = {}
        for q in self.qips:
            value = results.get(q.name)
            if value is None:
                value = (math.nan,) * len(q.columns)
            elif len(q.columns) == 1:
                value = (value,)
            for column, v in zip(q.columns, value):
                values[column] = float(v)
        return values, errors
