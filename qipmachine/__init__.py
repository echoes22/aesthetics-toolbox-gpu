"""GPU-accelerated quantitative image properties (QIPs) of the Aesthetics Toolbox.

>>> from qipmachine import QIPExtractor
>>> QIPExtractor().compute('painting.jpg')
"""

__version__ = '0.1.0'

__all__ = ['QIPExtractor', 'QIPS', 'run']


def __getattr__(name):
    # Lazy imports keep `qip-machine --help` fast and let the CLI configure
    # threading before NumPy and PyTorch are loaded.
    if name in ('QIPExtractor', 'QIPS'):
        from . import pipeline
        return getattr(pipeline, name)
    if name == 'run':
        from .runner import run
        return run
    raise AttributeError(f"module 'qipmachine' has no attribute '{name}'")
