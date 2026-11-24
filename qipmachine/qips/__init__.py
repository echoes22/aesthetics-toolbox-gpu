"""QIP implementations, one module per family.

Each module is a port of the corresponding ``AT/*_qips.py`` module of the
Aesthetics Toolbox (https://github.com/RBartho/Aesthetics-Toolbox). The
numerical algorithms are unchanged; only the expensive steps were moved to
PyTorch (GPU) or vectorised with NumPy.
"""
