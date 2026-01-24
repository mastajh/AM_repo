"""EEG feature encoders"""

from .wavelet_encoder import WaveletEncoder
from .riemannian_encoder import RiemannianEncoder

__all__ = ['WaveletEncoder', 'RiemannianEncoder']
