"""
wahkon.core -- Low-level building blocks for the Wahkon network.

Provides Gaussian kernel utilities and the WKN layer.
"""

from .spline import SS_batch, SS_coef2curve, SS_curve2coef
from .layer import WKNLayer
