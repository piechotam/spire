"""
Autoencoder types.
Code from https://github.com/neuroexplicit-saar/Discover-then-Name/blob/main/sparse_autoencoder/sparse_autoencoder/autoencoder/types.py
"""

from typing import NamedTuple

from torch.nn import Parameter


class ResetOptimizerParameterDetails(NamedTuple):
    """Reset Optimizer Parameter Details.

    Details of a parameter that should be reset in the optimizer, when resetting
    it's corresponding dictionary vectors.
    """

    parameter: Parameter
    """Parameter to reset."""

    axis: int
    """Axis of the parameter to reset."""