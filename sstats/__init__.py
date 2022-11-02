"""
synthetic_stats is not a utility package to generate and analyze random time
series.
"""

from . import tseries, signals, sigp, filtering

__all__ = [
    "tseries",
    "signals",
    "sigp",
    "filtering",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

### utils imports

# from . import tseries
import numpy as np
from scipy.special import kv, gamma

import matplotlib.colors as colors
import matplotlib.cm as cmx


def get_cmap_colors(Nc, cmap="plasma"):
    """load colors from a colormap to plot lines

    Parameters
    ----------
    Nc: int
        Number of colors to select
    cmap: str, optional
        Colormap to pick color from (default: 'plasma')
    """
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=Nc), cmap=cmap)
    return [scalarMap.to_rgba(i) for i in range(Nc)]


# ----------------------- standard autocovariance models ------------------------

# exponential covariance
sigma_exp = lambda tau, sigma0, T: sigma0 * np.exp(-abs(tau / T))
gamma_exp = lambda tau, sigma0, T: sigma0 * (1 - np.exp(-abs(tau / T)))
model_exp = dict(
    gamma=gamma_exp,
    sigma=sigma_exp,
    params=dict(sigma0=1, T=1),
)

# gaussian covariance
sigma_gaussian = lambda tau, sigma0, T: sigma0 * np.exp(-abs(tau / T) ** 2)
gamma_gaussian = lambda tau, sigma0, T: sigma0 * (1 - np.exp(-abs(tau / T) ** 2))
model_gaussian = dict(
    gamma=gamma_gaussian,
    sigma=sigma_gaussian,
    params=dict(sigma0=1, T=1),
)

# Matern covariance
# exponential: kappa = 0.5, gaussian: kappa=infty
sigma_matern = lambda tau, sigma0, T, kappa: (
    sigma0
    * 2 ** (1 - kappa)
    / gamma(kappa)
    * (tau / T) ** kappa
    * kv(kappa, abs(tau / T))
)
gamma_matern = lambda tau, sigma0, T, kappa: sigma0 * (
    1 - 2 ** (1 - kappa) / gamma(kappa) * (tau / T) ** kappa * kv(kappa, abs(tau / T))
)
model_matern = dict(
    gamma=gamma_matern,
    sigma=sigma_matern,
    params=dict(sigma0=1, T=1, kappa=0.5),
)

models = dict(exp=model_exp, gaussian=model_gaussian, matern=model_matern)
