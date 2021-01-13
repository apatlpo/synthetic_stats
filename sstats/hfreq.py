import numpy as np
import xarray as xr

from sympy import Symbol, fourier_transform, lambdify, \
                  exp, cos, sin, pi
from sympy.abc import omega, t, tau

T = Symbol('T', positive=True)
U = Symbol('U', positive=True)

# sympy fourier transform:
#    .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{-2\pi i x k} \mathrm{d} x.

from .tseries import exp_autocorr
#as ts
#import sstats.tseries as ts

_default_tau_bounds = [0., 100.] # days
_default_tau = 10
_default_omega_bounds = [.01, 6] # cpd

def _xr_tau(bounds=None,
            N=100,
           ):
    if bounds is None:
        bounds = _default_tau_bounds
    tau = np.linspace(bounds[0], bounds[1], N)
    da = xr.DataArray(tau,
                      dims=['tau'],
                      coords={'tau': (['tau'], tau)},
                     )
    return da

def _xr_omega(bounds=None,
              N=100,
             ):
    if bounds is None:
        bounds = _default_omega_bounds
    omega = np.linspace(bounds[0], bounds[1], N)
    da = xr.DataArray(omega,
                      dims=['omega'],
                      coords={'omega': (['omega'], omega)},
                     )
    return da

class low_frequency_signal(object):

    def __init__(self,
                 T=T,
                 U=U,
                 model='exponential',
                 ):
        self.T = T
        self.U = U # amplitude
        self.model = model
        if model=='exponential':
            self.autocorrelation = self.U**2 * exp(-abs(tau)/self.T)
            self.spectrum = fourier_transform(self.autocorrelation, tau, omega/(2*pi))
            #
            self.autocorrelation_lbd = lambdify([tau, self.T, self.U], self.autocorrelation, 'numpy')
            self.spectrum_lbd = lambdify([omega, self.T, self.U], self.spectrum, 'numpy')

    def generate_tseries(self, **kwargs):
        """ Generate synthetic data
        calls tseries methods
        """
        if self.model=='exponential':
            _kwargs = dict(time=(100,1/24.),
                           tau=_default_tau,
                           rms=1.,
                           name='low',
                          )
            _kwargs.update(**kwargs)
            da = exp_autocorr(**_kwargs)
        return da

    def evaluate_spectrum(self,
                          omega=None,
                          T=_default_tau,
                          U=1.,
                          name='spectrum',
                         ):
        if omega is None:
            omega = _xr_omega()
        elif isinstance(omega, dict):
            omega = _xr_omega(**omega)
        da = (self.spectrum_lbd(omega, T, U)
                .rename(name)
                .assign_attrs(T=T, U=U)
             )
        return da

    def plot_spectrum(self,
                      eval_kwargs={},
                      add_legend=True,
                      grid=True,
                      **kwargs,
                      ):
        E = self.evaluate_spectrum(**eval_kwargs)
        plt_kwargs = dict(xscale='log',
                          yscale='log',
                          lw=3,
                          label=None,
                          )
        plt_kwargs.update(**kwargs)
        label = plt_kwargs['label']
        hdl = E.plot(**plt_kwargs)
        ax = hdl[0].axes
        if label and add_legend:
            ax.legend()
        if grid:
            ax.grid()
        return hdl
