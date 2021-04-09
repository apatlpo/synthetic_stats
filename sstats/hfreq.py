import numpy as np
import xarray as xr

import sympy as sy
from sympy import Symbol, symbols, \
                  fourier_transform, inverse_fourier_transform, \
                  lambdify, \
                  exp, cos, sin, \
                  pi, oo

#from sympy.abc import omega, t, tau
omega, t, tau = symbols("omega, t, tau", real=True)

T = Symbol('T', positive=True)
U = Symbol('U', positive=True)
sigma = Symbol('sigma', positive=True)

# sympy fourier transform:
#    .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{-2\pi i x k} \mathrm{d} x.

from .tseries import exp_autocorr
#import sstats.tseries as ts # dev

_default_tau_bounds = [0., 100.] # days
_default_omega_bounds = [.01, 6] # cpd
#
_default_low = dict(T=T, U=U)
_default_low_values = dict(T=10, U=1.)
#
_default_high = dict(T=T, U=U, sigma=sigma)
_default_high_values = dict(T=10, U=1., sigma=2.)


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

class signal(object):

    def __init__(self,
                 model,
                 parameters={},
                 parameters_values={},
                 autocorrelation=True,
                 ):
        self.model = model
        self.p = parameters
        self.omega, self.tau = omega, tau
        self.p_values = parameters_values
        if autocorrelation:
            self.autocorrelation = self.init_autocorrelation()
            self.update_spectrum()
        else:
            self.spectrum = self.init_spectrum()
            self.update_autocorrelation()
        self.update_variance()
        #
        self.update_lambdas()

    def init_autocorrelation(self):
        return tau*0

    def init_spectrum(self):
        return omega*0

    def update_autocorrelation(self):
        self.autocorrelation = inverse_fourier_transform(u.spectrum, omega, tau/2/pi)/2/pi

    def update_spectrum(self):
        self.spectrum = fourier_transform(self.autocorrelation, tau, omega/(2*pi))

    def update_variance(self):
        self.variance = self.spectrum.integrate((omega, -oo, oo)) /2/pi

    def update_parameters(self):
        self.p = {str(s):s for s in self.autocorrelation.free_symbols if str(s)!='tau'}

    def update_lambdas(self):
        self.autocorrelation_lbd = lambdify([tau, *self.p],
                                            self.autocorrelation,
                                            'numpy'
                                            )
        self.spectrum_lbd = lambdify([omega, *self.p],
                                     self.spectrum,
                                     'numpy'
                                     )

    def evaluate_autocorrelation(self,
                                 tau=None,
                                 name='autocorrelation',
                                 **parameters_values,
                                 ):
        if tau is None:
            tau = _xr_tau()
        elif isinstance(tau, dict):
            tau = _xr_tau(**omega)
        p_values = self.p_values
        p_values.update(**parameters_values)
        p = [p_values[key] for key in self.p]
        da = (self.autocorrelation_lbd(tau, *p)
                .rename(name)
                .assign_attrs(**parameters_values)
             )
        return da

    def evaluate_spectrum(self,
                          omega=None,
                          name='spectrum',
                          **parameters_values,
                         ):
        if omega is None:
            omega = _xr_omega()
        elif isinstance(omega, dict):
            omega = _xr_omega(**omega)
        p_values = self.p_values
        p_values.update(**parameters_values)
        p = [p_values[key] for key in self.p]
        da = (self.spectrum_lbd(omega, *p)
                .rename(name)
                .assign_attrs(**parameters_values)
             )
        # sympy issues fakely complex number sometime
        da = np.abs(da)
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


class low_frequency_signal(signal):

    def __init__(self,
                 model='exponential',
                 parameters=None,
                 parameters_values=None,
                 ):
        if not parameters:
            p = dict(**_default_high)
        else:
            p = parameters
        if not parameters_values:
            p_values = dict(**_default_high_values)
        else:
            p_values = parameters_values
        super().__init__(model, parameters=p, parameters_values=p_values)

    def init_autocorrelation(self):
        if self.model=='exponential':
            R = self.p['U']**2 * exp(-abs(tau)/self.p['T'])
        else:
            R = tau*0
        return R

    def generate_tseries(self, **kwargs):
        """ Generate synthetic data
        calls tseries methods
        """
        if self.model=='exponential':
            p = dict(time=(100,1/24.),
                     tau=_default_low_values['T'],
                     rms=1.,
                     name='low',
                     )
            p.update(**kwargs)
            da = exp_autocorr(**p)
            # should rename tau into T here
        return da


class high_frequency_signal(signal):

    def __init__(self,
                 model='exponential',
                 parameters=None,
                 parameters_values=None,
                 ):
        if not parameters:
            p = dict(**_default_high)
        else:
            p = parameters
        if not parameters_values:
            p_values = dict(**_default_high_values)
        else:
            p_values = parameters_values
        super().__init__(model, parameters=p, parameters_values=p_values)

    def init_autocorrelation(self):
        if self.model=='exponential':
            R = self.p['U']**2 * exp(-abs(tau)/self.p['T']) \
                * cos(self.p['sigma']*tau)
        elif self.model=='gaussian':
            R = self.p['U']**2 * exp(-(tau/self.p['T'])**2) \
                * cos(self.p['sigma']*tau)
        else:
            R = tau*0
        return R

    def generate_tseries(self, **kwargs):
        """ Generate synthetic data
        calls tseries methods
        """
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name='high'
        if self.model=='exponential':
            p = dict(time=(100,1/24.),
                     tau=_default_high_values['T'],
                     sigma=_default_high_values['sigma'],
                     rms=1.,
                     )
            p.update(**kwargs)
            x = exp_autocorr(**p, name='x')
            if "seed" in p:
                p["seed"] = p["seed"]+1
            y = exp_autocorr(**p, name='y')
            with xr.set_options(keep_attrs=True):
                da = (np.real((x+1j*y)/np.sqrt(2)
                              *np.exp(2*np.pi*1j*p['sigma']*x['time'])
                              )
                      .rename(name)
                      )
            # should rename tau into T here
            da = da.assign_attrs(sigma=p['sigma'])
        return da

def add(*args, model='sum', labels=None, weights=None, auto2spec=True):
    if labels is None:
        labels = [str(i) for i in range(len(args))]
    if weights is None:
        weights = [1 for a in args]
    # create signal
    out = signal(model)
    # update autocorrelation and spectra
    for a, l in zip(args, labels):
        R = a.autocorrelation.copy()
        for s in R.free_symbols:
            if str(s)!='tau':
                R = R.subs(s, sy.Symbol(str(s)+'_'+l, positive=s.is_positive))
        out.autocorrelation = out.autocorrelation + R
    # update spectrum
    out.update_spectrum()
    # update parameter list
    out.update_parameters()
    # update lambdas
    out.update_lambdas()
    return out
