import numpy as np
import xarray as xr

import sympy as sy
from sympy import Symbol, symbols, \
                  integrate, \
                  fourier_transform, inverse_fourier_transform, \
                  lambdify, \
                  exp, cos, sin, sqrt, \
                  DiracDelta, \
                  pi, oo, I

#from sympy.abc import omega, t, tau

#omega, t, tau = symbols("omega, t, tau", real=True)
tau = Symbol('tau', positive=True)
omega = Symbol('omega', real=True)

T = Symbol('T', positive=True)
U = Symbol('U', positive=True)
Us = Symbol('U_s', positive=True)
sigma = Symbol('sigma', positive=True)

# sympy fourier transform:
#    .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{-2\pi i x k} \mathrm{d} x.

from .tseries import exp_autocorr
#import sstats.tseries as ts # dev

rads = 2*pi # sympy
rad = 2*np.pi

_default_tau_bounds = [0., 100.] # days
_default_omega_bounds = [.01*rad, 6*rad] # rpd
#
_default_low = dict(T=T, U=U)
_default_low_values = dict(T=10, U=1.)
#
_default_high = dict(T=T, U=U, Us=Us, sigma=sigma)
_default_high_values = dict(T=10, U=1., Us=1., sigma=2.*rad)


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
                      name="tau"
                     )
    return da

def _xr_omega(bounds=None,
              N=100,
             ):
    """ returns omega xarray dataarray in radian per day
    """
    if bounds is None:
        bounds = _default_omega_bounds
    omega = np.linspace(bounds[0], bounds[1], N)
    da = xr.DataArray(omega,
                      dims=['omega'],
                      coords={'omega': (['omega'], omega/rad)},
                      name="omega_rad"
                     )
    return da

def _fourier_transform(f, tau, omega):
    """ wrapper around sympy fourier transform required in order
    to obtain proper dirac functions for trigonometric functions

    !! Assumes input function is even

    But this still does not allow to compute the stationary contribution
    to the spectrum: practical solution, add two signals, one having a very
    long nonstationary timescale

    see:
    https://github.com/sympy/sympy/issues/2803
    https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.transforms.fourier_transform

    """
    #fourier_transform(autocorrelation, tau, omega)
    #return (integrate(f*exp(-I*2*pi*omega*tau), (tau, 0, oo))
    #        +integrate(f*exp(-I*2*pi*omega*tau), (tau, -oo, 0))
    #        )
    # assumes f is even
    return integrate(f*2*cos(2*pi*omega*tau), (tau, 0, oo)).simplify()

class signal(object):

    def __init__(self,
                 model,
                 parameters={},
                 parameters_values={},
                 autocorrelation=True,
                 analytical_spectrum=None,
                 ):
        self.model = model
        self.p = parameters
        self.omega, self.tau = omega, tau
        self.p_values = parameters_values
        self.analytical_spectrum = analytical_spectrum
        if autocorrelation:
            self.autocorrelation = self.init_autocorrelation()
            self.update_spectrum(spectrum=analytical_spectrum)
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

    def update_spectrum(self, spectrum=None):
        if spectrum is None:
            self.spectrum = _fourier_transform(self.autocorrelation, tau, omega/(2*pi))
        else:
            self.spectrum = spectrum

    def update_variance(self):
        self.variance = (self.spectrum.integrate((omega, -oo, oo))
                         /2/pi
                         ).simplify()

    def update_parameters(self):
        self.p = {str(s):s for s in self.autocorrelation.free_symbols if str(s)!='tau'}

    def update_lambdas(self):
        _p = list(self.p.values())
        self.autocorrelation_lbd = lambdify([tau, *_p],
                                            self.autocorrelation,
                                            'numpy'
                                            )
        if hasattr(self, "stationary") and self.stationary:
            eps = 1e-2 # less will be problematic
            self._dirac_eps = eps
            # https://fr.wikipedia.org/wiki/Distribution_de_Dirac
            modules = ["numpy",
                       {"DiracDelta":lambda x: np.exp(-np.abs(x/eps))/eps},
                       ]
        else:
            modules = "numpy"
        self.spectrum_lbd = lambdify([omega, *_p],
                                     self.spectrum,
                                     modules
                                     )

    def evaluate_autocorrelation(self,
                                 tau=None,
                                 name='autocorrelation',
                                 **parameters_values,
                                 ):
        if tau is None:
            tau = _xr_tau()
        elif isinstance(tau, dict):
            tau = _xr_tau(**tau)
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
        #print(E)
        hdl = E.plot(**plt_kwargs)
        ax = hdl[0].axes
        if label and add_legend:
            ax.legend()
        if grid:
            ax.grid()
        return hdl


_analytical_spectrum_low = {
    "exponential": ( U**2 *2*T /( T**2*omega**2 + 1 )),
}

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
        #
        # bypass symbolic computation of the spectrum
        spectrum = _analytical_spectrum_low[model]
        #
        super().__init__(model,
                         parameters=p,
                         parameters_values=p_values,
                         analytical_spectrum=spectrum,
                         )

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
                     T=_default_low_values['T'],
                     rms=1.,
                     name='low',
                     )
            p.update(**kwargs)
            da = exp_autocorr(**p)
        return da



# sympy is not able to perform fourier transforms trigonometric functions
# supplying analytical spectral form computed with wolframalpha is easiest

# https://github.com/sympy/sympy/issues/2803
# https://www.wolframalpha.com/input/?i=Fourier+transform+%5Bexp%28-abs%28t%2FT%29%29+cos%28s*t%29%2C+t%2C+omega+%5D
# https://www.wolframalpha.com/input/?i=Fourier+transform+%5B+exp%28-%28t%2FT%29**2%29+cos%28s*t%29%2C+t%2C+omega+%5D
_analytical_spectrum_high = {
    "exponential": ( U**2 *2*T
                      *( T**2 * ( sigma**2 + omega**2 ) + 1 )
                      /( T**4*(sigma**2-omega**2)**2
                         + 2*T**2*(sigma**2+omega**2)
                         + 1
                        )
                      ),
    "gaussian": ( U**2 *sqrt(pi) *T/2
                 *(exp( T * sigma * omega ) + 1)
                 *exp(-T**2*(sigma+omega)**2 /4)
                 ),
    "stationary": ( Us**2 *pi* ( DiracDelta( (sigma-omega) )
                                +DiracDelta( (sigma+omega) )
                                )
                   )
}


class high_frequency_signal(signal):
    """ High frequency signal model

    Parameters:
    -----------
    model: str, optional
    parameters=

    """

    def __init__(self,
                 model='exponential',
                 parameters=None,
                 parameters_values=None,
                 stationary=False,
                 ):
        if not parameters:
            p = dict(**_default_high)
        else:
            p = parameters
        #
        if not parameters_values:
            p_values = dict(**_default_high_values)
        else:
            p_values = parameters_values
        #
        # bypass symbolic computation of the spectrum
        spectrum = _analytical_spectrum_high[model]
        #
        self.stationary = stationary
        if stationary:
            spectrum = spectrum + _analytical_spectrum_high["stationary"]
        else:
            #p["Us"] = 0
            p.pop("Us", None)
            p_values.pop("Us", None)
        #
        super().__init__(model,
                         parameters=p,
                         parameters_values=p_values,
                         analytical_spectrum=spectrum,
                         )

    def init_autocorrelation(self):
        if self.model=='exponential':
            R = (self.p['U']**2 * exp(-abs(tau)/self.p['T'])
                 * cos(self.p['sigma']*tau)
                 )
        elif self.model=='gaussian':
            R = self.p['U']**2 * exp(-(tau/self.p['T'])**2) \
                * cos(self.p['sigma']*tau)
        else:
            R = tau*0
        # add stationary component
        if self.stationary:
            R = R + self.p['Us']**2 * cos(self.p['sigma']*tau)
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
                     T=_default_high_values['T'],
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
    #out.update_spectrum()
    for a, l in zip(args, labels):
        E = a.spectrum.copy()
        for s in E.free_symbols:
            if str(s)!='omega':
                E = E.subs(s, sy.Symbol(str(s)+'_'+l, positive=s.is_positive))
        out.spectrum = out.spectrum + E
    # update parameter list
    out.update_parameters()
    # update lambdas
    out.update_lambdas()
    return out
