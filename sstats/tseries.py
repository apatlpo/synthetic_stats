import numpy as np
import xarray as xr
import dask.array as da
import dask.dataframe as dd

from scipy import signal
import statsmodels.api as sm

from scipy.fft import fft, ifft, fftfreq

# notes on generation of data with prescribed autocovariance:
# - Lilly 2017b , section 5
# - http://www.falmity.com/stats/python/c/gpeff/
# - Percival 2006

# from prescribed spectrum:
# - Percival 1992

# Matern process:
# - https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html


def wrapper(
    func,
    time,
    params=None,
    dummy_dims=dict(draw=1),
    chunks=None,
    dtype="float",
    output="xarray",
    name="z",
    **kwargs,
):
    """Wraps timeseries generation code in order to distribute the generation

    Parameters
    ----------
        func: method
            Method wrapped, signature needs to be func(p1, p2, ..., time, draws=1, **kwargs)
            where p1, p2 are dimensioning parameters that are nor time nor draw
            Minimal signature is func(time, draws=1, **kwargs)
        time: int, np.ndarray, tuple
            Number of time steps, time array, tuple (T, dt)
        params: int, dict, optional
            Parameters that will lead to dimensions or required to generate
            time series.
            Each parameter can be a float, a list or a numpy array.
        dummy_dims: dict
            Dummy (extra) dimensions, e.g. dummy_dims={"draw": 10}
        chunks: dict, optional
            Associated chunks
        seed: int, optional
            numpy seed
        name: str, optional
            output name, may be required if multiple variables are correlated
            (dask otherwise will assume they are one and the same)
        **kwargs:
            passed to func
    """

    if isinstance(time, int):
        time = np.arange(time)
    elif isinstance(time, tuple):
        time = np.arange(0.0, time[0], time[1])
    else:
        time = np.array(time)
    Nt = time.size

    dims = {}
    if isinstance(params, dict):
        for d, v in params.items():
            dims[d] = np.array(v, ndmin=1)
    dims["time"] = time
    # dummy dimensions are last, just after the time dimension
    for d, v in dummy_dims.items():
        dims[d] = np.arange(v)
    Nd = len(dims)
    shape = tuple(v.size for d, v in dims.items())

    xr_chunks = {d: "auto" for d in dims}
    xr_chunks["time"] = -1
    if chunks:
        xr_chunks.update(**chunks)
    da_chunks = tuple(xr_chunks[d] for d in dims)

    # Note: this should be overlhauled with xr.map_blocks

    # transform dimensions into dask arrays with appropriate forms
    # dims_da = [da.from_array(dims[d]
    #                .reshape(tuple(dims[d].size if i==j else 1 for j in range(Nd))),
    #                chunks=tuple(xr_chunks[d] if i==j else -1 for j in range(Nd)),
    #                name=name+d
    #                         )
    #           for i, d in enumerate(dims)
    #          ]
    dims_values = [dims[d] for i, d in enumerate(dims)]
    # assert False, dims_da

    # wraps func to reinit numpy seed from chunk number
    def _func(*args, seed=None, block_info=None, **kwargs):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed + block_info[0]["num-chunks"][0])
        # need to subset args to keep only corresponding block
        # skips first, because it is the final array
        ilocations = block_info[0]["array-location"]
        nargs = [a[loc[0] : loc[1]] for a, loc in zip(args[1:], ilocations)]
        return func(*nargs, seed=seed, **kwargs)

    x = da.empty(shape=shape, chunks=da_chunks)
    # Note: adding name to dimension names below is pretty critical if
    #   multiple calls to wrapper are made.
    #   dask will create a single object ... danger
    x = da.map_blocks(_func, x, *dims_values, name=name, **kwargs, dtype=dtype)
    x = x.squeeze()
    dims = {d: v for d, v in dims.items() if v.size > 1}

    # put result in an xarray DataArray
    if output == "xarray":
        x = xr.DataArray(x, dims=tuple(dims), coords=dims).rename(name)
    elif output == "dask_dd":
        assert x.ndim < 3, (
            "Data generated is not 2D and cannot be transformed" + " into a dataframe"
        )
        to_index = lambda d: (
            dd.from_array(dims[d], columns=d).to_frame().set_index(d).index
        )
        if shape[0] == 1:
            i = to_index("time")
            c = "draw"
        else:
            i = to_index("draw")
            c = time
        x = dd.from_dask_array(x, index=i, columns=c)

    return x


def _uniform(low, high, time, *args, seed=None, **kwargs):
    """atomic wrapper around uniform distribution method"""
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros(
        (
            low.size,
            high.size,
        )
        + extra
    )
    for i, l in enumerate(low.flatten()):
        for j, h in enumerate(high.flatten()):
            out[i, j, :, :] = rng.uniform(l, h, extra)
    return out


def uniform(time, low=0.0, high=1.0, draws=1, dummy_dims=None, **kwargs):
    """Generates uniformly distributed numbers

    Parameters
    ----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    low, high: float, list, np.array, optional
        Lower and upper bounds.
        Default values are 0 and 1 respectively
    draws: int
        Number of random realizations

    References
    ----------
    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)
    x = wrapper(
        _uniform,
        time,
        params={"low": low, "high": high},
        dummy_dims=_dummy_dims,
        **kwargs,
    )
    if "low" not in x.dims:
        x = x.assign_attrs(low=low)
    if "high" not in x.dims:
        x = x.assign_attrs(high=high)
    return x


def _normal(loc, scale, time, *args, seed=None, **kwargs):  #
    """atomic wrapper around normal distribution method"""
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((loc.size, scale.size) + extra)
    for i, l in enumerate(loc.flatten()):
        for j, s in enumerate(scale.flatten()):
            out[i, j, ...] = rng.normal(l, s, extra)
    return out


def normal(time, loc=0.0, scale=1.0, draws=1, dummy_dims=None, **kwargs):
    """Generates normally distributed numbers

    Parameters
    ----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    loc, scale: float, list, np.array, optional
        Mean and standard deviation of the distribution
        Default values are 0 and 1 respectively
    draws: int, optional
        Number of random realizations
    dummy_dims: dict
        Additional dummy dimensions

    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)
    x = wrapper(
        _normal,
        time,
        params={"loc": loc, "scale": scale},
        dummy_dims=_dummy_dims,
        **kwargs,
    )
    if "loc" not in x.dims:
        x = x.assign_attrs(loc=loc)
    if "scale" not in x.dims:
        x = x.assign_attrs(scale=scale)
    return x


def _binomial(n, p, time, *args, seed=None, **kwargs):
    """atomic wrapper around binomial distribution method"""
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros(
        (
            n.size,
            p.size,
        )
        + extra
    )
    for i, _n in enumerate(n.flatten()):
        for j, _p in enumerate(p.flatten()):
            out[i, j, :, :] = rng.binomial(_n, _p, extra)
    return out


def binomial(time, n=1, p=0.5, draws=1, dummy_dims=None, **kwargs):
    """Generates binomial random numbers

    Parameters
    ----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    n, p: float, list, np.array, optional
        Parameters of binomial distribution (see numpy doc)
        Default values are 1 and 0.5 respectively
    draws: int
        Number of random realizations

    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)
    x = wrapper(
        _binomial, time, params={"n": n, "p": p}, dummy_dims=_dummy_dims, **kwargs
    )
    if "n" not in x.dims:
        x = x.assign_attrs(n=n)
    if "p" not in x.dims:
        x = x.assign_attrs(p=p)
    return x


def _exp_autocorr(T, rms, time, *args, dt=None, seed=None, **kwargs):
    """exp_autocorr core code. See exp_autocorr doc"""
    # time = float(time.squeeze())
    np.random.seed(seed)
    arma = sm.tsa.arma_generate_sample
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros(
        (
            T.size,
            rms.size,
        )
        + extra
    )
    for i, t in enumerate(T.flatten()):
        for j, r in enumerate(rms.flatten()):
            ar = np.array([1, -1 + dt / t])  # watch for sign
            am = np.array(
                [
                    1,
                ]
            )
            out[i, j, ...] = arma(
                ar,
                am,
                extra,
                axis=0,
                scale=np.sqrt(2 * dt / t) * r,
                **kwargs,
            )
    return out


def exp_autocorr(time, T, rms, draws=1, dummy_dims=None, **kwargs):
    """Generate exponentially correlated time series
    Implemented via ARMA
    x_{t} = x_{t-1} * (1-dt/T) + \sqrt{2*dt/T}*rms *e_t

    see Sawford 1991, Reynolds number effects in Lagrangian stochastic models
    of turbulent dispersion

    Parameters:
    -----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    T: float, iterable
        Decorrelation time scales
    rms: float, iterable
        Desired rms (not exact for each realization)
    draws: int, optional
        Size of the ensemble, default to 1
    seed: int, optional
        numpy seed
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)

    if isinstance(time, tuple):
        dt = time[1]
    else:
        dt = time[1] - time[0]

    x = wrapper(
        _exp_autocorr,
        time,
        params={
            "T": T,
            "rms": rms,
        },
        dummy_dims=_dummy_dims,
        dt=dt,
        **kwargs,
    )
    if "T" not in x.dims:
        x = x.assign_attrs(T=T)
    if "rms" not in x.dims:
        x = x.assign_attrs(rms=rms)

    return x


def _general_autocorr(rms, time, *args, c=None, dt=None, seed=None, **kwargs):
    """exp_autocorr core code. See exp_autocorr doc"""
    # prepare Cholesky decomposition
    t = time.squeeze()
    sigma = c(abs(t[:, None] - t[None, :]))
    # assert False, (t.shape, t, sigma)
    flag = True
    jitter_exp = -20.0
    while flag:
        try:
            L = np.linalg.cholesky(
                sigma + np.eye(time.size) * 10**jitter_exp,
            )
            flag = False
        except:
            jitter_exp += 1
    scale = rms / np.sqrt(c(0))
    #
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((rms.size,) + extra)
    for j, r in enumerate(rms.flatten()):
        noise = rng.normal(0.0, 1.0, extra)
        out[j, ...] = np.einsum("ij,j...->i...", L, noise) * scale
    return out


def general_autocorr(time, c, rms, draws=1, dummy_dims=None, **kwargs):
    """Generate a time series that will verify some autocorrelation
    Implemented with Cholesky decomposition, see:
    https://github.com/apatlpo/synthetic_stats/blob/master/sandbox/autocorrelation_cholesky.ipynb

    Parameters:
    -----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    c: lambda
        Autocorrelation, signature: c(tau)
    rms: float, iterable
        Desired rms (not exact for each realization)
    draws: int, optional
        Size of the ensemble, default to 1
    seed: int, optional
        numpy seed
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)

    if isinstance(time, tuple):
        dt = time[1]
    else:
        dt = time[1] - time[0]

    assert c is not None, "An autocorrelation c must be passed"

    x = wrapper(
        _general_autocorr,
        time,
        params={
            "rms": rms,
        },
        dummy_dims=_dummy_dims,
        dt=dt,
        c=c,
        **kwargs,
    )
    if "rms" not in x.dims:
        x = x.assign_attrs(rms=rms)

    return x


def _spectral_viggiano(T, tau_eta, time, *args, dt=None, n_layers=None, seed=None, **kwargs):
    """spectral_viggiano core code. See spectral_viggiano doc"""
    # prepare fourier decomposition
    #t = time.squeeze()    
    #scale = rms / np.sqrt(c(0))
    #
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    extra_ones = (extra[0],) + tuple(1 for e in extra[1:])
    out = np.zeros((T.size, tau_eta.size) + extra)
    #assert False, extra
    omega = 2*np.pi*fftfreq(time, d=dt).reshape(extra_ones)
    for i, _T in enumerate(T.flatten()):
        for j, _tau_eta in enumerate(tau_eta.flatten()):
            w = rng.normal(0.0, 1.0, extra)
            u_hat = fft(w, axis=0) /(1-1j*omega*_T) /(1-1j*omega*_tau_eta)**(n_layers-1)
            out[i,j, ...] = ifft(u_hat, axis=0).real
    return out


def spectral_viggiano(time, T, tau_eta, n_layers, draws=1, dummy_dims=None, **kwargs):
    """Generate a time series that will verify the autocorrelation (2.14)-(2.18) in Viggiano et al. 2020
    Modelling Lagrangian velocity and acceleration in turbulent flows as infinitely differentiable stochastic processes
    J. Fluid Mech.
    
    Parameters:
    -----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    T: float, iterable
        Long decorrelation timescale
    tau_eta: float, iterable
        Short decorrelation timescale
    n_layers: float, iterable
        Number of layers, 2(n_layers-1) is the high frequency spectral slope
    draws: int, optional
        Size of the ensemble, default to 1
    seed: int, optional
        numpy seed
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)

    if isinstance(time, tuple):
        dt = time[1]
    else:
        dt = time[1] - time[0]

    x = wrapper(
        _spectral_viggiano,
        time,
        params={
            "T": T,
            "tau_eta": tau_eta,
        },
        dummy_dims=_dummy_dims,
        dt=dt,
        n_layers=n_layers,
        **kwargs,
    )

    # normalize time series
    x = x/x.std("time")

    if "T" not in x.dims:
        x = x.assign_attrs(T=T)
    if "tau_eta" not in x.dims:
        x = x.assign_attrs(tau_eta=tau_eta)

    return x

def _spectral(time, *args, dt=None, spectrum=None, seed=None, **kwargs):
    """spectral core code. See spectral doc"""
    # prepare fourier decomposition
    #t = time.squeeze()    
    #scale = rms / np.sqrt(c(0))
    #
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    extra_ones = (extra[0],) + tuple(1 for e in extra[1:])
    out = np.zeros(extra, dtype=complex) # critical to specify complex dtype or imaginary part will be discarded in assignment below
    omega = 2*np.pi*fftfreq(time, d=dt).reshape(extra_ones) # rad/d
    w = rng.normal(0.0, 1.0, extra) + 1j*rng.normal(0.0, 1.0, extra)
    u_hat = fft(w, axis=0) * np.sqrt(spectrum(omega))
    out[...] = ifft(u_hat, axis=0)
    # detects any NaN
    assert not np.isnan(out).any(), out
    return out


def spectral(time, draws=1, dummy_dims=None, spectrum=None, dtype=complex, **kwargs):
    """Generate a time series based on a general spectrum
    
    Parameters:
    -----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    spectrum: lambda
        spectrum shape, takes the frequency in rad/time units (consistently with time variable)
    draws: int, optional
        Size of the ensemble, default to 1
    seed: int, optional
        numpy seed
    """
    _dummy_dims = dict(draw=draws)
    if dummy_dims is not None:
        _dummy_dims.update(**dummy_dims)

    if isinstance(time, tuple):
        dt = time[1]
    else:
        dt = time[1] - time[0]

    x = wrapper(
        _spectral,
        time,
        dummy_dims=_dummy_dims,
        dt=dt,
        spectrum=spectrum,
        **kwargs,
    )

    # normalize time series
    # must be turned off for now because of a bug in dask: https://github.com/dask/dask/issues/5679
    #x = x/x.std("time")
    xm = x.mean("time")
    std = np.sqrt(( np.real(x-xm)**2 + np.imag(x-xm)**2 ).mean("time"))
    x = x/std

    return x