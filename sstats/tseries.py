import numpy as np
import xarray as xr
import dask.array as da
import dask.dataframe as dd

from scipy import signal
import statsmodels.api as sm

# notes on generation of data with prescribed autocovariance:
# - Lilly 2017b , section 5
# - http://www.falmity.com/stats/python/c/gpeff/
# - Percival 2006

# from prescribed spectrum:
# - Percival 1992

# Matern process:
# - https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html


def wrapper(func,
            time,
            params=None,
            dummy_dims=dict(draw=1),
            chunks=None,
            dtype='float',
            output='xarray',
            name='z',
            **kwargs,
           ):
    """ Wraps timeseries generation code in order to distribute the generation

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
        time = np.arange(0., time[0], time[1])
    else:
        time = np.array(time)
    Nt = time.size

    dims = {}
    if isinstance(params, dict):
        for d, v in params.items():
            dims[d] = np.array(v, ndmin=1)
    dims['time'] = time
    # dummy dimensions are last, just after the time dimension
    for d, v in dummy_dims.items():
        dims[d] = np.arange(v)
    Nd = len(dims)
    shape = tuple(v.size for d, v in dims.items())

    xr_chunks = {d: 'auto' for d in dims}
    xr_chunks['time'] = -1
    if chunks:
        xr_chunks.update(**chunks)
    da_chunks = tuple(xr_chunks[d] for d in dims)

    # transform dimensions into dask arrays with appropriate forms
    # Note: adding name to dimension names below is pretty critical if
    #   multiple calls to wrapper are made.
    #   dask will create a single object ... danger
    dims_da = [da.from_array(dims[d]
                                  .reshape(tuple(dims[d].size if i==j else 1 for j in range(Nd))),
                                  chunks=tuple(xr_chunks[d] if i==j else -1 for j in range(Nd)),
                                  name=name+d
                                 )
                     for i, d in enumerate(dims)
              ]

    # wraps func to reinit numpy seed from chunk number
    def _func(*args, seed=None, block_info=None, **kwargs):
        if seed is None:
            seed = np.random.randint(0,2**32-1)
        np.random.seed(seed+block_info[0]['num-chunks'][0])
        return func(*args[1:], # skips first, because it is the final array
                    seed=seed,
                    **kwargs)

    x = da.empty(shape=shape, chunks=da_chunks)
    x = da.map_blocks(_func, x, *dims_da, **kwargs, dtype=dtype)
    x = x.squeeze()
    dims = {d: v for d, v in dims.items() if v.size>1}

    # put result in an xarray DataArray
    if output=='xarray':
        x = xr.DataArray(x, dims=tuple(dims), coords=dims).rename(name)
    elif output=='dask_dd':
        assert x.ndim<3, 'Data generated is not 2D and cannot be transformed' \
                +' into a dataframe'
        to_index = lambda d: (dd
                              .from_array(dims[d], columns=d)
                              .to_frame()
                              .set_index(d)
                              .index
                             )
        if shape[0]==1:
            i=to_index('time')
            c='draw'
        else:
            i=to_index('draw')
            c=time
        x = dd.from_dask_array(x, index=i, columns=c)

    return x

def _uniform(low, high, time, *args, seed=None, **kwargs):
    """ atomic wrapper around uniform distribution method
    """
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((low.size, high.size,)+extra)
    for i, l in enumerate(low.flatten()):
        for j, h in enumerate(high.flatten()):
            out[i,j,:,:] = rng.uniform(l, h, extra)
    return out

def uniform(time, low=0., high=1., draws=1, dummy_dims=None, **kwargs):
    """ Generates uniformly distributed numbers

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
    x = wrapper(_uniform,
                time,
                params={'low': low, 'high': high},
                dummy_dims=_dummy_dims,
                **kwargs)
    if 'low' not in x.dims:
        x = x.assign_attrs(low=low)
    if 'high' not in x.dims:
        x = x.assign_attrs(high=high)
    return x

def _normal(loc, scale, time, *args, seed=None, **kwargs): #
    """ atomic wrapper around normal distribution method
    """
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((loc.size, scale.size)+extra)
    for i, l in enumerate(loc.flatten()):
        for j, s in enumerate(scale.flatten()):
            out[i, j, ...] = rng.normal(l, s, extra)
    return out

def normal(time, loc=0., scale=1., draws=1, dummy_dims=None, **kwargs):
    """ Generates normally distributed numbers

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
    x = wrapper(_normal,
                time,
                params={'loc': loc, 'scale': scale},
                dummy_dims=_dummy_dims,
                **kwargs,
               )
    if 'loc' not in x.dims:
        x = x.assign_attrs(loc=loc)
    if 'scale' not in x.dims:
        x = x.assign_attrs(scale=scale)
    return x

def _binomial(n, p, time, *args, seed=None, **kwargs):
    """ atomic wrapper around binomial distribution method
    """
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((n.size, p.size,)+extra)
    for i, _n in enumerate(n.flatten()):
        for j, _p in enumerate(p.flatten()):
            out[i,j,:,:] = rng.binomial(_n, _p, extra)
    return out

def binomial(time, n=1, p=.5, draws=1, dummy_dims=None, **kwargs):
    """ Generates binomial random numbers

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
    x = wrapper(_binomial,
                time,
                params={'n': n, 'p': p},
                dummy_dims=_dummy_dims,
                **kwargs)
    if 'n' not in x.dims:
        x = x.assign_attrs(n=n)
    if 'p' not in x.dims:
        x = x.assign_attrs(p=p)
    return x

def _exp_autocorr(T, rms, time, *args, dt=None, seed=None, **kwargs):
    """ exp_autocorr core code. See exp_autocorr doc
    """
    #time = float(time.squeeze())
    np.random.seed(seed)
    arma = sm.tsa.arma_generate_sample
    extra = (time,) + tuple(a.size for a in args)
    out = np.zeros((T.size, rms.size,)+extra)
    for i, t in enumerate(T.flatten()):
        for j, r in enumerate(rms.flatten()):
            ar = np.array([1, -1+dt/t]) # watch for sign
            am = np.array([1,])
            out[i,j,...] = arma(ar,
                                am,
                                extra,
                                axis=0,
                                scale=np.sqrt(2*dt/t)*r,
                               )
    return out

def exp_autocorr(time, T, rms, draws=1, dummy_dims=None, **kwargs):
    """Generate exponentially correlated time series
    Implemented via ARMA
    x_{t} = x_{t-1} * (1-dt/T) + \sqrt{2*dt/T}*rms *e_t

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
        dt = time[1]-time[0]

    x = wrapper(_exp_autocorr,
                time,
                params={'T': T, 'rms': rms,},
                dummy_dims=_dummy_dims,
                dt=dt,
                **kwargs)
    if 'T' not in x.dims:
        x = x.assign_attrs(T=T)
    if 'rms' not in x.dims:
        x = x.assign_attrs(rms=rms)

    return x
