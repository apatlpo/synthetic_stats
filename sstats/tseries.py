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
            params=1,
            chunks=None,
            dtype='float',
            output='xarray',
            name='z',
            **kwargs):
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
            time series
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

    if isinstance(params, dict):
        dims = {}
        for d, v in params.items():
            if d=='draw' and isinstance(v, int):
                dims[d] = np.arange(v)
            else:
                dims[d] = np.array(v, ndmin=1)
    else:
        dims = {'draw': np.array(range(params), ndmin=1)}
    dims['time'] = time
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
    dims_da = tuple(da.from_array(dims[d]
                                  .reshape(tuple(dims[d].size if i==j else 1 for j in range(Nd))),
                                  chunks=tuple(xr_chunks[d] if i==j else -1 for j in range(Nd)),
                                  name=name+d
                                 )
                     for i, d in enumerate(dims)
                    )

    # wraps func to reinit numpy seed from chunk number
    def _func(*args, seed=None, block_info=None, **kwargs):
        if seed is None:
            seed = np.random.randint(0,2**32-1)
        np.random.seed(seed+block_info[0]['num-chunks'][0])
        return func(*args[1:],
                    draws=args[0].shape[-2],
                    seed=seed,
                    **kwargs)

    x = da.empty(shape=shape, chunks=da_chunks)
    dims_da = tuple(d for d in dims_da if d.name!=name+'draw')
    x = x.map_blocks(_func, *dims_da, **kwargs, dtype=dtype)
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

def _uniform(low, high, time, draws=1, seed=None, **kwargs):
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    out = np.zeros((low.size, high.size, draws, time))
    for i, l in enumerate(low[:,0,0,0]):
        for j, h in enumerate(high[0,:,0,0]):
            out[i,j,:,:] = rng.uniform(l, h, (draws, time))
    return out

def uniform(time, low=0., high=1., draws=1, **kwargs):
    """ wraps numpy random methods
    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    x = wrapper(_uniform,
                time,
                params={'low': low, 'high': high, 'draw': draws},
                **kwargs)
    if 'low' not in x.dims:
        x = x.assign_attrs(low=low)
    if 'high' not in x.dims:
        x = x.assign_attrs(high=high)
    return x

def _normal(loc, scale, time, draws=1, seed=None, **kwargs):
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    out = np.zeros((loc.size, scale.size, draws, time))
    for i, l in enumerate(loc[:,0,0,0]):
        for j, s in enumerate(scale[0,:,0,0]):
            out[i,j,:,:] = rng.normal(l, s, (draws, time))
    return out

def normal(time, loc=0., scale=1., draws=1, **kwargs):
    """ wraps numpy random methods
    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    x = wrapper(_normal,
                time,
                params={'loc': loc, 'scale': scale, 'draw': draws},
                **kwargs)
    if 'loc' not in x.dims:
        x = x.assign_attrs(loc=loc)
    if 'scale' not in x.dims:
        x = x.assign_attrs(scale=scale)
    return x

def _binomial(n, p, time, draws=1, seed=None, **kwargs):
    rng = np.random.default_rng(seed=seed)
    if not isinstance(time, int):
        time = time.size
    out = np.zeros((n.size, p.size, draws, time))
    for i, _n in enumerate(n[:,0,0,0]):
        for j, _p in enumerate(p[0,:,0,0]):
            out[i,j,:,:] = rng.binomial(_n, _p, (draws, time))
    return out

def binomial(time, n=1, p=.5, draws=1, **kwargs):
    """ wraps numpy random methods
    https://numpy.org/doc/stable/reference/random/index.html#quick-start
    https://docs.python.org/dev/library/random.html#random.random
    """
    x = wrapper(_binomial,
                time,
                params={'n': n, 'p': p, 'draw': draws},
                **kwargs)
    if 'n' not in x.dims:
        x = x.assign_attrs(n=n)
    if 'p' not in x.dims:
        x = x.assign_attrs(p=p)
    return x

def _exp_autocorr(tau, rms, time, draws=1, dt=None, **kwargs):
    """ exp_autocorr core code. See exp_autocorr doc
    """
    #time = float(time.squeeze())
    arma = sm.tsa.arma_generate_sample
    out = np.zeros((tau.size, rms.size, draws, time.size))
    for i, t in enumerate(tau[:,0,0,0]):
        for j, r in enumerate(rms[0,:,0,0]):
            ar = np.array([1, -1+dt/t]) # watch for sign
            am = np.array([1,])
            out[i,j,:,:] = arma(ar,
                                am,
                                (draws, time.size),
                                axis=-1,
                                scale=np.sqrt(2*dt/t)*r,
                               )
    return out

def exp_autocorr(time, tau, rms, draws=1, **kwargs):
    """Generate exponentially correlated time series
    Implemented via ARMA
    x_{t} = x_{t-1} * (1-dt/tau) + \sqrt{2*dt/tau}*rms *e_t

    Parameters:
    -----------
    time: int, np.ndarray, tuple
        Number of time steps, time array, tuple (T, dt)
    tau: float, iterable
        Decorrelation time scales
    rms: float, iterable
        Desired rms (not exact for each realization)
    draws: int, optional
        Size of the ensemble, default to 1
    seed: int, optional
        numpy seed
    """
    if isinstance(time, tuple):
        dt = time[1]
    else:
        dt = time[1]-time[0]

    x = wrapper(_exp_autocorr,
                time,
                params={'tau': tau, 'rms': rms, 'draw': draws},
                dt=dt,
                **kwargs)
    if 'tau' not in x.dims:
        x = x.assign_attrs(tau=tau)
    if 'rms' not in x.dims:
        x = x.assign_attrs(rms=rms)

    return x

# ---------------------------- analysis ---------------------------------------

def _correlate(v1, v2, dt=None, detrend=False, ufunc=True, **kwargs):
    ''' Compute a lagged correlation between two time series
    These time series are assumed to be regularly sampled in time
    and along the same time line.
    Note: takes the complex conjugate of v2

    Parameters
    ----------

        v1, v2: ndarray, pd.Series
            Time series to correlate, the index must be time if dt is not provided

        dt: float, optional
            Time step

        detrend: boolean, optional
            Turns detrending on or off. Default is False.

    See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html
    '''

    assert v1.shape == v2.shape

    _correlate = np.correlate
    if detrend:
        v1 = signal.detrend(v1)
        v2 = signal.detrend(v2)

    _kwargs = {'mode': 'same'}
    _kwargs.update(**kwargs)

    # loop over all dimensions but the last one to apply correlate
    Ni = v1.shape[:-1]
    # infer number of lags from dummy computation
    i0 = tuple(0 for i in Ni) + np.s_[:,]
    f = _correlate(v1[i0], v2[i0], **_kwargs)
    vv = np.full(Ni+f.shape, np.NaN, dtype=v1.dtype)
    for ii in np.ndindex(Ni):
        f = _correlate(v1[ii + np.s_[:,]], v2[ii + np.s_[:,]], **_kwargs)
        Nj = f.shape
        for jj in np.ndindex(Nj):
            vv[ii + jj] = f[jj]

    # select only positive lags
    vv = vv[...,int(vv.shape[-1]/2):]

    # normalized by number of points
    vv = vv/v1.shape[-1]

    if ufunc:
        return vv
    else:
        lags = np.arange(vv.shape[-1])*dt
        if len(vv.shape)==3:
            vv = vv.transpose((2,1,0))
        elif len(vv.shape)==2:
            vv = vv.transpose((1,0))
        return lags, vv

def correlate(v1, v2, lags=None, **kwargs):
    """ Lagged cross-correlation with xarray objects

    Parameters:
    -----------
    v1, v2: xr.DataArray
        Input arrays, need to have a dimension called 'time'
    detrend: boolean
        Detrend or not arrays
    """

    v1 = v1.chunk({'time': -1})
    v2 = v2.chunk({'time': -1})
    dt = (v1.time[1]-v1.time[0]).values

    if lags is None:
        _v1 = v1.isel(**{d: slice(0,2) for d in v1.dims if d is not 'time'})
        _v2 = v2.isel(**{d: slice(0,2) for d in v2.dims if d is not 'time'})
        lags, _ = _correlate(_v1, _v2, dt=dt, ufunc=False, **kwargs)
        return correlate(v1, v2, lags=lags, **kwargs)
    gufunc_kwargs = dict(output_sizes={'lags': lags.size})
    C = xr.apply_ufunc(_correlate, v1, v2,
                dask='parallelized', output_dtypes=[v1.dtype],
                input_core_dims=[['time'], ['time']],
                output_core_dims=[['lags']],
                dask_gufunc_kwargs=gufunc_kwargs,
                kwargs=kwargs,
                )
    return C.assign_coords(lags=lags).rename(v1.name+'_'+v2.name)
