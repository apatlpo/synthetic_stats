import numpy as np
import xarray as xr

from scipy import signal, linalg
from scipy.optimize import minimize, LbfgsInvHessProduct
from scipy.stats import chi2

from numba import float64, guvectorize

import xrft


# ---------------------------- cross-correlations ------------------------------

def correlate_np_old(v1, v2, maxlag=None, **kwargs):
    if maxlag is None:
        return np.correlate(v1, v2, **kwargs)/v1.size
    R = np.correlate(v1[maxlag:-maxlag],
                     v2[:],
                     mode="valid",
                     )
    return R/v1[maxlag:-maxlag].size

def correlate_np(u, v,
              biased=True,
              one_sided=True,
              weights=False,
             ):
    """ custom correlation

      corr[lag] = 1/w(lag) sum_lag u(t) x v(t+lag)


    Parameters
    ----------
    u, v: np.array
        Input timeseries, must be of the same length
    biased: boolean, optional
        Returns a biased estimation of the correlation. Default is True
            Biased: corr[lag] = 1/N sum ...
            Unbiased: corr[lag] = 1/(N-lag) sum ...
    one_sided: boolean, optional
        Outputs only positive lag. Default is True
    weights: boolean, optional
        Returns weights. Default is False

    Returns
    -------
    c: np.array
        Autocorrelation
    lag: np.array of int
        Lag in index (nondimensional units)
    w: np.array of int
        Weights used for the calculation of the autocorrelation

    """
    n = u.size
    assert u.size==v.size, "input vectors must have the same size"
    # build arrays of weights
    if biased:
        w = n
    else:
        _w = np.arange(1,n+1)
        w = np.hstack([_w, _w[-2::-1]])
    #
    c = np.correlate(u, v, mode="full") / w
    lag = np.arange(-n+1,n)
    #
    if one_sided:
        c, lag = c[n-1:], lag[n-1:]
        if not biased:
            w = w[n-1:]
    if weights:
        return c, lag, w
    else:
        return c, lag

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

    #_correlate = np.correlate
    #_correlate = correlate_np

    if detrend:
        v1 = signal.detrend(v1)
        v2 = signal.detrend(v2)

    #_kwargs = {'mode': 'same'}
    #_kwargs.update(**kwargs)

    # loop over all dimensions but the last one to apply correlate
    if len(v1.shape)==1:
        vv, lags = correlate_np(v1, v2, **kwargs)
    else:
        Ni = v1.shape[:-1]
        # infer number of lags from dummy computation
        i0 = tuple(0 for i in Ni) + np.s_[:,]
        f, lags = correlate_np(v1[i0], v2[i0], **kwargs)
        vv = np.full(Ni+f.shape, np.NaN, dtype=v1.dtype)
        for ii in np.ndindex(Ni):
            f, _lags = correlate_np(v1[ii + np.s_[:,]],
                                    v2[ii + np.s_[:,]],
                                    **kwargs,
                                    )
            Nj = f.shape
            for jj in np.ndindex(Nj):
                vv[ii + jj] = f[jj]

    # select only positive lags
    #vv = vv[..., int(vv.shape[-1]/2):]

    # normalized by number of points
    #vv = vv/v1.shape[-1]
    # normalization done in correlate_np

    if ufunc:
        return vv
    else:
        #lags = np.arange(vv.shape[-1])*dt
        if len(vv.shape)==3:
            vv = vv.transpose((2,1,0))
        elif len(vv.shape)==2:
            vv = vv.transpose((1,0))
        return lags*dt, vv

def correlate(v1, v2,
              lags=None,
              #maxlag=None,
              **kwargs,
              ):
    """ Lagged cross-correlation with xarray objects

    Parameters:
    -----------
    v1, v2: xr.DataArray
        Input arrays, need to have a dimension called 'time'
    !working? ! lags: np.array
        Array of lags in input array time units
    !not working ! maxlag: float, optional
        Maximum lag in input arrays, index units
    **kwargs:
        Passed to np.correlate
    """

    # make sure time is last
    dims_ordered = [d for d in v1.dims if d!="time"]+ ["time"]
    v1 = v1.transpose(*dims_ordered)
    v2 = v2.transpose(*dims_ordered)
    # rechunk along time
    v1 = v1.chunk({'time': -1})
    v2 = v2.chunk({'time': -1})

    _kwargs = dict(**kwargs)
    _kwargs["dt"] = float( (v1.time[1]-v1.time[0]).values )

    #if maxlag is not None:
    #    _kwargs["maxlag"] = int(maxlag/_kwargs["dt"])
    #print(_kwargs["maxlag"])

    if lags is None:
        _v1 = v1.isel(**{d: slice(0,2) for d in v1.dims if d!='time'})
        _v2 = v2.isel(**{d: slice(0,2) for d in v2.dims if d!='time'})
        lags, _ = _correlate(_v1, _v2, ufunc=False, **_kwargs)
        return correlate(v1, v2, lags=lags, **_kwargs)

    gufunc_kwargs = dict(output_sizes={'lags': lags.size})
    C = xr.apply_ufunc(_correlate, v1, v2,
                dask='parallelized', output_dtypes=[v1.dtype],
                input_core_dims=[['time'], ['time']],
                output_core_dims=[['lags']],
                dask_gufunc_kwargs=gufunc_kwargs,
                kwargs=_kwargs,
                )
    return C.assign_coords(lags=lags).rename(v1.name+'_'+v2.name)

def effective_DOF(sigma, dt, N):
    """ Returns effective degrees of freedom (DOF) for the sample mean
    and variance along with the small sample scaling factor for variance
    References: Bailey and Hammersley 1946, Priestley chapter 5.3.2

    Parameters
    ----------
    sigma: lambda, xr.DataArray
        Autocorrelation function
    dt: float
        sampling interval
    N: int
        Timeseries length
    Returns
    -------
    mean_Ne: float
        Sample mean effective DOF
    variance_Ne: float
        Sample variance effective DOF
    variance_scale: float
        Sample variance scale correction

    """

    if isinstance(sigma, xr.DataArray):
        # transform to lambda
        assert False, "not implemented yet"

    ## sample mean
    # Priestley (5.3.5), general
    lags = np.arange(-N-1,N)
    mean_Ne = N / np.sum( ( 1-np.abs(lags)/N )*sigma(lags*dt) )

    ## sample variance
    # Priestley (5.3.23) with r=0, Gaussian assumption
    #lags = np.arange(-N-1,N)
    variance_Ne = N / np.sum( ( 1-np.abs(lags)/N ) * sigma(lags*dt)**2 )
    # small sample scaling factor - Bayley and Hammersley 1946 (10)
    variance_scale = mean_Ne * (N-1) /N /(mean_Ne-1)

    return mean_Ne, variance_Ne, variance_scale

@guvectorize("(float64[:], float64[:])", "(n) -> (n)", nopython=True)
def _barlett_np_gufunc(R, V):
    """ Autocovariance estimate variance
    Pierce 5.3.23
    """
    N = R.shape[0]
    for r in range(N):
        V[r] = (1 - r/N) * 2 * R[0]**2 /N
        for m in range(N-r-1):
            V[r] += 2*( 1 - (m+r)/N ) * ( R[m]**2 + R[m+r]*R[abs(m-r)] ) /N

def barlett(da, dim):
    """ Variance estimate for the autocovariance estimate
    See Pierce 5.3.23

    Parameters
    ----------
    da: xr.DataArray
        Input autocovariance
    dim: str
        Lag dimension
    """

    V = xr.apply_ufunc(
        _barlett_np_gufunc,  # first the function
        da,  # now arguments in the order expected by 'interp1_np'
        input_core_dims=[[dim],],  # list with one entry per arg
        output_core_dims=[[dim]],  # returned data has one dimension
        dask="parallelized",
        output_dtypes=[
            da.dtype
        ],
    )

    return V.rename(da.name+"_var")

@guvectorize("(float64[:], float64[:])", "(n) -> (n)", nopython=True)
def _svariance_np_gufunc(u, svar):
    """ Semi-variance estimate
    """
    N = u.shape[0]
    for r in range(N):
        svar[r] = 0.5*np.sum((u[r:]-u[:N-r])**2)/(N-r)

def svariance(da, dim):
    """ Semi-variance estimate

    Parameters
    ----------
    da: xr.DataArray
        Input time series
    dim: str
        Time dimension
    """

    dt = float(da[dim][1]-da[dim][0])

    V = xr.apply_ufunc(
        _svariance_np_gufunc,  # first the function
        da,  # now arguments in the order expected by 'interp1_np'
        input_core_dims=[[dim],],  # list with one entry per arg
        output_core_dims=[["lags"]],  # returned data has one dimension
        dask="parallelized",
        output_dtypes=[
            da.dtype
        ],
    )

    V["lags"] = ("lags", np.arange(V["lags"].size)*dt)

    return V.rename(da.name+"_svar")


_minimize_outputs = ["x", "hess_inv", "jac", "fun"]

def _minimize(*args, x0=None, fun=None, **kwargs):
    """ wrapper around minimized tailored for apply_ufunc call
    in minimize_xr
    """
    res = minimize(fun, x0, args=args, **kwargs)
    if isinstance(res["hess_inv"], LbfgsInvHessProduct):
        res["hess_inv"] = res["hess_inv"].todense()
    res["hess_inv"] = np.diag(res["hess_inv"])
    return tuple(res[o] for o in _minimize_outputs)

def minimize_xr(fun, ds, params, variables, dim, **kwargs):
    """ xarray wrapper around minimize

    Parameters
    ----------
    fun: method
        signature must look like: fun(x, *args)
    ds: xr.Dataset
        dataset containing `args` used by fun
    params: dict
        dict of parameters that will be fitted with values
        corresponding to initial guesses
    variables: list
        list of variables passed as `args` to fun
    dim: str
        core dimension passed to apply_ufunc
    """

    labels = list(params)
    _x0 = np.array([v for k, v in params.items()])

    res = xr.apply_ufunc(
        _minimize,
        *[ds[v] for v in variables],
        input_core_dims=[[dim],]*len(variables),
        output_core_dims=[["parameters"]]*3+[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64]*len(_minimize_outputs),
        kwargs=dict(x0=_x0, fun=fun, **kwargs),
    )

    ds_min = xr.merge([r.rename(o) for r, o in zip(res, _minimize_outputs)])
    ds_min["parameters"] = ("parameters", labels)

    return ds_min

# ---------------------------- spectra-- ---------------------------------------

def xrft_spectrum(da):
    ps = (xrft
          .power_spectrum(da.chunk({"time": 24*40}),
                          dim=["time"],
                          chunks_to_segments=True,
                          window="hann",
                          window_correction=True,
                         )
          .mean("time_segment")
          .rename(da.name)
          .persist()
         )
    return ps


def welch(x, fs=None, ufunc=True, alpha=0.5, **kwargs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    """
    ax = -1 if ufunc else 0
    assert fs is not None, "fs needs to be provided"
    #
    dkwargs = dict(window="hann", return_onesided= False,
                   detrend = False, scaling= "density",
                  )
    dkwargs.update(kwargs)
    dkwargs["noverlap"] = int(alpha*kwargs["nperseg"])
    f, E = signal.welch(x, fs=fs, axis=ax, **dkwargs)
    #
    if ufunc:
        return E
    else:
        return f, E

def spectrum_welch(v, T=60, **kwargs):
    """

    Parameters:
    -----------
    T: float
        window size in days
    """
    v = v.chunk({"time": -1})
    kwargs["fs"] = 1/float(v.time[1]-v.time[0])
    #print(kwargs["fs"]) # 24
    if "nperseg" in kwargs:
        Nb = kwargs["nperseg"]
    else:
        Nb = int(T * kwargs["fs"])
        kwargs["nperseg"] = Nb
    #print(Nb) # 1440 = 24*60
    if "return_onesided" in kwargs and kwargs["return_onesided"]:
        Nb = int(Nb/2)+1
    #
    f, _ = welch(v.isel(**{d: 0 for d in v.dims if d!="time"}).values,
                 ufunc=False, **kwargs)
    #
    E = xr.apply_ufunc(
        welch,
        v,
        dask="parallelized",
        output_dtypes=[np.float64],
        input_core_dims=[["time"]],
        output_core_dims=[["freq_time"]],
        dask_gufunc_kwargs={"output_sizes": {"freq_time": Nb}},
        kwargs=kwargs,
    )
    E = E.assign_coords(freq_time=f).sortby("freq_time")
    return E, f


# ---------------------------- filtering ---------------------------------------

def bpass_filter(omega,
                 hbandwidth,
                 numtaps,
                 dt,
                 ftype,
                ):
    ''' Wrapper around scipy.signal.firwing

    Parameters:
    -----------
    dt: float
        days
    omega: float
        central frequency, units need to be cycle per time units
    hbandwidth: float
        half bandwidth, units need to be cycle per time units
    numptaps: int
        window size in number of points
    dt: float
        sampling interval
    ftype: str
        filter type
    '''
    #
    if ftype=='firwin':
        h = signal.firwin(numtaps,
                          cutoff=[omega-hbandwidth, omega+hbandwidth],
                          pass_zero=False,
                          fs=1./dt,
                          scale=True,
                         )
    elif ftype=='firwin2':
        h = signal.firwin2(numtaps,
                           [0, omega-hbandwidth, omega-hbandwidth*0.5, omega+hbandwidth*0.5, omega+hbandwidth, 1./2/dt],
                           [0, 0, 1, 1, 0, 0],
                           fs=1./dt,
                          )
    #
    t = np.arange(numtaps)*dt
    return h, t

def filter_response(h, dt):
    ''' Returns the frequency response
    '''
    w, h_hat = signal.freqz(h, worN=8000, fs=1/dt)
    #return h_hat, (w/np.pi)/2/dt
    return h_hat, w


def convolve(x, h=None, hilbert=False):
    """ Convolve an input signal with a kernel
    Optionaly compute the Hilbert transform of the resulting time series
    """
    #x_f = im.convolve1d(x, h, axis=1, mode='constant')
    x_f = signal.filtfilt(h, [1], x, axis=-1)
    if hilbert:
        return signal.hilbert(x_f)
    else:
        return x_f

def filt(v, h, hilbert=False):
    output_dtype = complex if hilbert else float
    gufunc_kwargs = dict(output_sizes={'time': len(v.time)})
    return xr.apply_ufunc(convolve, v, kwargs={'h': h, 'hilbert': hilbert},
                    dask='parallelized', output_dtypes=[output_dtype],
                    input_core_dims=[['time']],
                    output_core_dims=[['time']],
                    dask_gufunc_kwargs = gufunc_kwargs,
                         )

def bpass_demodulate(ds, omega, hbandwidth, T, ftype="firwin"):
    """ create filter, filter time series, hilbert transform, demodulate

    ds: xr.DataArray
        input time series, "time" should be the time dimension
    omega: float
        central frequency, units need to be cycle per time units
    hbandwidth: float
        half bandwidth, units need to be cycle per time units
    T: float
        Window length in ds["time"] units
    """
    dt = float(ds["time"][1]-ds["time"][0])

    h, t  = bpass_filter(omega, hbandwidth, int(T/dt), dt, ftype)
    h_hat, w = filter_response(h, dt)

    exp = np.exp(-1j*2*np.pi*omega*ds["time"])
    for v in ds:
        ds[v+'_bpassed'] = filt(ds[v], h, hilbert=True).persist()
        ds[v+'_demodulated'] = ds[v+'_bpassed']*exp
    ds['exp'] = exp

    return ds, h, h_hat, w

# ---------------------------- Max Likelihood ----------------------------------

def likelihood(u, t, c, *args,
              mu=None,
              sigma0=None,
              jitter=None,
              debug=False,
             ):
    """ evaluate the log likelihood

    Parameters
    ----------
    u: np.array
        data time series
    t: np.array
        time series
    c: method
        returns the autocorrelation (1 at lag 0) as a function of lag and args:
            c(lag, *args)
    *args: floats
        parameters required to evaluate the autocorrelation with c
    mu: float, optional
        Mean of u time series.
        Estimated via analytical profiling if not provided
    sigma0: float, optional
        Variance of u time series.
        Estimated via analytical profiling if not provided
    """

    if isinstance(u, xr.DataArray):
        u = u.values
    if isinstance(t, xr.DataArray):
        t = t.values

    N = u.size

    C = c(t[:,None]-t[None,:], *args)
    # Matern covariance is NaN at the origin
    C = np.nan_to_num(C, nan=1)

    # add jitter
    if jitter:
        C0 = C
        flag=True
        while flag:
            assert jitter<0, "jitter is larger than 0, stops"
            try:
                C = C0 + np.eye(C.shape[0])* 10**jitter
                # Cholesky decomposition
                L_solve = linalg.cho_factor(C, lower=True)
                flag=False
                #print(f"jitter = 10**{jitter}")
            except:
                jitter+=+1
    else:
        L_solve = linalg.cho_factor(C, lower=True)

    # solve for mu
    if mu is None:
        ones = np.ones(C.shape[0])
        w = linalg.cho_solve(L_solve, ones)
        W = w.sum()
        mu_hat = np.dot(w, u) / W
    else:
        mu_hat = mu

    up = u - mu_hat

    # estimate variance sigma0
    if sigma0 is None:
        Cinv_u = linalg.cho_solve(L_solve, up)
        sigma0_hat = np.dot(up, Cinv_u) / N
    else:
        sigma0_hat = sigma0

    # estimate the log likelihood function
    s, logdet = np.linalg.slogdet(C)
    pL = -1/2 * logdet - N/2*np.log(sigma0_hat) - N/2
    # note that np.linalg.det(C) would output 0 here, hence the call to slogdet

    out = dict(L=pL,
               mu=mu_hat,
               sigma0=sigma0_hat,
               jitter=jitter,
              )
    if mu is None:
        out["mu_err"] = np.sqrt( W / sigma0_hat )
    if sigma0 is None:
        out["sigma0_err"] = sigma0_hat / np.sqrt(N)

    if debug:
        out.update(C=C, Cinv=Cinv)

    return out

def likelihood_only(*args, **kwargs):
    # wrapper to feed np.vectorize
    return likelihood(*args, **kwargs)["L"]

likelihood_vec = np.vectorize(likelihood_only, excluded=[0,1,2])

_likelihood_outputs = ["L", "mu", "sigma0", "jitter",]

def likelihood_xr(u, c, params, *args, **kwargs):
    """ Compute the likelihood and profiled parametes (mu, sigma0)

    Parameters
    ----------
    u: xr.DataArray
        Input timeseries
    c: Method
        Autocorrelation, signature: c(lag, *args)
    params: dict
        Dict of autocorrelation parameters values (need to be arrays)
        will be passed to c (!! order issue at the moment !!)
    *args: floats
        Other parameters fed to the autocorrelation
    **kwargs: passed to likelihood method (see associated doc)
    """

    dim = "time" # core dimension
    _params = xr.Dataset(params) # dict to xarray dataset

    def _likelihood_wrapper(u, time,  *args, outputs=None, **kwargs):
        new_args = tuple(float(arg) for arg in args)
        d = likelihood(u, time, c, *new_args, **kwargs)
        return tuple(d[o] for o in outputs)

    outputs = _likelihood_outputs
    if "mu" not in kwargs:
        outputs = outputs + ["mu_err"]
    if "sigma0" not in kwargs:
        outputs = outputs + ["sigma0_err"]

    res = xr.apply_ufunc(_likelihood_wrapper,
                         u, u.time, *[_params[p] for p in params],
                         input_core_dims=[[dim], [dim]] + [[]]*len(params),
                         output_core_dims=[[]]*len(outputs),
                         vectorize=True,
                         dask="parallelized",
                         output_dtypes=[np.float64]*len(outputs),
                         kwargs=dict(outputs=outputs, **kwargs),
                        )

    ds = xr.merge([r.rename(o) for r, o in zip(res, outputs)])

    ds.attrs.update(**kwargs)

    return ds

def likelihood_confidence_ratio(alpha=0.1):
    """ Ratio used to define the confidence ratio
        Prob(L>L_max/c) = 1 - alpha

    Pawitan 2.5
    """
    return np.exp(-chi2.ppf(1-alpha,1)/2)

def compute_hessian_diag(u, t, c, params, deltas, **kwargs):
    """ Manually compute the Hessian diagonal of the log likelihood
    """

    I = {}
    for p, value in params.items():
        _params = dict(**params)
        _params[p] = [value-deltas[p], value, value+deltas[p]]
        ds = likelihood_xr(u, c, _params, **kwargs)
        I[p] = -float(ds.L.differentiate(p).differentiate(p).sel(**{p: value}))

    return I
