import numpy as np
import xarray as xr

from scipy import signal

import xrft


# ---------------------------- cross-correlations ------------------------------


def correlate_np(v1, v2, maxlag=None, **kwargs):
    if maxlag is None:
        return np.correlate(v1, v2, **kwargs)/v1.size
    R = np.correlate(v1[maxlag:-maxlag],
                     v2[:],
                     mode="valid",
                     )
    return R/v1[maxlag:-maxlag].size

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
    _correlate = correlate_np

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
    vv = vv[..., int(vv.shape[-1]/2):]

    # normalized by number of points
    #vv = vv/v1.shape[-1]
    # normalization done in correlate_np

    if ufunc:
        return vv
    else:
        lags = np.arange(vv.shape[-1])*dt
        if len(vv.shape)==3:
            vv = vv.transpose((2,1,0))
        elif len(vv.shape)==2:
            vv = vv.transpose((1,0))
        return lags, vv

def correlate(v1, v2, lags=None, maxlag=None, **kwargs):
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

    v1 = v1.chunk({'time': -1})
    v2 = v2.chunk({'time': -1})

    _kwargs = dict(**kwargs)
    _kwargs["dt"] = float( (v1.time[1]-v1.time[0]).values )

    #if maxlag is not None:
    #    _kwargs["maxlag"] = int(maxlag/_kwargs["dt"])
    #print(_kwargs["maxlag"])

    if lags is None:
        _v1 = v1.isel(**{d: slice(0,2) for d in v1.dims if d is not 'time'})
        _v2 = v2.isel(**{d: slice(0,2) for d in v2.dims if d is not 'time'})
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
    f, _ = welch(v.isel(**{d: 0 for d in v.dims if d is not "time"}).values,
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

def bpass_hilbert(ds, omega, hbandwidth, T, ftype="firwin"):
    """ create filter, filter time series, hilbert transform

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
