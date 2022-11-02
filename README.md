# Synthetic data and statistics

Gathers various notebooks and tools to assess generate synthetic data
and test statistical diagnostics.

Data considered:
- time series (1D)
- spatiotemporal data (2D/3D)

Analysis:
- mean, variance
- autocorrelations
- spectral estimations
- confidence intervals are systematic

Leverages xarray, pandas objects and dask distributed.

Useful ressources:

- [statsmodel](https://www.statsmodels.org/stable/index.html): time series statistical analysis with pandas objects


Various interesting material:

- dask seeding [stackoverflow post](https://stackoverflow.com/questions/56799621/use-dask-to-return-several-dataframes-after-computing-over-a-single-dataframe)
- statsmodel [ARMA generation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html#statsmodels.tsa.arima_process.arma_generate_sample)
- what is a partial autocorrelation [wiki link](https://en.wikipedia.org/wiki/Partial_autocorrelation_function)
- [robust correlations with xarray and dask](http://martin-jung.github.io/post/2018-xarrayregression/)
- time series for scikit tools [post](https://www.ethanrosenthal.com/2018/03/22/time-series-for-scikit-learn-people-part2/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [dask-ml](https://ml.dask.org/hyper-parameter-search.html)
