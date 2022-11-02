import numpy as np


def analytical_variance_slow(Td, T, omega, std=1.0):
    """Predicts the variance of a filtered signal according to:
    \int_{t-T/2}^{t+T/2} u(t) e^{-i\omega t} dt
    The signal is assumed to be exponentionally decorrelated.

    Parameters:
    -----------
    Td: float, ndarray, xr.DataArray
        Decorrelation time scale
    T: float, ndarray, xr.DataArray
        Length of the time window
    omega: float, ndarray, xr.DataArray
        Frequency of the kernel
    """
    Ip = _analytical_It(Td / T, omega * T)
    return std**2 * Ip


def analytical_variance_stationary(omega_0, T, omega, std=1.0):
    """Predicts the variance of a filtered signal according to:
    \int_{t-T/2}^{t+T/2} u(t) e^{-i\omega t} dt
    The signal is assumed to be a stationary periodical signal with
    frequency omega_0

    Parameters:
    -----------
    omega0: float, ndarray, xr.DataArray
        Frequency of the stationary signal
    T: float, ndarray, xr.DataArray
        Length of the time window
    omega: float, ndarray, xr.DataArray
        Frequency of the kernel
    """
    Ic = _analytical_Ic(omega_0 * T, omega * T)
    return std**2 * Ic / 2.0


def analytical_variance_nonstationary(Td, omega_0, T, omega, std=1.0):
    """Predicts the variance of a filtered signal according to:
    \int_{t-T/2}^{t+T/2} u(t) e^{-i\omega t} dt
    The signal is assumed to be a nonstationary periodical signal with
    frequency omega_0. The complex amplitude is assumed to be exponentionally
    decorrelated.

    Parameters:
    -----------
    Td: float, ndarray, xr.DataArray
        Decorrelation/nonstationary time scale
    omega0: float, ndarray, xr.DataArray
        Frequency of the stationary signal
    T: float, ndarray, xr.DataArray
        Length of the time window
    omega: float, ndarray, xr.DataArray
        Frequency of the kernel
    """
    Ip0 = _analytical_It(Td / T, (omega - omega_0) * T)
    Ip1 = _analytical_It(Td / T, (omega + omega_0) * T)
    return std**2 * (Ip0 + Ip1) / 4.0


def _analytical_It(Td, omega):
    """Core integral used to compute analytical variances"""
    return (
        2
        * Td
        * (
            Td**3 * omega**2
            + Td**2 * omega**2
            + Td
            * (
                -(Td**2) * omega**2 * np.cos(omega)
                - 2 * Td * omega * np.sin(omega)
                + np.cos(omega)
            )
            * np.exp(-1 / Td)
            - Td
            + 1
        )
        / (Td**4 * omega**4 + 2 * Td**2 * omega**2 + 1)
    )


def _analytical_Ic(omega_0, omega):
    """Core integral used to compute analytical variances
    Projection a stationary signal
    """
    # if omega_0==omega:
    #    return ((1/2)*omega_0**2 - 1/4*np.cos(2*omega_0) + 1/4)/omega_0**2
    # else:
    return (
        4
        * (
            (
                omega * np.sin((1 / 2) * omega) * np.cos((1 / 2) * omega_0)
                - omega_0 * np.sin((1 / 2) * omega_0) * np.cos((1 / 2) * omega)
            )
            ** 2
            + (
                omega * np.sin((1 / 2) * omega_0) * np.cos((1 / 2) * omega)
                - omega_0 * np.sin((1 / 2) * omega) * np.cos((1 / 2) * omega_0)
            )
            ** 2
        )
        / (omega**2 - omega_0**2) ** 2
    )
