import numpy as np


def log_normalize(data, vmin, vmax, tau):

    vspan = vmax - vmin
    tmin = np.log(tau)
    tmax = np.log(vspan + tau)

    return 2 * (np.log(data - vmin + tau) - tmin) / (tmax - tmin) - 1


def log_normalize_reverse(data, vmin, vmax, tau):

    vspan = vmax - vmin
    tmin = np.log(tau)
    tmax = np.log(vspan + tau)

    return np.exp((data + 1) / 2 * (tmax - tmin) + tmin) - tau + vmin
