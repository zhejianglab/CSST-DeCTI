import numpy as np
from astropy.io import fits
from astropy.time import Time

from .transform import log_normalize, log_normalize_reverse

__all__ = ["DataManager", "load_manager", "Data_HST_ACS_WFC", "Data_CSST_MSC_SIM"]


class DataManager:

    def __init__(self,
                 name: str,
                 nx: int,
                 ny: int,
                 ):

        self.name = name
        self.nx = nx
        self.ny = ny

    def read(self, path, **kwargs):

        raise NotImplementedError

    def write(self, data, path, **kwargs):

        raise NotImplementedError

    def pre_process(self, data):

        raise NotImplementedError

    def post_process(self, data):

        raise NotImplementedError

    def __str__(self):

        return 'data type: "{}", nx = {}, ny = {}'.format(self.name, self.nx, self.ny)


class Data_HST_ACS_WFC(DataManager):

    def __init__(self):

        super().__init__("hst_acs_wfc", 4096, 2048 * 2)
        self.vmin = -100.0
        self.vmax = 60000.0
        self.tau = 100.0

    def read(self,
             path: str,
             header_only: bool = False,
             ):

        with fits.open(path) as f:
            timestr = "{}T{}".format(f[0].header["DATE-OBS"], f[0].header["TIME-OBS"])
            mjd_60k = Time(timestr).mjd - 60000
            texp = float(f[0].header["EXPTIME"])
            if header_only:
                data = None
            else:
                data = np.concatenate((f[1].data, f[4].data), axis=0)

        return data, mjd_60k, texp

    def write(self,
              data: np.ndarray,
              path: str,
              overwrite: bool = False,
              verbose: bool = False,
              ):

        ny = data.shape[0] // 2
        hdulist = [fits.PrimaryHDU(), ] + [fits.ImageHDU(), ] * 6
        hdulist = fits.HDUList(hdulist)
        hdulist[0].data = data[:ny, :]
        hdulist[3].data = data[ny:, :]
        hdulist.writeto(path, overwrite=overwrite)
        if verbose:
            print("output has written to {}, in format of {}".format(path, self.name))

    def pre_process(self,
                    data: np.ndarray,
                    ):

        data = np.clip(data, a_min=self.vmin, a_max=self.vmax)
        data = log_normalize(data, self.vmin, self.vmax, self.tau)

        return data

    def post_process(self,
                     data: np.ndarray
                     ):

        return log_normalize_reverse(data, self.vmin, self.vmax, self.tau)


class Data_CSST_MSC_SIM(DataManager):

    def __init__(self):

        super().__init__("csst_msc_sim", 9216, 9232)
        self.vmin = -100.0
        self.vmax = 100000.0
        self.tau = 100.0

    def read(self,
             path: str,
             header_only: bool = False,
             ):

        with fits.open(path) as f:
            mjd_60k = Time(f[0].header["DATE-OBS"]).mjd - 60000
            texp = float(f[0].header["EXPTIME"])
            if header_only:
                data = None
            else:
                data = f[1].data

        return data, mjd_60k, texp

    def write(self,
              data: np.ndarray,
              path: str,
              overwrite: bool = False,
              verbose: bool = False,
              ):

        hdulist = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU()])
        hdulist[1].data = data
        hdulist.writeto(path, overwrite=overwrite)
        if verbose:
            print("output has written to {}, in format of {}".format(path, self.name))

    def pre_process(self,
                    data: np.ndarray,
                    ):

        data = np.clip(data, a_min=self.vmin, a_max=self.vmax)
        data = log_normalize(data, self.vmin, self.vmax, self.tau)

        return data

    def post_process(self,
                     data: np.ndarray,
                     ):

        return log_normalize_reverse(data, self.vmin, self.vmax, self.tau)


def load_manager(name: str = "csst_msc_sim",
                 ):

    if name.lower() == "csst_msc_sim":
        output = Data_CSST_MSC_SIM()
    elif name.lower() == "hst_acs_wfc":
        output = Data_HST_ACS_WFC()
    else:
        raise Exception("Invalid data manager name: {}".format(name))

    return output
