import numpy as np
from typing import IO
from astropy.io import fits
from astropy.time import Time

from .transform import log_normalize, log_normalize_reverse

__all__ = ["DataManager", "Data_HST_ACS_WFC", "Data_CSST_MSC_SIM", "init_manager"]


class DataManager:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'data type: "{}"'.format(self.name)

    # noinspection PyMethodMayBeStatic
    def open(self, path: str, mode: str = "readonly") -> IO | fits.HDUList:
        return fits.open(path, mode=mode, memmap=True)

    # noinspection PyMethodMayBeStatic
    def close(self, file_pointer: IO | fits.HDUList) -> None:
        file_pointer.close()

    def get_dim(self, file_pointer: IO | fits.HDUList) -> tuple[int, int]:
        return int(file_pointer[0].header["NAXIS2"]), int(file_pointer[0].header["NAXIS1"])  # (ny, nx)

    def get_meta(self, file_pointer: IO | fits.HDUList):
        return list()

    def get_image(self, file_pointer: IO | fits.HDUList):
        return file_pointer[0].data

    def get_column(self, file_pointer: IO | fits.HDUList, index: int):
        return file_pointer[0].data[index, :]

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        return data

    def post_process(self, data: np.ndarray) -> np.ndarray:
        return data

    def read(self, file_pointer: IO | fits.HDUList):
        return self.get_image(file_pointer), self.get_meta(file_pointer)

    def write(self, data: np.ndarray, meta: list | tuple, path: str, overwrite: bool = False, verbose: bool = False):
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(path, overwrite=overwrite)
        if verbose:
            print("file written to {}, in format of {}".format(path, self.name))

# noinspection PyPep8Naming
class Data_HST_ACS_WFC(DataManager):

    def __init__(self):
        super().__init__("hst_acs_wfc")
        self.vmin = -100.0
        self.vmax = 60000.0
        self.tau = 100.0

    def get_dim(self, file_pointer: fits.HDUList) -> tuple[int, int]:
        ny = file_pointer[1].header["NAXIS2"]
        nx = file_pointer[1].header["NAXIS1"] + file_pointer[4].header["NAXIS1"]
        return int(ny), int(nx)

    def get_meta(self, file_pointer: fits.HDUList) -> tuple:
        header = file_pointer[0].header
        timestr = "{}T{}".format(header["DATE-OBS"], header["TIME-OBS"])
        mjd_60k = Time(timestr).mjd - 60000
        texp = header["EXPTIME"]
        return mjd_60k, texp

    def get_image(self, file_pointer: fits.HDUList) -> np.ndarray:
        return np.concatenate((file_pointer[1].data, file_pointer[4].data), axis=0)

    def get_column(self, file_pointer: fits.HDUList, index: int) -> np.ndarray:
        nx1 = file_pointer[1].header["NAXIS1"]
        if index < nx1:
            return file_pointer[1].data[:, index]
        else:
            return file_pointer[4].data[:, index]

    def pre_process(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        dd = np.clip(data, a_min=self.vmin, a_max=self.vmax)
        dd = log_normalize(dd, self.vmin, self.vmax, self.tau)
        return dd

    def post_process(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        return log_normalize_reverse(data, self.vmin, self.vmax, self.tau)

    def write(
        self, data: np.ndarray, meta: list | tuple, path: str, overwrite: bool = False, verbose: bool = False
    ) -> None:
        ny = data.shape[0] // 2
        hdulist = [fits.PrimaryHDU()] + [fits.ImageHDU()] * 6
        hdulist = fits.HDUList(hdulist)
        hdulist[1].data = data[:ny, :]
        hdulist[4].data = data[ny:, :]
        timestr = Time(meta[0] + 60000, format="mjd").to_value(format="fits").split("T")
        hdulist[0].header["DATE-OBS"] = timestr[0]
        hdulist[0].header["TIME-OBS"] = timestr[1]
        hdulist[0].header["EXPTIME"] = meta[1]
        hdulist.writeto(path, overwrite=overwrite)
        if verbose:
            print("file written to {}, in format of {}".format(path, self.name))


# noinspection PyPep8Naming
class Data_CSST_MSC_SIM(DataManager):

    def __init__(self):
        super().__init__("csst_msc_sim")
        self.vmin = -100.0
        self.vmax = 100000.0
        self.tau = 100.0

    def get_dim(self, file_pointer: fits.HDUList) -> tuple[int, int]:
        return int(file_pointer[1].header["NAXIS2"]), int(file_pointer[1].header["NAXIS1"])

    def get_meta(self, file_pointer: fits.HDUList) -> tuple:
        mjd_60k = Time(file_pointer[0].header["DATE-OBS"]).mjd - 60000
        texp = file_pointer[0].header["EXPTIME"]
        return mjd_60k, texp

    def get_image(self, file_pointer: fits.HDUList) -> np.ndarray:
        return file_pointer[1].data

    def get_column(self, file_pointer: fits.HDUList, index: int) -> np.ndarray:
        return file_pointer[1].data[:, index]

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        dd = np.clip(data, a_min=self.vmin, a_max=self.vmax)
        dd = log_normalize(dd, self.vmin, self.vmax, self.tau)
        return dd

    def post_process(self, data: np.ndarray) -> np.ndarray:
        return log_normalize_reverse(data, self.vmin, self.vmax, self.tau)

    def write(
        self, data: np.ndarray, meta: list | tuple, path: str, overwrite: bool = False, verbose: bool = False
    ) -> None:
        hdulist = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU()])
        hdulist[1].data = data
        timestr = Time(meta[0] + 60000, format="mjd").to_value(format="fits")
        hdulist[0].header["DATE-OBS"] = timestr
        hdulist[0].header["EXPTIME"] = meta[1]
        hdulist.writeto(path, overwrite=overwrite)
        if verbose:
            print("file written to {}, in format of {}".format(path, self.name))


def init_manager(manager: str | DataManager = "csst_msc_sim") -> DataManager:

    if isinstance(manager, DataManager):
        return manager
    elif type(manager) is str:
        if manager.lower() == "csst_msc_sim" or manager.lower() == "csst_msc_sim":
            return Data_CSST_MSC_SIM()
        if manager.lower() == "hst_acs_wfc" or manager.lower() == "wfc":
            return Data_CSST_MSC_SIM()

    raise Exception("Invalid data manager name: {}".format(manager))
