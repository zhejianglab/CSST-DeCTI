import os
import numpy as np
import pandas as pd
import torch
import gc
import fitsio
from fitsio import FITS,FITSHDR

from torch.utils.data import Dataset, DataLoader

# each index is a batch
class Dataset_HST(Dataset):
    def __init__(self, files_gt, files_lq, half_plane, left_quarter=0, files_pr=[]):
        assert( len(files_gt) == len(files_lq))

        self.len = len(files_gt)
        self.files_gt = files_gt
        self.files_lq = files_lq
        self.half_plane = half_plane
        self.left_quarter = left_quarter
        self.files_pr = files_pr

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        path_gt = self.files_gt[index]
        path_lq = self.files_lq[index]
        fits_gt = fitsio.FITS(path_gt)
        fits_lq = fitsio.FITS(path_lq)
        data_gt = fits_gt[1].read()
        data_lq = fits_lq[1].read()
        
        if not self.half_plane:
            bottom_gt = fits_gt[4].read()
            bottom_lq = fits_lq[4].read()
            data_gt = np.concatenate((data_gt, bottom_gt), axis=0)
            data_lq = np.concatenate((data_lq, bottom_lq), axis=0)
        
        if self.left_quarter == 1:
            data_gt = data_gt[:, 0:1024]
            data_lq = data_lq[:, 0:1024]
        elif self.left_quarter == -1:
            data_gt = data_gt[:, -1024:]
            data_lq = data_lq[:, -1024:]
        
        data_gt = np.expand_dims(data_gt, axis=-1)
        data_lq = np.expand_dims(data_lq, axis=-1)
    
        head = fits_gt[0].read_header()

        time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
        expo = head['EXPTIME']
        # dftime = pd.to_datetime(time_str)
        # data_stamp = np.array([dftime.month, 
        #                       dftime.day,
        #                       dftime.year-1996,
        #                       dftime.hour,
        #                       int(dftime.minute) // 15]).astype(np.int64)

        if self.files_pr:
            path_pr = self.files_pr[index]
            fits_pr = fitsio.FITS(path_pr)
            data_pr = fits_pr[1].read()
                
            if not self.half_plane:
                bottom_pr = fits_pr[4].read()
                data_pr = np.concatenate((data_pr, bottom_pr), axis=0)
            if self.left_quarter == 1:
                data_pr = data_pr[:, 0:1024]
            elif self.left_quarter == -1:
                data_pr = data_pr[:, -1024:]
            data_pr = np.expand_dims(data_pr, axis=-1)
            fits_pr.close()
            del fits_pr

        fits_gt.close()
        fits_lq.close()
        del fits_gt
        del fits_lq
        gc.collect()

        #datalen(row) x batchsize x nvar
        #input lq output gt
        if self.files_pr:
            return data_lq, data_gt, data_pr, time_str, path_gt, path_lq, path_pr  
        else:
            return data_lq, data_gt, time_str, expo, path_gt, path_lq
        #input gt output lq
        # return data_gt, data_lq, data_stamp, path_gt