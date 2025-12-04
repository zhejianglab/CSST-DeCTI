import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits


class Dataset_PairedImageFiles(Dataset):

    def __init__(self, input_paths, target_paths, train=True, read_func=None, proc_func=None, **kwargs):
        
        self.input_paths = input_paths
        self.target_paths = target_paths
        if len(input_paths) != len(target_paths):
            raise Exception('input_paths and target_paths must have the same length')
        
        self.train = train
        if read_func is None:
            self.read_func = fits.getdata
        else:
            self.read_func = read_func
        self.proc_func = proc_func
        self.proc_kwargs = kwargs

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):

        input_path = self.input_paths[index]
        if not os.path.exists(input_path):
            raise Exception('Input file does not exists: {}'.format(input_path))
        input_data, mjd, texp = self.read_func(input_path)
        if self.proc_func is not None:
            input_data = self.proc_func(input_data, **(self.proc_kwargs))
        input_data = np.expand_dims(input_data, axis=-1)  # (ny, nx, 1)
        input_data = torch.as_tensor(input_data.astype(np.float32))

        mjd = torch.tensor(mjd, dtype=torch.float32)
        texp = torch.tensor(texp, dtype=torch.float32)
        meta = {'MJD': mjd, 'EXPTIME': texp}

        target_path = self.target_paths[index]
        if target_path == '':
            raise Exception('Target file must have meaningful file name')
        if self.train:
            target_data, _, _ = self.read_func(target_path)
            if self.proc_func is not None:
                target_data = self.proc_func(target_data, **(self.proc_kwargs))
            target_data = np.expand_dims(target_data, axis=-1)
            target_data = torch.as_tensor(target_data.astype(np.float32))
        else:
            target_data = []

        return input_data, input_path, meta, target_data, target_path


def load_data(data_style, input_paths, target_paths, train=True,
        distributed=True, world_size=1, rank=0, num_workers=1):

    data_set = Dataset_PairedImageFiles(
        input_paths, target_paths, train=train,
        read_func=data_style.read, proc_func=data_style.pre_process)
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_set, shuffle=train, num_replicas=world_size, rank=rank)
        data_loader = DataLoader(
            data_set, batch_size=1, sampler=sampler, num_workers=num_workers)  # make each image as 1 batch
    else:
        sampler = None
        data_loader = DataLoader(
            data_set, batch_size=1, sampler=sampler, num_workers=num_workers, shuffle=train)
    
    return data_set, data_loader, sampler
