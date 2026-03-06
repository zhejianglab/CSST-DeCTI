import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .manager import DataManager

__all__ = ["ColumnImagePairDataset", "ImageFileDataset"]


class ColumnImagePairDataset(Dataset):

    def __init__(
        self,
        input_paths: list[str] | tuple[str],
        target_paths: list[str] | tuple[str],
        data_manager: DataManager,
    ):
        # input
        self.dm = data_manager
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.n_files = len(self.input_paths)
        if len(target_paths) != self.n_files:
            raise ValueError("target_paths and input_paths must have same length")
        for i in range(self.n_files):
            if not os.path.exists(self.input_paths[i]):
                raise Exception("input file not found: {}".format(self.input_paths[i]))
            if not os.path.exists(self.target_paths[i]):
                raise Exception("target file not found: {}".format(self.target_paths[i]))

        # get file pointers
        self.fp_input = [self.dm.open(path) for path in self.input_paths]
        self.fp_target = [self.dm.open(path) for path in self.target_paths]
        # extract metadata (only from input)
        self.meta = [self.dm.get_meta(fp) for fp in self.fp_input]

        # get dimensions
        self.ny, self.nx = self.dm.get_dim(self.fp_input[0])
        for i in range(self.n_files):
            ny, nx = self.dm.get_dim(self.fp_input[i])
            if ny != self.ny or nx != self.nx:
                raise ValueError("Shape does not match: {}".format(self.input_paths[i]))
            ny, nx = self.dm.get_dim(self.fp_target[i])
            if ny != self.ny or nx != self.nx:
                raise ValueError("Shape does not match: {}".format(self.target_paths[i]))
        self.nx_total = self.nx * self.n_files

    def __len__(self):
        return self.nx_total

    def __getitem__(self, idx):
        if idx >= self.nx_total:
            raise IndexError("index {} out of boundary: {}".format(idx, self.nx_total))
        idx_file = idx // self.nx
        idx_col = idx - idx_file * self.nx  # strictly match index

        col_in = self.dm.get_column(self.fp_input[idx_file], idx_col)
        col_in = self.dm.pre_process(col_in).astype(np.float32)
        col_in = torch.from_numpy(col_in).float()  # shape = (ny, )

        col_tr = self.dm.get_column(self.fp_target[idx_file], idx_col)
        col_tr = self.dm.pre_process(col_tr).astype(np.float32)
        col_tr = torch.from_numpy(col_tr).float()

        return col_in, col_tr, self.meta[idx_file]

    def __del__(self):
        for i in range(self.n_files):
            self.dm.close(self.fp_input[i])
            self.dm.close(self.fp_target[i])


class ImageFileDataset(Dataset):

    def __init__(
        self,
        input_paths: list[str] | tuple[str],
        output_paths: list[str] | tuple[str],
        data_manager: DataManager,
    ):

        self.dm = data_manager
        self.input_paths = input_paths
        self.output_paths = output_paths
        self.n_files = len(self.input_paths)
        if len(output_paths) != self.n_files:
            raise ValueError("output_paths and input_paths must have same length")
        for i in range(self.n_files):
            if not os.path.exists(self.input_paths[i]):
                raise Exception("input file not found: {}".format(self.input_paths[i]))

        # get file pointers
        self.fp_input = [self.dm.open(path) for path in self.input_paths]

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx):
        data, meta = self.dm.read(self.fp_input[idx])
        data = self.dm.pre_process(data)
        return data, meta, self.output_paths[idx]  # don't use GPU memory

    def __del__(self):
        for i in range(self.n_files):
            self.dm.close(self.fp_input[i])
