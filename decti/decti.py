import os
import time
import sys
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

from .dataset.dataset import PairedImageFileDataset
from .dataset.manager import DataManager
from .utils import setup_dist_env

__all__ = ["DeCTI"]


class DeCTI:

    def __init__(
        self,
        model: nn.Module,
        data_manager: DataManager,
        model_params: dict = None,
        state_path: str = None,
    ):

        # model and data manager
        self.model = model.to("cpu")
        self.data_manager = data_manager
        self.model_params = dict() if model_params is None else model_params
        if "name" not in self.model_params:
            self.model_params["name"] = type(self.model).__name__
        else:
            if self.model_params["name"] != type(self.model).__name__:
                raise ValueError('Invalid "name" value in "model_params"')
        if "seq_len" not in self.model_params:
            self.model_params["seq_len"] = model.seq_len
        else:
            if self.model_params["seq_len"] != model.seq_len:
                raise ValueError('Invalid "seq_len" value in "model_params"')

        if state_path is not None:
            self.load(state_path)

    def load(self, params_path):

        if not os.path.isfile(params_path):
            raise FileNotFoundError(params_path)
        params = torch.load(params_path, map_location="cpu")
        self.model.load_state_dict(params["model"])
        self.model_params = params["model_params"]
        print('Parameters successfully loaded from {}'.format(params_path))

    def predict(
        self,
        image: np.ndarray,
        batch_size: int = 0,
        device: str = "cpu",
        verbose: bool = False,
    ):

        if image.ndim != 2 or image.shape[0] != self.model.seq_len:
            raise Exception("Invalid input image shape.")

        if not torch.cuda.is_available():
            device = "cpu"
        device = torch.device(device)
        model = self.model.to(device)
        model.eval()

        range_func = trange if verbose else range
        ncol = image.shape[1]
        with torch.no_grad():
            img_in = torch.as_tensor(image).permute(2, 0, 1)  # (ncol, 1, seq_len)

            nbatch_per_img = img_in.shape[0] // batch_size
            if img_in.shape[0] != nbatch_per_img * batch_size:
                nbatch_per_img += 1

            for i in range_func(nbatch_per_img):
                col_start = i * batch_size
                col_end = min(col_start + batch_size, ncol)
                batch_in = torch.tensor(img_in[col_start:col_end, :, :], device=device)
                batch_out = model(batch_in)

                if i == 0:
                    img_out = torch.squeeze(batch_out, dim=1)
                else:
                    img_out = torch.cat([img_out, torch.squeeze(batch_out, dim=1)], dim=0)

        img_out = self.data_manager.post_process(img_out.to("cpu").numpy())
        img_out = np.transpose(img_out).astype(float)

        return img_out

    def batch_predict(
        self,
        input_paths: list[str],
        output_paths: list[str],
        batch_size: int = 256,
        use_gpu: bool = True,
        num_workers: int = 1,
        verbose: bool = True,
        print_step: int = 100,
    ):

        # check input
        n_images = len(input_paths)
        if len(output_paths) != n_images:
            raise Exception("The length of output_paths does not match input_paths.")

        # parallelization settings
        if use_gpu and torch.cuda.is_available():
            use_gpu = True
            backend = "nccl"
        else:
            use_gpu = False
            backend = "gloo"
        world_size, rank, local_rank, _, _ = setup_dist_env(verbose=verbose)

        # initialize parallelization
        dist.init_process_group(backend=backend, init_method="env://")
        if not dist.is_initialized():
            raise Exception("process group not initialized at rank {}".format(rank))

        # assign model to device
        if use_gpu:
            device = torch.device("cuda:{}".format(local_rank))
        else:
            device = torch.device("cpu")
        model = self.model.to(device)
        dist.barrier()
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
        ddp_model.eval()

        # load data
        dist.barrier()
        dataset = PairedImageFileDataset(input_paths, output_paths, self.data_manager, inference_mode=True)
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=num_workers)
        if verbose:
            if rank == 0:
                print('Total number of input images: {}'.format(len(dataset)))
            print("Rank {}: {} images allocated".format(rank, len(dataloader)))

        with torch.no_grad():
            for idx, (image_in, _, meta, _, path_ta) in enumerate(dataloader):

                img_in = torch.as_tensor(image_in).permute(2, 0, 1)  # (ncol, 1, seq_len)
                nbatch_per_img = img_in.shape[0] // batch_size
                if img_in.shape[0] != nbatch_per_img * batch_size:
                    nbatch_per_img += 1
                t0 = time.time()

                for i_in_img in range(nbatch_per_img):
                    col_start = i_in_img * batch_size
                    col_end = min((i_in_img + 1) * batch_size, img_in.shape[0])
                    batch_in = img_in[col_start:col_end, :, :].to(device)

                    batch_out = ddp_model(batch_in)
                    if i_in_img == 0:
                        img_out = torch.squeeze(batch_out, dim=1)
                    else:
                        img_out = torch.cat([img_out, torch.squeeze(batch_out, dim=1)], dim=0)

                    if verbose and (idx+1) % print_step == 0:
                        print('    rank {} image {}/{}: progress {}/{}, {:.2f} min'.format(
                            rank, idx+1, len(dataloader), i_in_img, nbatch_per_img, (time.time() - t0) / 60))

                img_out = self.data_manager.post_process(img_out.to("cpu").numpy())
                img_out = np.transpose(img_out).astype(float)
                meta_out = (meta[0].item(), meta[1].item())
                self.data_manager.write(img_out, meta_out, path_ta[0], overwrite=True, verbose=verbose)
                if rank == 0 and verbose:
                    print("Rank {}: {}/{} processed".format(rank, idx + 1, len(dataloader)))
                    sys.stdout.flush()

        dist.barrier()
        if rank == 0:
            print("Processing finished.")
