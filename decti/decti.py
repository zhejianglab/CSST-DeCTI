import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import make_model
from .dataset import DataManager, load_manager, load_data
from .utils import setup_dist_env

__all__ = ["DeCTI"]


class DeCTI:

    def __init__(self,
                 model_param: str | dict = None,
                 model_state: str | dict = None,
                 data_manager: DataManager | str = "csst_msc_sim",
                 ):

        # model
        if model_param is None:
            model_param = dict()
        else:
            if type(model_param) is str:
                model_param = torch.load(model_param)
        self.model, self.model_par = make_model(**model_param)

        if model_state is not None:
            if type(model_state) is str:
                model_state = torch.load(model_state)
            self.model.load_state_dict(model_state, strict=True)
        self.model.eval()

        # data manager
        if type(data_manager) is str:
            self.data_manager = load_manager(name=data_manager)
        elif issubclass(type(data_manager), DataManager):
            self.data_manager = data_manager
        else:
            raise Exception("Invalid data_manager")
        if (
            self.model_par["model_name"] == "dectiabla"
            and self.model_par["data_length"] != self.data_manager.ny
        ):
            raise Exception("model data_length does not match data_manager.ny")

    def predict(self,
                image: np.ndarray,
                batch_size: int = 0,
                device: str = "cpu",
                ):

        model = self.model.to(device)
        model.eval()
        with torch.no_grad():

            if image.ndim != 2 or image.shape[0] != self.data_manager.ny:
                raise Exception("Invalid input image shape.")

            if batch_size == 0:
                batch_size = image.shape[1]
                nbatch_per_img = 1
            else:
                nbatch_per_img = image.shape[1] // batch_size
                if image.shape[1] > nbatch_per_img * batch_size:
                    nbatch_per_img += 1

            img_in = self.data_manager.pre_process(image)
            img_in = np.expand_dims(img_in, axis=-1)
            img_in = torch.as_tensor(img_in.astype(np.float32)).to(device)

            for i_in_img in range(nbatch_per_img):
                col_start = i_in_img * batch_size
                col_end = col_start + batch_size

                batch_in = img_in[:, col_start:col_end, :]
                batch_out = model(batch_in)

                if i_in_img == 0:
                    img_out = torch.squeeze(batch_out, axis=-1)
                else:
                    img_out = torch.cat([img_out, torch.squeeze(batch_out, axis=-1)], axis=1)

        img_out = img_out.to("cpu").numpy()
        img_out = self.data_manager.post_process(img_out)

        return img_out

    def batch_predict(self,
                      input_paths: list[str],
                      output_paths: list[str],
                      loader_workers: int = 1,
                      batch_size: int = 256,
                      use_gpu: bool = True,
                      master_addr: str = "localhost",
                      master_port: str = "12345",
                      backend: str = "nccl",
                      verbose: bool = True,
                      ):

        # parallelization settings
        if use_gpu and torch.cuda.is_available():
            print("Use GPU")
        else:
            print("Use CPU")
            backend = "gloo"
        _, rank, local_rank = setup_dist_env(
            master_addr=master_addr, master_port=master_port)

        try:
            # initialize parallelization
            dist.init_process_group(backend=backend, init_method="env://")
            if not dist.is_initialized():
                raise Exception("process group not initialized at rank {}".format(rank))

            # assign model to device
            device = torch.device("cuda:{}".format(local_rank)) if use_gpu else torch.device("cpu")
            ddp_model = self.model.to(device)
            dist.barrier()
            ddp_model = DistributedDataParallel(ddp_model, device_ids=[local_rank])
            ddp_model.eval()

            # load data
            dist.barrier()
            _, ev_loader, _ = load_data(self.data_manager, input_paths, output_paths,
                    train=False, num_workers=loader_workers)

            # cut into batches
            nbatch_per_img = self.data_manager.nx // batch_size
            if self.data_manager.nx > nbatch_per_img * batch_size:
                nbatch_per_img += 1

            # loop for each image
            with torch.no_grad():
                for idx, (img_in, _, _, _, path_out) in enumerate(ev_loader):

                    img_in = torch.squeeze(img_in, dim=0)

                    for i_in_img in range(nbatch_per_img):
                        col_start = i_in_img * batch_size
                        col_end = col_start + batch_size
                        batch_in = img_in[:, col_start:col_end, :]
                        batch_out = ddp_model(batch_in)

                        if i_in_img == 0:
                            img_out = torch.squeeze(batch_out, dim=-1)
                        else:
                            img_out = torch.cat([img_out, torch.squeeze(batch_out, dim=-1)], dim=1)

                    img_out = self.data_manager.post_process(img_out.to("cpu").numpy())
                    self.data_manager.write(img_out, path_out[0], overwrite=True, verbose=verbose)
                    if verbose and rank == 0:
                        print("image {}/{} processed at rank {}".format(
                                idx + 1, len(ev_loader), rank))
                        sys.stdout.flush()

        finally:
            # finish parallelization in case of exception
            dist.destroy_process_group()
