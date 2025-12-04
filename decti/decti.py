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

    def __init__(
        self,
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

    def predict(self, image, batch_size=0, device="cpu"):

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

            img_input = self.data_manager.pre_process(image)
            img_input = np.expand_dims(img_input, axis=-1)
            img_input = torch.as_tensor(img_input.astype(np.float32)).to(device)

            for i_in_img in range(nbatch_per_img):
                col_start = i_in_img * batch_size
                col_end = col_start + batch_size

                batch_input = img_input[:, col_start:col_end, :]
                batch_output = model(batch_input)

                if i_in_img == 0:
                    img_output = torch.squeeze(batch_output, axis=-1)
                else:
                    img_output = torch.cat(
                        [img_output, torch.squeeze(batch_output, axis=-1)], axis=1
                    )

        img_output = img_output.to("cpu").numpy()
        img_output = self.data_manager.post_process(img_output)

        return img_output

    def batch_predict(
        self,
        input_paths,
        output_paths,
        loader_workers=1,
        batch_size=256,
        use_gpu=True,
        master_addr="localhost",
        master_port="12345",
        backend="nccl",
        verbose=True,
    ):

        # parallelization settings
        if use_gpu and torch.cuda.is_available():
            print("Use GPU")
        else:
            print("Use CPU")
            backend = "gloo"
        _, rank, local_rank = setup_dist_env(
            master_addr=master_addr, master_port=master_port
        )

        try:
            # initialize parallelization
            dist.init_process_group(backend=backend, init_method="env://")
            if not dist.is_initialized():
                raise Exception(
                    "process group not properly initialized at rank {}".format(rank)
                )

            # assign model to device
            device = (
                torch.device("cuda:{}".format(local_rank))
                if use_gpu
                else torch.device("cpu")
            )
            ddp_model = self.model.to(device)
            dist.barrier()
            ddp_model = DistributedDataParallel(ddp_model, device_ids=[local_rank])
            ddp_model.eval()

            # load data
            dist.barrier()
            _, ev_loader, _ = load_data(
                self.data_manager,
                input_paths,
                output_paths,
                train=False,
                num_workers=loader_workers,
            )

            # cut into batches
            nbatch_per_img = self.data_manager.nx // batch_size
            if self.data_manager.nx > nbatch_per_img * batch_size:
                nbatch_per_img += 1

            # loop for each image
            with torch.no_grad():
                for idx, (img_input, _, _, _, path_output) in enumerate(ev_loader):

                    img_input = torch.squeeze(img_input, dim=0)

                    for i_in_img in range(nbatch_per_img):
                        col_start = i_in_img * batch_size
                        col_end = col_start + batch_size
                        batch_input = img_input[:, col_start:col_end, :]
                        batch_output = ddp_model(batch_input)

                        if i_in_img == 0:
                            img_output = torch.squeeze(batch_output, dim=-1)
                        else:
                            img_output = torch.cat(
                                [img_output, torch.squeeze(batch_output, dim=-1)], dim=1
                            )

                    img_output = self.data_manager.post_process(
                        img_output.to("cpu").numpy()
                    )
                    self.data_manager.write(
                        img_output, path_output[0], overwrite=True, verbose=verbose
                    )
                    if verbose and rank == 0:
                        print(
                            "image {}/{} processed at rank {}".format(
                                idx + 1, len(ev_loader), rank
                            )
                        )
                        sys.stdout.flush()

        finally:
            # finish parallelization in case of exception
            dist.destroy_process_group()
