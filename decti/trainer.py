import os
import sys
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from matplotlib import pyplot as plt

from .model import make_model
from .dataset import DataManager, load_manager, load_data
from .utils import setup_dist_env, shuffle_paired_lists

__all__ = ["Trainer"]


class Trainer:

    def __init__(
        self,
        input_paths: list[str],
        target_paths: list[str],
        model_param: str | dict = None,
        data_manager: DataManager | str = "csst_msc_sim",
        validate_ratio: float = 0.1,
        num_workers: int = 2,
        batch_size: int = 16,
        find_unused_parameters: bool = False,
        master_addr: str = "localhost",
        master_port: str = "12345",
        backend: str = "nccl",
        random_seed: int = 0,
        file_shuffle_seed: int = 2025,
    ):

        # input file lists
        self.input_paths = input_paths
        self.target_paths = target_paths
        if len(self.input_paths) != len(self.target_paths):
            raise Exception("input_paths and target_paths must have the same length")

        # model
        if model_param is None:
            model_param = dict()
        else:
            if type(model_param) is str:
                model_param = torch.load(model_param)
        self.model, self.model_par = make_model(**model_param)

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

        # file grouping and shuffling
        self.validate_ratio = validate_ratio
        self.input_tr, self.target_tr, self.input_va, self.target_va = (
            shuffle_paired_lists(
                self.input_paths,
                self.target_paths,
                self.validate_ratio,
                seed=file_shuffle_seed,
            )
        )

        # environment
        if not torch.cuda.is_available():
            raise Exception("Not GPU available")
        self.world_size, self.rank, self.local_rank = setup_dist_env(
            master_addr=master_addr, master_port=master_port
        )
        dist.init_process_group(backend=backend)
        if not dist.is_initialized():
            raise Exception(
                "process group not initialized at rank {}".format(self.rank)
            )

        # assign model to device
        self.device = torch.device("cuda:{}".format(self.local_rank))
        self.model = self.model.to(self.device)
        dist.barrier()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=find_unused_parameters,
        )

        # random seed
        self.random_seed = random_seed
        self._init_seeds()

        # optimization
        self.num_workers = num_workers
        self.batch_size = batch_size

        # useful output
        self.train_losses = list()
        self.validate_losses = list()
        self.best_epoch = -1

    def _init_seeds(self):

        seed = self.random_seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(
        self,
        n_epochs: int = 50,
        learning_rate: float = 0.0001,
        loss_function: str = "mse",
        optimizer: str = "Adamax",
        pct_start: float = 0.3,
        patience: int = 10,
        delta: float = 0.0,
        save_epoch_step: int = 5,
        log_path: str = "./log",
        verbose: bool = True,
    ):

        if self.rank == 0:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
        dist.barrier()

        _, tr_loader, tr_sampler = load_data(
            self.data_manager,
            self.input_tr,
            self.target_tr,
            train=True,
            world_size=self.world_size,
            rank=self.rank,
            num_workers=self.num_workers,
        )
        _, va_loader, va_sampler = load_data(
            self.data_manager,
            self.input_va,
            self.target_va,
            train=True,
            world_size=self.world_size,
            rank=self.rank,
            num_workers=self.num_workers,
        )

        # optimizer
        if optimizer == "Adamax":
            model_optim = torch.optim.Adamax(
                [p for p in self.model.parameters() if p.requires_grad == True],
                lr=learning_rate,
            )
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # loss function
        if loss_function == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.SmoothL1Loss(beta=0.5)

        # number of iterations
        nbatch_per_img = self.data_manager.nx // self.batch_size
        train_steps = len(tr_loader) * nbatch_per_img
        total_steps = int(train_steps * n_epochs)
        if self.rank == 0 and verbose:
            print(f"number of batches per image: {nbatch_per_img}")
            print(f"total number of batches: {train_steps}")
            print(f"total number of steps: {total_steps}")

        scheduler = OneCycleLR(
            optimizer=model_optim,
            total_steps=total_steps,
            pct_start=pct_start,
            max_lr=learning_rate,
        )

        early_stopping = None
        if self.rank == 0:
            early_stopping = EarlyStopping(
                patience=patience,
                delta=delta,
                save_epoch_step=save_epoch_step,
                verbose=verbose,
            )
            writer = SummaryWriter(log_dir=log_path, filename_suffix="")
            if verbose:
                print("training start!")
        else:
            writer = None

        i_step = 0
        lr = learning_rate
        self.train_losses, self.validate_losses = list(), list()
        best_epoch = torch.tensor([0]).to(self.device)
        for epoch in range(n_epochs):

            train_loss = []
            self.model.train()

            if tr_sampler is not None:
                tr_sampler.set_epoch(epoch)

            for idx, (img_in, path_in, _, img_ta, path_ta) in enumerate(tr_loader):

                img_in = torch.squeeze(img_in, dim=0)
                img_ta = torch.squeeze(img_ta, dim=0)

                for i_in_img in range(nbatch_per_img):
                    i_step += 1

                    col_start = i_in_img * self.batch_size
                    col_end = col_start + self.batch_size
                    model_optim.zero_grad()
                    batch_in = img_in[:, col_start:col_end, :].to(self.device)
                    batch_ta = img_ta[:, col_start:col_end, :].to(self.device)

                    # calculate loss
                    loss = criterion(self.model(batch_in), batch_ta)
                    train_loss.append(loss.item())

                    if (i_step - 1) % 1 == 0:
                        step_loss_tensor = torch.tensor([loss.item()]).to(self.device)
                        dist.all_reduce(step_loss_tensor)
                        mean_loss = step_loss_tensor.item() / self.world_size

                        if self.rank == 0 and i_step % 50 == 0:
                            if verbose:
                                print(
                                    "\titer: {0}/{1}, epoch: {2}/{3} | loss: {4:.7f}  lr: {5:.7f}".format(
                                        i_step,
                                        total_steps,
                                        epoch + 1,
                                        n_epochs,
                                        loss.item(),
                                        lr,
                                    )
                                )
                                if mean_loss > 5.0:
                                    print(f"bad loss img_path: {path_in} {path_ta}")
                                    print(
                                        f"patch_id: {i_in_img}  learning rate: {str(lr)}"
                                    )
                                sys.stdout.flush()
                            writer.add_scalar("train_loss_step", mean_loss, i_step)

                    loss.backward()
                    model_optim.step()

                    # adjust learning rate
                    lr = scheduler.get_last_lr()[0]
                    for param_group in model_optim.param_groups:
                        param_group["lr"] = lr
                    scheduler.step()

            dist.barrier()

            # loss calc in one epoch
            epoch_loss_mean = np.average(train_loss)
            epoch_loss_tensor = torch.tensor([epoch_loss_mean]).to(self.device)
            dist.all_reduce(epoch_loss_tensor)
            train_loss = epoch_loss_tensor.item() / self.world_size
            self.train_losses.append(train_loss)

            # loss of vali
            val_loss = self.validate(
                va_loader,
                criterion,
                sampler=va_sampler,
                writer=writer,
                epoch=int(epoch + 1),
            )
            self.validate_losses.append(val_loss)

            # earlystop check & loss conclusion
            stop_train = torch.tensor([0]).to(self.device)
            if self.rank == 0:
                writer.add_scalar("train_loss_epoch", train_loss, epoch + 1)
                writer.add_scalar("val_loss_epoch", val_loss, epoch + 1)
                if verbose:
                    print(
                        "Epoch {0}: Train Loss = {1:.7f}, Validate Loss = {2:.7f}".format(
                            epoch + 1, train_loss, val_loss
                        )
                    )
                early_stopping(
                    val_loss, self.model.module.state_dict(), log_path, epoch + 1
                )
                stop_train = torch.tensor([int(early_stopping.early_stop)]).to(
                    self.device
                )
                best_epoch = torch.tensor([int(early_stopping.best_epoch)]).to(
                    self.device
                )
            dist.broadcast(stop_train, src=0)
            dist.broadcast(best_epoch, src=0)
            if stop_train.item():
                if verbose and self.rank == 0:
                    print(
                        "Early stopping: best model is at epoch {}".format(
                            early_stopping.best_epoch + 1
                        )
                    )
                break

        self.best_epoch = best_epoch.item()
        self.load_checkpoint("{}/best_model.pth".format(log_path))
        if self.rank == 0:
            self.plot_loss("{}/loss.png".format(log_path))
        dist.barrier()

    def validate(self, data_loader, criterion, sampler, writer, epoch):

        nbatch_per_img = self.data_manager.nx // self.batch_size
        epoch_loss = []

        self.model.eval()
        with torch.no_grad():
            i_step = 0
            for idx, (img_in, _, _, img_ta, _) in enumerate(data_loader):

                img_in = torch.squeeze(img_in, dim=0)
                img_ta = torch.squeeze(img_ta, dim=0)

                for i_in_img in range(nbatch_per_img):
                    i_step += 1
                    if sampler is not None:
                        sampler.set_epoch(i_step)

                    col_start = i_in_img * self.batch_size
                    col_end = col_start + self.batch_size
                    batch_in = img_in[:, col_start:col_end, :].to(self.device)
                    batch_ta = img_ta[:, col_start:col_end, :].to(self.device)

                    # encoder - decoder
                    loss = criterion(self.model(batch_in), batch_ta).detach().cpu()

                    if writer is not None:
                        writer.add_scalar(f"val_loss step epoch:{epoch}", loss, i_step)
                    epoch_loss.append(loss)

        epoch_loss = np.average(epoch_loss)
        epoch_loss_tensor = torch.tensor([epoch_loss]).to(self.device)
        dist.all_reduce(epoch_loss_tensor)
        epoch_loss = epoch_loss_tensor.item() / self.world_size
        self.model.train()

        return epoch_loss

    def load_checkpoint(self, path):

        state_dict = torch.load(path, map_location="cpu")
        state_dict_module = dict()
        for k, v in state_dict.items():
            state_dict_module["module.{}".format(k)] = v
        self.model.load_state_dict(state_dict_module, strict=True)

    def plot_loss(
        self,
        output_path: str,
        matplotlib_device: str = "agg",
    ):

        matplotlib.use(matplotlib_device)
        x = np.arange(len(self.train_losses)) + 1
        plt.figure(figsize=(8, 6))
        (l1,) = plt.plot(x, self.train_losses, marker="o", color="green", zorder=3)
        (l2,) = plt.plot(x, self.validate_losses, marker="+", color="red", zorder=3)
        if 0 < self.best_epoch < len(self.train_losses):
            ylim = plt.gca().get_ylim()
            plt.plot(
                [self.best_epoch + 1] * 2, ylim, c="orange", lw=2, ls="--", zorder=2
            )
            plt.ylim(ylim)
        plt.legend([l1, l2], ["train", "validate"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def save(
        self,
        output_path: str,
    ):

        if self.rank == 0:
            name = re.sub(r".pth$", "", output_path)
            state_dict = {k: v.cpu() for k, v in self.model.module.state_dict().items()}
            torch.save(state_dict, name + ".state.pth")
            torch.save(self.model_par, name + ".param.pkl")
        dist.barrier()


class EarlyStopping:

    def __init__(self, patience=7, delta=0.0, save_epoch_step=5, verbose=False):

        self.patience = patience
        self.delta = delta
        self.save_epoch_step = save_epoch_step
        self.verbose = verbose

        self.counter = 0
        self.val_loss_min = np.Inf
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, state_dict, logdir, epoch):

        if epoch % self.save_epoch_step == 0:
            torch.save(state_dict, "{}/checkpoint_epoch_{}.pth".format(logdir, epoch))

        if val_loss > self.val_loss_min - self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    "Epoch {}: stopping counter {} out of {}".format(
                        epoch, self.counter, self.patience
                    )
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(
                    "Epoch {}: validate loss decreases {:.6f} --> {:.6f}".format(
                        epoch, self.val_loss_min, val_loss
                    )
                )
            torch.save(state_dict, "{}/best_model.pth".format(logdir))
            self.best_epoch = epoch
            self.val_loss_min = val_loss
            self.counter = 0
