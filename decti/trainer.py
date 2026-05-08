import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from matplotlib import pyplot as plt

from .dataset.manager import DataManager
from .dataset.dataset import ColumnImagePairDataset
from .utils import setup_dist_env, EarlyStopping

__all__ = ["Trainer"]


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        data_manager: DataManager,
        model_params: dict = None,
        num_workers: int = 1,
        validate_ratio: float = 0.2,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        loss_function: str = "mse",
        optimizer: str = "Adamax",
        pct_start: float = 0.3,
        backend: str = "nccl",
        timeout: int = 10,
        random_seed: int = 0,
        file_shuffle_seed: int = 2025,
    ):

        # model and data manager
        self.model = model
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

        # other parameters
        self.num_workers = num_workers
        self.validate_ratio = validate_ratio
        self.batch_size = batch_size
        self.lr = learning_rate
        self.pct_start = pct_start
        self.random_seed = random_seed
        self.file_shuffle_seed = file_shuffle_seed

        # environment
        if not torch.cuda.is_available():
            raise Exception("Not GPU available")
        self.world_size, self.rank, self.local_rank, _, _ = setup_dist_env(verbose=True)
        timeout = datetime.timedelta(minutes=timeout)
        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)
        if not dist.is_initialized():
            raise Exception("process group not initialized at rank {}".format(self.rank))

        # assign model to device
        self.device = torch.device("cuda:{}".format(self.local_rank))
        self.model = self.model.to(self.device)
        dist.barrier()
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # initialize components
        if loss_function == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.SmoothL1Loss(beta=0.5)

        if optimizer == "Adamax":
            params = [p for p in self.model.parameters() if p.requires_grad == True]
            self.optimizer = torch.optim.Adamax(params, lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # useful objects
        self.scheduler = None
        self.early_stopping = None
        self.writer = None
        self.timer = None

        # useful parameters
        self.train_input = None
        self.train_target = None
        self.valid_input = None
        self.valid_target = None
        self.train_losses = None
        self.valid_losses = None
        self.best_model_dict = None

    def train(
        self,
        input_paths: list[str],
        target_paths: list[str],
        output_dir: str,
        n_epochs: int = 50,
        stop_patience: int = 10,
        stop_delta: float = 0.0,
        save_epoch_step: int = 5,
        verbose: bool = False,
    ):

        logdir = os.path.join(output_dir, "log")
        if self.rank == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            self.writer = SummaryWriter(log_dir=logdir, filename_suffix="")
            self.timer = time.time()
        else:
            self.writer = None  # make sure all others won't write

        self._init_seeds()
        self._divide_sample(input_paths, target_paths)
        if self.rank == 0 and verbose:
            print('Total input files: {}/{} (train/validate)'.format(len(self.train_input), len(self.valid_input)))
        train_loader = self._get_dataloader(self.train_input, self.train_target)
        valid_loader = self._get_dataloader(self.valid_input, self.valid_target, distributed=True, shuffle=False)
        if verbose:
            print('Rank {}: assigned batches: {}/{} (train/validate)'.format(self.rank, len(train_loader), len(valid_loader)))

        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=self.pct_start,
            max_lr=self.lr,
        )
        self.early_stopping = EarlyStopping(
            patience=stop_patience,
            delta=stop_delta,
            verbose=verbose & (self.rank == 0),
        )

        print("Rank {}: training starting".format(self.rank))
        dist.barrier()
        self.train_losses, self.valid_losses = list(), list()
        for epoch in range(n_epochs):
            train_loss = self._run_epoch(train_loader, epoch, is_train=True, verbose=verbose, print_step=30)
            self.train_losses.append(train_loss)
            dist.barrier()
            valid_loss = self._run_epoch(valid_loader, epoch, is_train=False, verbose=verbose, print_step=100)
            self.valid_losses.append(valid_loss)

            # early stopping and best model
            self.early_stopping(epoch, valid_loss)
            if not self.early_stopping.early_stop:
                self.best_model_dict = self.model.module.state_dict()
                for key, value in self.best_model_dict.items():
                    self.best_model_dict[key] = value.cpu()  # move to CPU (save GPU memory)

            if self.rank == 0:
                # save current model
                if (epoch + 1) % save_epoch_step == 0:
                    fcheck_current = os.path.join(logdir, "checkpoint_epoch_{:04d}.pth".format(epoch))
                    self.save(fcheck_current, best=False, full=True)
                    if verbose:
                        print('Epoch {}: current state saved to {}'.format(epoch, fcheck_current))

                # progress monitor
                t = (time.time() - self.timer) / 60
                if verbose:
                    print("Epoch {}: {:.2f} min ({:.2f} min/epoch), train loss {:.9f}, validate loss {:.9f}".format(
                        epoch, t, t / (epoch + 1), train_loss, valid_loss))

            stop_info = torch.tensor([int(self.early_stopping.early_stop)], device=self.device)
            dist.broadcast(stop_info, src=0)
            if stop_info[0].item():
                break

        dist.barrier()
        print("Rank {}: training finished".format(self.rank))
        if self.rank == 0:
            self.save(os.path.join(output_dir, 'checkpoint_last.pth'), best=False, full=True)
            self.save(os.path.join(output_dir, 'best_model.pth'))
            self.plot_loss(os.path.join(output_dir, "loss.png"))
        if self.rank == 0 and self.writer:
            self.writer.close()
            self.writer = None

    def _init_seeds(self):

        seed = self.random_seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _divide_sample(self, input_paths, target_paths):
        # check input
        input_paths = list(input_paths)
        target_paths = list(target_paths)
        if len(input_paths) != len(target_paths):
            raise Exception("input_paths and target_paths must have the same length")

        # shuffle image lists
        combined = list(zip(input_paths, target_paths))
        random.seed(self.file_shuffle_seed)
        random.shuffle(combined)
        input_paths, target_paths = zip(*combined)
        n = len(input_paths)

        # calculate validation sample size
        n_train = n - max(int(n * self.validate_ratio + 0.5), 1)
        if n_train == 0:
            raise Exception("too few input images")

        self.train_input = input_paths[0:n_train]
        self.train_target = target_paths[0:n_train]
        self.valid_input = input_paths[n_train:]
        self.valid_target = target_paths[n_train:]

    def _get_dataloader(self, input_paths, target_paths, distributed=True, shuffle=True):

        dataset = ColumnImagePairDataset(input_paths, target_paths, data_manager=self.data_manager)
        if self.model.module.seq_len != len(dataset[0][0]):
            raise Exception("model data_length does not match input image")

        if distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                num_replicas=self.world_size,
                rank=self.rank,
            )
        else:
            sampler = None
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return dataloader

    def _run_epoch(self, loader, epoch, is_train=True, verbose=False, print_step=100):

        if is_train:
            mode = "train"
            self.model.train()
        else:
            mode = "valid"
            self.model.eval()
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        n_batches = len(loader)
        base_step = epoch * n_batches
        epoch_loss = 0
        t0 = time.time()
        for step, (batch_in, batch_ta, _) in enumerate(loader):
            batch_in = torch.unsqueeze(batch_in.to(self.device), dim=1)
            batch_ta = torch.unsqueeze(batch_ta.to(self.device), dim=1)

            if is_train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                loss = self.criterion(self.model(batch_in), batch_ta)
                epoch_loss += loss.item()

            if is_train:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if self.rank == 0 and self.writer:
                self.writer.add_scalar(f"{mode}/loss_step", loss.item(), base_step + step)
                if is_train:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar(f"{mode}/lr", current_lr, base_step + step)
            if verbose and (step + 1) % print_step == 0:
                print('    epoch {} at rank {} "{}" progress {}/{}, {:.2f} min'.format(
                    epoch, self.rank, mode, step + 1, n_batches, (time.time() - t0) / 60))

        epoch_loss_tensor = torch.tensor([epoch_loss / n_batches], device=self.device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss_tensor.item() / self.world_size
        if self.rank == 0 and self.writer:
            self.writer.add_scalar(f"{mode}/loss_epoch", epoch_loss, epoch)

        return epoch_loss

    def save(self, path, best=True, full=False):

        # load state dict and convert to cpu
        if best:
            model_state_dict = self.best_model_dict
        else:
            model_state_dict = self.model.module.state_dict()
            for key, value in model_state_dict.items():
                model_state_dict[key] = value.cpu()

        if best or not full:
            output = {
                "model_params": self.model_params,
                "model": model_state_dict,
            }
        else:
            output = {
                "model_params": self.model_params,
                "model": model_state_dict,
                "optim": self.optimizer.state_dict(),
                "estop": self.early_stopping.state_dict(),
                "train_losses": self.train_losses,
                "valid_losses": self.valid_losses,
            }
        torch.save(output, path)

    def load(self, path):

        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        params = torch.load(path, map_location=self.device)

        self.model_params = params["model_params"]
        model_state_dict_module = dict()
        for k, v in params["model"].items():
            model_state_dict_module["module.{}".format(k)] = v
        self.model.load_state_dict(model_state_dict_module)

        if 'optim' in params:
            self.optimizer.load_state_dict(params["optim"])
        if 'estop' in params:
            if self.early_stopping is None:
                self.early_stopping = EarlyStopping()
            self.early_stopping.load_state_dict(params["estop"])
        if 'train_losses' in params:
            self.train_losses = params["train_losses"]
        if 'valid_losses' in params:
            self.valid_losses = params["valid_losses"]

    def plot_loss(self, output_path: str, matplotlib_device: str = "agg"):

        if self.train_losses is None or self.valid_losses is None or self.early_stopping is None:
            raise ValueError("run training first")

        matplotlib.use(matplotlib_device)
        x = np.arange(len(self.train_losses)) + 1
        plt.figure(figsize=(8, 6))
        (l1,) = plt.plot(x, self.train_losses, marker="o", color="green", zorder=3)
        (l2,) = plt.plot(x, self.valid_losses, marker="+", color="red", zorder=3)
        if 0 < self.early_stopping.best_epoch < len(self.train_losses):
            ylim = plt.gca().get_ylim()
            plt.plot([self.early_stopping.best_epoch + 1] * 2, ylim, c="orange", lw=2, ls="--", zorder=2)
            plt.ylim(ylim)
        plt.legend([l1, l2], ["train", "validate"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
