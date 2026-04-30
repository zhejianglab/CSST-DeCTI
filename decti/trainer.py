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
from .dataset.dataset import PairedImageFileDataset
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
        file_shuffle_seed: int = 2026,
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
        print_step: int = 100,
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
        train_loader = self._get_dataloader(self.train_input, self.train_target, distributed=True)
        train_nbatch = self._count_nbatch(train_loader)
        if len(self.valid_input) // self.world_size > 2:
            valid_loader = self._get_dataloader(self.valid_input, self.valid_target, distributed=True)
        else:
            valid_loader = self._get_dataloader(self.valid_input, self.valid_target, distributed=False)
        if verbose:
            print('Rank {}: assigned files: {}/{} (train/validate)'.format(self.rank, len(train_loader), len(valid_loader)))

        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            epochs=n_epochs,
            steps_per_epoch=train_nbatch,
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
            train_loss = self._run_epoch(
                epoch, train_loader, is_train=True, verbose=verbose, nstep=train_nbatch, print_step=print_step)
            self.train_losses.append(train_loss)
            dist.barrier()
            valid_loss = self._run_epoch(epoch, valid_loader, is_train=False)
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
                    fcheck_current = os.path.join(logdir, "checkpoint_epoch_{:03d}.pth".format(epoch))
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

        dataset = PairedImageFileDataset(input_paths, target_paths, self.data_manager, inference_mode=False)
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
            batch_size=1,  # one image per batch
            num_workers=self.num_workers,
        )

        return dataloader

    def _count_nbatch(self, dataloader):

        nbatch = 0
        for idx, batch in enumerate(dataloader):
            nbatch += batch[0].shape[2] // self.batch_size

        return nbatch

    def _run_epoch(self, epoch, loader, is_train=True, verbose=False, nstep=None, print_step=100):

        if is_train:
            mode = "train"
            self.model.train()
        else:
            mode = "valid"
            self.model.eval()
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_ncol = 0
        current_step = 0
        current_image_step = 0
        t0 = time.time()
        for idx, (image_in, image_ta, _, path_in, path_ta) in enumerate(loader):
            img_in = torch.as_tensor(image_in).permute(2, 0, 1)  # (ncol, 1, seq_len)
            img_ta = torch.as_tensor(image_ta).permute(2, 0, 1)
            nbatch_per_img = img_in.shape[0] // self.batch_size

            losses = np.zeros(nbatch_per_img)
            for i_in_img in range(nbatch_per_img):
                col_start = i_in_img * self.batch_size
                col_end = min((i_in_img + 1) * self.batch_size, img_in.shape[0])
                batch_in = img_in[col_start:col_end, :, :].to(self.device)
                batch_ta = img_ta[col_start:col_end, :, :].to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    loss = self.criterion(self.model(batch_in), batch_ta)
                    losses[i_in_img] = loss.item()

                if is_train:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                current_step += 1
                if verbose and current_step % print_step == 0:
                    print('    epoch {} at rank {} progress {}/{}, {:.2f} min'.format(
                        epoch, self.rank, current_step, nstep, (time.time() - t0) / 60))

            image_loss = np.sum(losses)
            total_loss += image_loss
            total_ncol += nbatch_per_img * self.batch_size
            if self.rank == 0 and self.writer:
                self.writer.add_text(f"{mode}/{epoch}: input_image", path_in[0], current_image_step)
                self.writer.add_text(f"{mode}/{epoch}: target_image", path_ta[0], current_image_step)
                self.writer.add_scalar(f"{mode}/{epoch}: image_loss", image_loss, current_image_step)
                if is_train:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar(f"{mode}/{idx}: lr", current_lr, current_image_step)
            current_image_step += 1

        total_loss = torch.tensor(total_loss, device=self.device)
        total_ncol = torch.tensor(total_ncol, device=self.device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ncol, op=dist.ReduceOp.SUM)
        epoch_loss = (total_loss / total_ncol).item()  # calculate loss per column
        if self.rank == 0 and self.writer:
            self.writer.add_scalar(f"{mode}/{epoch}: avg_loss", epoch_loss)

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
