import os
import numpy as np

__all__ = ["check_dist_env", "setup_dist_env", "EarlyStopping"]


def check_dist_env():

    for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if k not in os.environ:
            return False
    if int(os.environ["WORLD_SIZE"]) < 1:
        return False
    if int(os.environ["RANK"]) < 0 or int(os.environ["LOCAL_RANK"]) < 0:
        return False

    return True


def setup_dist_env(verbose=False):

    if not check_dist_env():
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
    if int(os.environ["WORLD_SIZE"]) < 1:
        raise Exception("Invalid init_process_group parameter")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = max(int(os.environ.get("LOCAL_RANK", -1)), 0)
    address = os.environ.get("MASTER_ADDR")
    port = os.environ.get("MASTER_PORT")

    if verbose and rank == 0:
        print("Environment: WORLD_SIZE={}, ADDRESS={}:{}".format(world_size, address, port))

    return world_size, rank, local_rank, address, port


class EarlyStopping:

    def __init__(self, patience=7, delta=0.0, verbose=False):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.val_loss_min = np.Inf
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, val_loss):

        if val_loss > self.val_loss_min - self.delta:
            self.counter += 1
            if self.verbose:
                print("Epoch {}: stopping counter {} out of {}".format(epoch, self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print("Epoch {}: stopping here. The best epoch is {}".format(epoch, self.best_epoch))
        else:
            if self.verbose:
                print("Epoch {}: validate loss decreases {:.9f} --> {:.9f}".format(epoch, self.val_loss_min, val_loss))
            self.best_epoch = int(epoch)
            self.val_loss_min = float(val_loss)
            self.counter = 0

    def state_dict(self):

        return {'patience': self.patience,
                'delta': self.delta,
                'verbose': self.verbose,
                'counter': self.counter,
                'val_loss_min': self.val_loss_min,
                'best_epoch': self.best_epoch,
                'early_stop': self.early_stop}

    def load_state_dict(self, state_dict):

        self.patience = state_dict['patience']
        self.delta = state_dict['delta']
        self.verbose = state_dict['verbose']
        self.counter = state_dict['counter']
        self.val_loss_min = state_dict['val_loss_min']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']
