import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import fitsio
import copy
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from matplotlib.cm import ScalarMappable
import os

plt.switch_backend('agg')

class ClockTimer:
    def __init__(self):
        self.num = 0
        self.tick_tmp = 0.0
        self.time_passed = 0.0

        self.ticking = False

    def start(self):
        self.tick_tmp = time.perf_counter()

        self.ticking = True

    def pause(self):
        if self.ticking:
            tick_now = time.perf_counter()
            self.time_passed += (tick_now - self.tick_tmp)
            self.tick_tmp = tick_now
            
            self.ticking = False

    def num_adder(self):
        self.num += 1

    def time_avr(self):
        return self.time_passed / self.num
    
    def clear(self):
        self.__init__()

def scale_by_expo(input, real_expo, target_expo=600):
    if real_expo > 0:
        return input * (target_expo / real_expo)
    else: 
        return input

def unscale_by_expo(input, real_expo, target_expo=600):
    if real_expo > 0:
        return input * (real_expo / target_expo)
    else:
        return input

def log_normalize_tsr(input, min, max, add=1e2):
    a1 = 1
    # input = input.double()
    rslt = 2*(torch.log(input * a1 + torch.mul(torch.ones_like(input),add - min)) - torch.log(torch.mul(torch.ones_like(input), min+add - min))) / \
        (torch.log(torch.tensor(a1 * max+add - min)) - torch.log(torch.tensor(min+add - min))) - 1
    return rslt

def rlog_normalize_tsr(input, min, max, add=1e2):
    a1 = 1
    # a = 1000.0
    # loga =  np.log(a)
    # def log_unit(data, add_data, min_data, a=1):
    #     return torch.log(torch.tensor(a * data + add_data - min_data))
    
    # exp_ind = torch.tensor(torch.mul((input+1)/2, log_unit(max, add, min, a1) - log_unit(min, add, min, 1)) + torch.ones_like(input) * log_unit(min, add, min, 1)).double()
    
    # return (torch.exp(exp_ind) - torch.ones_like(input)*(min-add)) / a1
    # input = input.double()
    rslt = (torch.exp( (input + 1) / 2 * (torch.log(torch.tensor(a1 * max+add - min)) - torch.log(torch.tensor(min+add - min))) + torch.log(torch.mul(torch.ones_like(input), min+add - min)) ) \
        - torch.mul(torch.ones_like(input),add - min) ) / a1
    return rslt

def get_val_range():
    return  -100, 60000.0

def init_seeds(RANDOM_SEED, no):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_date(path):
    fits = fitsio.FITS(path)
    head = fits[0].read_header()
    time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
    dftime = pd.to_datetime(time_str)
    
    return dftime

def load_expo(path):
    fits = fitsio.FITS(path)
    head = fits[0].read_header()
    
    return head['EXPTIME'] 

def load_dq(path, half_plane=False, left_quarter=0):
    fits = fitsio.FITS(path)
    
    mask = fits[4].read()

    if not half_plane:
        mask = np.concatenate((mask, fits[6].read()), axis=0)
        
    if left_quarter == 1:
        mask = mask[:, 0:1024]
    elif left_quarter == -1:
        mask = mask[:, -1024:]
    
    return mask 

def load(file):
    if file.endswith('.npy'):
        return np.load(file), []
    elif file.endswith('.fits'):
        with fits.open(file) as hdul:
            if 'SCI' in hdul:
                data = hdul['SCI'].data
                hdul.close()
            else:
                data = hdul['IMAGE'].data
            return data, hdul
    else:
        print('Error: wrong image format!!!')
        return [],[]

def load_test(file):
    with fits.open(file) as hdul:
        data = hdul['SCI'].data
        data = None
    hdul.close()
    
def save(data, hdul, file):
    if file.endswith('.fits'):
        hdul_new = fits.HDUList(hdul)
        hdul_new=hdul
        if 'SCI' in hdul: 
            hdul_new['SCI'].data = data
        else:
            hdul_new['IMAGE'].data = data
        hdul_new.writeto(file, overwrite=True)
    else:
        np.save(file, data)

def adjust_learning_rate(optimizer, scheduler):
    lr = scheduler.get_last_lr()[0]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, distributed=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.distributed = distributed
        self.save_epoch_step = 1

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        if epoch % self.save_epoch_step == 0:
            self.save_current_checkpoint(model, path, epoch)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.distributed:
            torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def save_current_checkpoint(self, model, path, epoch):
        if self.distributed:
            torch.save(model.module.state_dict(), f"{path}/checkpoint_epoch_{epoch}.pth")
        else:
            torch.save(model.state_dict(), f"{path}/checkpoint_epoch_{epoch}.pth")
