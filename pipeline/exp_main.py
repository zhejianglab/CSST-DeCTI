import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import matplotlib.pyplot as plt
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import gc
import fitsio
from fitsio import FITS,FITSHDR
from torch.utils.tensorboard import SummaryWriter

from pipeline.exp_basic import Exp_Basic
from models.DeCTIAbla import DeCTIAbla
from models.DnCNN import DnCNN
from data_provider.data_factory import data_provider_astro, retrieve_CTI, split_CTI, select_by_temporal, ClassifyByExpo, ClassifyByYear
from utils.tools import EarlyStopping, adjust_learning_rate, ClockTimer
from utils.tools import load, get_val_range, log_normalize_tsr, rlog_normalize_tsr

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if args.rank == 0:
            valmin, valmax = get_val_range()
            print("val: {0:.15f} noramalized to : {1:.15f}".format(1000.0, log_normalize_tsr(torch.tensor(1000).to(self.device), valmin, valmax)))
        
    def _build_model(self):
        model_dict = {
            'DnCNN': DnCNN,
            'DeCTIAbla': DeCTIAbla
        }

        if self.args.distributed:
            if self.args.model == 'DnCNN':
                input_channel = 2 if self.args.colid_condition else 1
                model = model_dict[self.args.model](in_nc=input_channel, out_nc=1, nc=96, nb=20, act_mode='bR').to(self.device)
            elif self.args.model == 'DeCTIAbla':
                ####baseline
                model = model_dict[self.args.model](seq_len=4096, patch_size=self.args.abla_patch_size, in_chans=1, window_size=self.args.window_size, \
                                    depths=[6, 6, 6, 6, 6, 6], embed_dim=96, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2., \
                                    ape=self.args.abla_ape, rpe=self.args.abla_rpe, residual=self.args.abla_residual).to(self.device)
                
                # #4by4
                # model = model_dict[self.args.model](seq_len=4096, patch_size=self.args.patch_size, in_chans=1, window_size=self.args.window_size, \
                #     depths=[4, 4, 4, 4], embed_dim=96, num_heads=[6, 6, 6, 6], mlp_ratio=2., \
                #     ape=self.args.abla_ape, rpe=self.args.abla_rpe, residual=self.args.abla_residual).to(self.device)
                
                # #2by2
                # model = model_dict[self.args.model](seq_len=4096, patch_size=self.args.patch_size, in_chans=1, window_size=self.args.window_size, \
                #     depths=[2, 2], embed_dim=96, num_heads=[6, 6], mlp_ratio=2., \
                #     ape=self.args.abla_ape, rpe=self.args.abla_rpe, residual=self.args.abla_residual).to(self.device)
            else:
                print("wrong model name in args!")
                return -1

            print('model architecture:')
            print(model)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[self.device], output_device=[self.device], find_unused_parameters=True)

        return model

    def _select_optimizer(self, otype=""):
        if otype == "Adamax":
            model_optim = torch.optim.Adamax([p for p in self.model.parameters() if p.requires_grad == True], lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, ltype='mse'):
        if ltype == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.SmoothL1Loss(beta=0.5)
        return criterion
            
    def vali(self, vali_data, vali_loader, criterion, sampler, nbatch_per_img, tensorb, epoch):
        epoch_loss = []
        self.model.eval()
        with torch.no_grad():
            i_step = 0
            for i_img, (img_x, img_y, dftime, _, _, _) in enumerate(vali_loader):
                
                img_x = np.squeeze(img_x, axis=0)
                img_y = np.squeeze(img_y, axis=0)
                for i_in_img in range(nbatch_per_img):
                    i_step += 1
                    
                    if not sampler == None:
                        sampler.set_epoch(i_step)
                    col_start = i_in_img * self.args.batch_size
                    col_end = col_start + self.args.batch_size
                    batch_x = img_x[:,col_start:col_end,:].float().to(self.device)
                    batch_y = img_y[:,col_start:col_end,:].float().to(self.device)

                    #limit value range
                    valmin, valmax = get_val_range()
                    batch_x = torch.clamp(batch_x,valmin,valmax)
                    batch_y = torch.clamp(batch_y,valmin,valmax)
                    batch_x = log_normalize_tsr(batch_x, min=valmin, max=valmax)
                    batch_y = log_normalize_tsr(batch_y, min=valmin, max=valmax)
                    
                    # encoder - decoder                   
                    outputs = self.model(batch_x)

                    loss = criterion(outputs, batch_y).detach().cpu()

                    if not tensorb == None:
                        print(f"val step: {str(i_step)}  val loss: {str(loss)}")
                        tensorb.add_scalar(f'val_loss step epoch:{epoch}', loss, i_step)
                    epoch_loss.append(loss)

        epoch_loss = np.average(epoch_loss)

        if self.args.distributed:
            epoch_loss_tensor = torch.tensor([epoch_loss]).to(self.device)
            dist.all_reduce(epoch_loss_tensor)
            epoch_loss = epoch_loss_tensor.item() / self.args.world_size

        self.model.train()
        return epoch_loss

    def prepare_dataset(self, config_path, obs_year=-1):
        # path = os.path.join(self.args.checkpoints, setting)
        
        if self.args.redivide_files and self.args.rank == 0:
            
            files_gt, files_lq = retrieve_CTI(self.args.data_path,\
                                    expected_shape=[self.args.seq_len_perchannel, self.args.img_width_perchannel])
            
            files_gt, files_lq = select_by_temporal(self.args.data_path, files_gt, files_lq, year_range=[obs_year, obs_year+1])
            
            train_files_gt, train_files_lq,\
            val_files_gt, val_files_lq,\
            test_files_gt, test_files_lq = split_CTI(files_gt, files_lq, self.args.test_data_ratio, self.args.val_data_ratio, shuffle=True)
        
            files_struct = {'gt': train_files_gt,
                            'lq': train_files_lq}
            files_pd = pd.DataFrame(data=files_struct)
            files_pd.to_csv(os.path.join(config_path, 'train.csv'), index=False)
            
            files_struct = {'gt': val_files_gt,
                            'lq': val_files_lq}
            files_pd = pd.DataFrame(data=files_struct)
            files_pd.to_csv(os.path.join(config_path, 'val.csv'), index=False)
            
            files_struct = {'gt': test_files_gt,
                            'lq': test_files_lq}
            files_pd = pd.DataFrame(data=files_struct)
            files_pd.to_csv(os.path.join(config_path, 'test.csv'), index=False)
        else:

            def filter_by_year(gts, lqs, dates, year):
                gts_new, lqs_new, dates_new = [], [], []
                for gt, lq, date in zip(gts, lqs, dates):
                    if pd.to_datetime(date).year == year:
                        gts_new.append(gt)
                        lqs_new.append(lq)
                        dates_new.append(date)
                return gts_new, lqs_new
                        
            train_pd = pd.read_csv(os.path.join(config_path, 'train.csv'))
            val_pd = pd.read_csv(os.path.join(config_path, 'val.csv'))
            test_pd = pd.read_csv(os.path.join(config_path, 'test.csv'))
                        
            train_files_gt = train_pd['gt']
            train_files_lq = train_pd['lq']         
            
            val_files_gt = val_pd['gt']
            val_files_lq = val_pd['lq']
            
            test_files_gt = test_pd['gt']
            test_files_lq = test_pd['lq']
        
            single_year = None
            if not isinstance(obs_year, list):
                single_year = obs_year
            if isinstance(obs_year, list) and len(obs_year)==1:
                single_year = obs_year[0]
            if single_year is not None:
                if single_year > 0 \
                    and "date" in train_pd \
                    and "date" in val_pd \
                    and "date" in test_pd:

                    train_date = train_pd['date']  
                    val_date = val_pd['date']
                    test_date = test_pd['date']
                    
                    train_files_gt, train_files_lq = filter_by_year(train_files_gt, train_files_lq, train_date, year=single_year)
                    val_files_gt, val_files_lq = filter_by_year(val_files_gt, val_files_lq, val_date, year=single_year)
                    test_files_gt, test_files_lq = filter_by_year(test_files_gt, test_files_lq, test_date, year=single_year)              
                 
        return train_files_gt, train_files_lq, val_files_gt, val_files_lq, test_files_gt, test_files_lq

    def cal_nbatch_per_img(self, img_path, batchsize, left_quarter=0):
        img, _ = load(img_path)
        img_width = img.shape[1]
        if left_quarter == 0:
            return img_width // batchsize 
        else:
            return img_width // 4 // batchsize

    def distributed_concat(self, tensor):
        output_tensors = [torch.empty_like(tensor) for _ in range(self.args.world_size)]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat

    def train(self, log_path, train_gt, train_lq, val_gt, val_lq):
        if self.args.rank == 0:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
        if self.args.distributed:
            dist.barrier()

        train_data, train_loader, train_sampler = data_provider_astro(self.args, files_gt=train_gt, files_lq=train_lq, flag="train", nworkers=self.args.num_workers)
        vali_data, vali_loader, vali_sampler = data_provider_astro(self.args, files_gt=val_gt, files_lq=val_lq, flag="val", nworkers=self.args.num_workers)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(ltype='mse')

        nbatch_per_img = self.cal_nbatch_per_img(os.path.join(self.args.data_path, train_gt[0]), self.args.batch_size, self.args.left_quarter)
        train_steps = len(train_loader) * nbatch_per_img
        total_steps = int(train_steps * self.args.train_epochs)

        print(f"nbatch_per_img: {str(nbatch_per_img)}   train_steps_per_epoch : {str(train_steps)}  length of trainloader: {str(len(train_loader))}")
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            total_steps = total_steps,
                                            pct_start = self.args.pct_start,
                                            max_lr = self.args.learning_rate)
        
        if self.args.rank == 0:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, distributed=self.args.distributed)
            writer = SummaryWriter(log_dir=log_path, filename_suffix='')
            print("training start!")
        else:
            writer = None

        i_step = 0
        lr = self.args.learning_rate
        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            if not train_sampler == None:
                train_sampler.set_epoch(epoch)

            for i_img, (img_x, img_y, dftime, _, gt_path, _) in enumerate(train_loader):

                img_x = np.squeeze(img_x, axis=0)
                img_y = np.squeeze(img_y, axis=0)           
                for i_in_img in range(nbatch_per_img):

                    i_step += 1
                    
                    col_start = i_in_img * self.args.batch_size
                    col_end = col_start + self.args.batch_size
                    model_optim.zero_grad()
                    batch_x = img_x[:,col_start:col_end,:].float().to(self.device)
                    batch_y = img_y[:,col_start:col_end,:].float().to(self.device)

                    #limit value range
                    valmin, valmax = get_val_range()
                    
                    batch_x = torch.clamp(batch_x,valmin,valmax)
                    batch_y = torch.clamp(batch_y,valmin,valmax)
                    batch_x = log_normalize_tsr(batch_x, min=valmin, max=valmax)
                    batch_y = log_normalize_tsr(batch_y, min=valmin, max=valmax)
                    
                    # encoder - decoder
                    if self.args.rank == 0:
                        print("before model")                   
                    outputs = self.model(batch_x)
                    if self.args.rank == 0:
                        print("after model") 
                    
                    ## input output use same here
                    loss = criterion(outputs, batch_y)
                    input_dif = criterion(batch_x, batch_y)
                    train_loss.append(loss.item())


                    if (i_step-1) % 1 == 0:
                        if self.args.distributed:
                            step_loss_tensor = torch.tensor([loss.item()]).to(self.device)
                            dist.all_reduce(step_loss_tensor)
                            mean_loss = step_loss_tensor.item() / self.args.world_size
                        else:
                            mean_loss = loss.item()
                        if self.args.rank == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}  input_dif: {3:.7f} lr: {3:.7f}".format(i_step, epoch + 1, loss.item(), input_dif.item(), lr))
                            if mean_loss > 5.0:
                                print(f'bad loss img_path: {gt_path}  patch_id: {i_in_img}  input_dif:{str(input_dif.item())} lr: {str(lr)}')

                            writer.add_scalar('train_loss_step', mean_loss, i_step)
                            sys.stdout.flush()

                    loss.backward()
                    model_optim.step()
                    
                    lr = adjust_learning_rate(model_optim, scheduler)
                    scheduler.step()
            
            if self.args.distributed:
                    dist.barrier()
         
            #loss calc in one epoch
            epoch_loss_mean = np.average(train_loss)
            if self.args.distributed:
                epoch_loss_tensor = torch.tensor([epoch_loss_mean]).to(self.device)
                dist.all_reduce(epoch_loss_tensor)
                train_loss = epoch_loss_tensor.item() / self.args.world_size

            #loss of vali
            val_loss = self.vali(vali_data, vali_loader, criterion, sampler=vali_sampler, nbatch_per_img=nbatch_per_img, tensorb=writer, epoch=int(epoch + 1))

            #ealystop check& loss conclusion
            stop_train = torch.tensor([0]).to(self.device)

            if self.args.rank == 0:
                writer.add_scalar('train_loss_epoch', train_loss, epoch+1)
                writer.add_scalar('val_loss_epoch', val_loss, epoch+1)          
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps / self.args.world_size, train_loss, val_loss))
                early_stopping(val_loss, self.model, log_path, epoch+1)
                stop_train = torch.tensor([int(early_stopping.early_stop)]).to(self.device)
                
            if self.args.distributed:
                dist.broadcast(stop_train, src=0)
            
            if stop_train.item():
                print("Early stopping")
                break

        best_model_path = log_path + '/' + 'checkpoint.pth'
        if self.args.distributed:
            self.model.module.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, log_path, predict_path, model_path, test_gt, test_lq, checkpts=-1):
        SAVE_NUM = 0
        CALC_METRICS = 0

        if self.args.rank == 0:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            if not os.path.exists(predict_path):
                os.makedirs(predict_path)
        if self.args.distributed:
            dist.barrier()

        test_data, test_loader, test_sampler = data_provider_astro(self.args, files_gt=test_gt, files_lq=test_lq, flag="test")

        print('loading model')
        if checkpts < 0:
            state_dict = torch.load(os.path.join(model_path, 'checkpoint.pth'))
        else:
            state_dict = torch.load(os.path.join(model_path, f'checkpoint_epoch_{str(checkpts)}.pth'))

        self.model.module.load_state_dict(state_dict)
        print(self.model.state_dict().keys())
        print(state_dict.keys())

        nbatch_per_img = self.cal_nbatch_per_img(os.path.join(self.args.data_path, test_gt[0]), batchsize=self.args.batch_size, left_quarter=self.args.left_quarter)
        criterion = self._select_criterion(ltype='mse')

        self.model.eval()
        with torch.no_grad():
            
            groups_mean = None
            groups_var = None
            all_timer = ClockTimer()
            io_timer = ClockTimer()
            alg_timer = ClockTimer()
            
            io_timer.start()
            all_timer.start()

            for i_img, (img_x, img_y, dftime, _, gt_path, lq_path) in enumerate(test_loader):

                io_timer.pause()
                io_timer.num_adder()
                
                img_x = np.squeeze(img_x, axis=0)
                img_y = np.squeeze(img_y, axis=0)
                
                img_name = os.path.basename(str(lq_path)).split('.')[0]
                img_rela_error = None      
                for i_in_img in range(nbatch_per_img):

                    alg_timer.start()
                    
                    col_start = i_in_img * self.args.batch_size
                    col_end = col_start + self.args.batch_size
                    
                    batch_x = img_x[:,col_start:col_end,:].float().to(self.device)
                    batch_y = img_y[:,col_start:col_end,:].float().to(self.device)
                    
                    #normalization
                    valmin, valmax = get_val_range()
                    
                    batch_x = torch.clamp(batch_x,valmin,valmax)
                    batch_y = torch.clamp(batch_y,valmin,valmax)
                    
                    batch_x = log_normalize_tsr(batch_x, min=valmin, max=valmax)
                    batch_y = log_normalize_tsr(batch_y, min=valmin, max=valmax)
            
                    outputs = self.model(batch_x, 0)

                    #reverse normalization after inference
                    raw_x = rlog_normalize_tsr(batch_x, valmin, valmax)
                    raw_y = rlog_normalize_tsr(batch_y, valmin, valmax)
                    raw_predict = rlog_normalize_tsr(outputs, valmin, valmax)
                    
                    loss = criterion(outputs, batch_y)
                    input_diff = criterion(batch_x, batch_y)

                    
                    if i_in_img == 0:
                        complete_img = torch.squeeze(raw_predict, axis=-1)
                    else:
                        complete_img = torch.cat([complete_img, torch.squeeze(raw_predict, axis=-1)], axis=1)

                    alg_timer.pause()
                    
                    #statistics for relative error
                    if CALC_METRICS:

                        den = raw_x - raw_y
                        nu = raw_predict - raw_y
                        den = torch.squeeze(torch.norm(den, p=1, dim=0).view(1,-1), 0)
                        nu = torch.squeeze(torch.norm(nu, p=1, dim=0).view(1,-1), 0)
                        if i_in_img == 0:
                            cur_img_rerror = torch.div(nu, den)
                            cur_img_indiff = den
                            cur_img_prerr = nu
                        else: 
                            cur_img_rerror = torch.concat((cur_img_rerror, torch.div(nu, den)), dim=0)
                            cur_img_indiff = torch.concat((cur_img_prerr, den), dim=0)
                            cur_img_prerr = torch.concat((cur_img_indiff, nu), dim=0)
                            
                        if (i_img == 0) & (i_in_img == 0):
                            relative_error = torch.div(nu, den)
                            abs_error_raw = torch.clone(den)
                            abs_error_pr = torch.clone(nu)
                        else:
                            relative_error = torch.concat((relative_error, torch.div(nu, den)), dim=0)
                            abs_error_raw = torch.concat((abs_error_raw, den), dim=0)
                            abs_error_pr = torch.concat((abs_error_pr, nu), dim=0)

                        if img_rela_error is None:
                            img_rela_error = torch.div(nu, den)
                        else:
                            img_rela_error = torch.concat((img_rela_error, torch.div(nu, den)), dim=0)
                        
                        print(f'{str(gt_path)} relative_error: {str(torch.mean(relative_error).item()*100)}% normalized prediction loss:{str(loss.item())}  input diff:{str(input_diff)}')

                        if groups_mean is None:
                            groups_mean = torch.mean(img_rela_error).unsqueeze(0)
                        else:
                            groups_mean = torch.concat((torch.mean(img_rela_error).unsqueeze(0), groups_mean), dim=0)
                        if groups_var is None:
                            groups_var = torch.var(img_rela_error).unsqueeze(0)
                        else:
                            groups_var = torch.concat((torch.var(img_rela_error).unsqueeze(0), groups_var), dim=0)

                alg_timer.num_adder()
                io_timer.start()

                if i_img < SAVE_NUM:
                    if 0:
                        np.save(os.path.join(predict_path, f"{str(img_name)} + _pr.npy"), complete_img.cpu().numpy())
                        print(f"image {img_name}.npy saved")
                    else:
                        assert(self.args.img_width == complete_img.shape[1])
                        height = complete_img.shape[0]
                        top_half = torch.zeros(height//2, self.args.img_width)
                        bottom_half = torch.zeros(height//2, self.args.img_width)
                        quarter_len = self.args.img_width // 4
                        if self.args.left_quarter == 0:
                            top_half = complete_img[0:height//2,:]
                            bottom_half = complete_img[height//2:,:]
                        elif self.args.left_quarter == 1:
                            top_half[:, 0:quarter_len] = complete_img[0:height//2,:]
                            bottom_half[:, 0:quarter_len] = complete_img[height//2:,:]
                            top_half[:, quarter_len:] = 0
                            bottom_half[:, quarter_len:] = 0
                        elif self.args.left_quarter == -1:    
                            top_half[:, -quarter_len:] = complete_img[0:height//2,:]
                            bottom_half[:, -quarter_len:] = complete_img[height//2:,:]
                            top_half[:, :-quarter_len] = 0
                            bottom_half[:, :-quarter_len] = 0
                        else:
                            print("Error: wrong left_quater value")
                            return
                        
                        file_out = os.path.join(predict_path, f"{str(img_name)}_pr.fits")
                        print(f"saving prediciton image: {file_out}")
                        os.system(f"cp {lq_path[0]} {file_out}")

                        with fitsio.FITS(file_out, 'rw') as fits_out:
                            fits_out[1].write(top_half.cpu().numpy())
                            fits_out[4].write(bottom_half.cpu().numpy())

                
                print(f'rank {str(self.args.rank)} \
                        avr_time_alg: {str(alg_timer.time_avr())}  \
                        avr_time_io: {str(io_timer.time_avr())} \
                        number: {str(alg_timer.num)}')

            if CALC_METRICS:

                relative_error_gather = self.distributed_concat(relative_error)
                abs_error_raw_gather = self.distributed_concat(abs_error_raw)
                abs_error_pr_gather = self.distributed_concat(abs_error_pr)
                
                groups_mean_gather = self.distributed_concat(groups_mean)
                groups_var_gather = self.distributed_concat(groups_var)

                if self.args.rank == 0:
                    # change1 save hstogram && npy file
                    #
                    inner_var = torch.mean(groups_var_gather).item()
                    
                    global_mean = torch.mean(groups_mean_gather).item()
                    inter_var = torch.mean((groups_mean_gather - global_mean)**2).item() 
                    
                    #calculate peak value
                    rela_statis = relative_error.cpu().numpy()
                    bin_edges = np.arange(min(rela_statis), min(50, max(rela_statis)) + 0.01, 0.01)
                    cnts, edges = np.histogram(rela_statis, bins=bin_edges)
                    max_bin_idx = np.argmax(cnts)
                    mode= (edges[max_bin_idx] + edges[max_bin_idx+1]) / 2
                    
                    plt.hist(rela_statis, log=True, color='blue')
                    plt.savefig(os.path.join(log_path, 'vector_L1_relative_error'))
                    plt.clf() 

                    plt.hist(rela_statis, range=(0, 2), bins=1000, log=True, color='blue')
                    plt.axvline(mode, color='r', linestyle='--', alpha=0.5)

                    plt.title(f"Network: {self.args.model} Year: {self.args.obs_year[0]}")
                    # use relative coordinates (axes coordinates) avoid overlap
                    x_pos = 0.95  # right 95% loc (range 0-1)
                    y_start = 0.95  # start from 95% to the top (range 0-1)
                    y_offset = 0.05  # fix distance between lines (5% of height of image)

                    # get current axis
                    ax = plt.gca()

                    ax.text(x_pos, y_start,                f"mean     :   {global_mean:.2f}", 
                            ha='right', va='top', fontsize=10, transform=ax.transAxes)
                    ax.text(x_pos, y_start - y_offset,     f"peak loc :   {mode:.2f}", 
                            ha='right', va='top', fontsize=10, transform=ax.transAxes)
                    ax.text(x_pos, y_start - 2*y_offset, f"inner var: {inner_var:.4f}", 
                            ha='right', va='top', fontsize=10, transform=ax.transAxes)
                    ax.text(x_pos, y_start - 3*y_offset, f"inter_var: {inter_var:.4f}", 
                            ha='right', va='top', fontsize=10, transform=ax.transAxes)
                    plt.savefig(os.path.join(log_path, 'vector_L1_relative_error_within_2'))
                    plt.clf()
                    print(f'global_mean :{global_mean}  peak_loc :{mode} inner_var :{inner_var}  inter_var :{inter_var}')
                    
                    plt.hist(abs_error_raw.cpu().numpy(), log=True, color='blue', label='L1(raw, gtruth)')
                    plt.hist(abs_error_pr.cpu().numpy(), log=True, color='red', label='L1(predict, gtruth)')
                    plt.savefig(os.path.join(log_path, 'vector_L1_abs_error'))
                    plt.legend()
                    plt.clf()

            all_timer.pause()
            all_timer.num_adder()
            print(f'rank {str(self.args.rank)} \
                        avr_time_all: {str(all_timer.time_avr())}')
            return