import argparse
import os
import torch
from pipeline.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
               
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='0: inference; 1: train;')
    parser.add_argument('--model', type=str, default='DeCTIAbla', help='DnCNN/DeCTIAbl')
    parser.add_argument('--test_epoch_chpt', type=int, default=-1, help='choose checkpoint id, -1 means best one')

    # data loader
    parser.add_argument('--data_path', type=str, default='/mnt/nas/mzh/project/dataset/HST_F814W', help='root path of the data file')
    parser.add_argument('--prediction_path', type=str, default='/mnt/nas/mzh/project/dataset/HST_F814W_prediction', help='root path of the data file')
    parser.add_argument('--redivide_files', type=int, default=0, help='randomly redivide files to train&valid&test')
    parser.add_argument('--seq_len_perchannel', type=int, default=2048, help='check formatting: input sequence length of single channel')
    parser.add_argument('--img_width_perchannel', type=int, default=4096, help='check formatting: input sequence length of single channel')
    
    #data limitation
    parser.add_argument('--half_plane', type=int, default=0, help='each .fits have 4k x 4k image composed of two 2k x 4k half plane, choose to use up half or complete one')
    parser.add_argument('--left_quarter', type=int, default=-1, help='1 use left quarter column part of data for computing acceleration, -1 use right quarter, 0 means all columns')
    parser.add_argument('--obs_year', type=int, nargs='+', default=[2012], help='use data from single year')
    # parser.add_argument('--obs_year', type=int, nargs='+', default=[2004, 2005, 2006, 2010, 2011, 2012, 2013, 2024], help='use data from single year')
    
    parser.add_argument('--log_path', type=str, default='/mnt/nas/mzh/project/log', help='location of model checkpoints')
    parser.add_argument('--log_sfolder', type=str, default='debug', help='location of model checkpoints')
    parser.add_argument('--loaded_chpt_sfolder', type=str, default='', help='location of model checkpoints')
    parser.add_argument('--config_subpath', type=str, default='config/remove_j92t', help='location of model checkpoints')
    
    # forecasting task
    parser.add_argument('--test_data_ratio', type=float, default=0.1, help='ratio of test data in complete dataset')
    parser.add_argument('--val_data_ratio', type=float, default=0.1, help='ratio of vali data in complete dataset')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--patience', type=int, default=300, help='early stopping patience')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--rank', type=int, default=0, help='if not ddp, default rank for single gpu')
    parser.add_argument('--world_size', type=int, default=1, help='if not ddp, default world_size for single gpu')
    parser.add_argument('--distributed', type=int, default=0, help='if distributed train with ddp or not')   

    #ablation study
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--abla_rpe', type=int, default=1, help='valid only when "model=DeCTIAbla"')  
    parser.add_argument('--abla_ape', type=int, default=1, help='valid only when "model=DeCTIAbla"')
    parser.add_argument('--abla_residual', type=int, default=1, help='valid only when "model=DeCTIAbla"')
    parser.add_argument('--abla_patch_size', type=int, default=1, help='')
    
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('raw Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training == 1:

        exp = Exp(args)  # set experiments
        args = exp.args
        print('adjusted Args in experiment:')
        print(args)
        
        train_gts, train_lqs, val_gts, val_lqs, test_gts, test_lqs \
        = exp.prepare_dataset(args.config_subpath, obs_year=args.obs_year)

        log_full_path = os.path.join(args.log_path, args.log_sfolder)

        exp.train(log_full_path, train_gts, train_lqs, val_gts, val_lqs)

        torch.cuda.empty_cache()
                
    elif args.is_training == 0:

        exp = Exp(args)  # set experiments
        args = exp.args
        print('adjusted Args in experiment:')
        print(args)
        
        train_gts, train_lqs, val_gts, val_lqs, test_gts, test_lqs \
        = exp.prepare_dataset(args.config_subpath, obs_year=args.obs_year)
        
        log_full_path = os.path.join(args.log_path, args.log_sfolder, '/inference_log')
        
        predict_full_path = os.path.join(args.prediction_path, args.log_sfolder)
        
        if not loaded_chpt_sfolder == "":
            model_path = os.path.join(args.log_path, loaded_chpt_sfolder)
        else:
            model_path = os.path.join(args.log_path, args.log_sfolder)
            
        exp.test(log_full_path, predict_full_path, model_path, test_gts, test_lqs, args.test_epoch_chpt)
        torch.cuda.empty_cache()