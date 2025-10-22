from data_provider.data_loader import Dataset_HST
from torch.utils.data import DataLoader
import gzip
import os
import shutil
import numpy as np
import gc
import sys
import torch
import matplotlib.pyplot as plt
import fitsio
import pandas as pd
import bisect

from utils.tools import load, load_expo, load_date

def un_gz(infile, outfile=''):
    if outfile == '':
        outfile = infile.replace(".gz", "")
    with gzip.open(infile, 'rb') as fin:
        with open(outfile, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    del fin
    del fout
    gc.collect()
    return outfile

def select_by_temporal(root_path, files_gt, files_lq, year_range=[2005,2006]):
    
    years = []
    new_gts, new_lqs = [],[]
    for id in range(len(files_gt)):
        fits_gt = fitsio.FITS(os.path.join(root_path, files_gt[id]))

        head = fits_gt[0].read_header()

        time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
        dftime = pd.to_datetime(time_str)

        years.append(dftime.year)
        if int(dftime.year) >= year_range[0] and (dftime.year) < year_range[1]:
            new_gts.append(files_gt[id])
            new_lqs.append(files_lq[id])

    plt.hist(years, bins=30, log=True)
    plt.savefig('./RAW_HIST.png')
    plt.clf()
    
    return new_gts, new_lqs

def classify_by_year(root_path, files_gt, files_lq, year_range=[2005,2006]):
    
    years = []
    new_gts, new_lqs = [],[]
    for id in range(len(files_gt)):
        fits_gt = fitsio.FITS(os.path.join(root_path, files_gt[id]))

        head = fits_gt[0].read_header()

        time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
        dftime = pd.to_datetime(time_str)

        years.append(dftime.year)
        if int(dftime.year) >= year_range[0] and (dftime.year) < year_range[1]:
            new_gts.append(files_gt[id])
            new_lqs.append(files_lq[id])

    plt.hist(years, bins=30, log=True)
    plt.savefig('./RAW_HIST.png')
    plt.clf()
    
    return new_gts, new_lqs

def select_by_temporal(root_path, files_gt, files_lq, year_range=[2005,2006]):
    
    years = []
    new_gts, new_lqs = [],[]
    for id in range(len(files_gt)):
        fits_gt = fitsio.FITS(os.path.join(root_path, files_gt[id]))

        head = fits_gt[0].read_header()

        time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
        dftime = pd.to_datetime(time_str)

        years.append(dftime.year)
        if int(dftime.year) >= year_range[0] and (dftime.year) < year_range[1]:
            new_gts.append(files_gt[id])
            new_lqs.append(files_lq[id])

    plt.hist(years, bins=30, log=True)
    plt.savefig('./RAW_HIST.png')
    plt.clf()
    
    return new_gts, new_lqs

def resort_by_temporal(datetimes=[], root_path=[], files_gt=[], files_lq=[], valid_ids=[]):
    
    # root_path, files_gt, files_lq  no empty  or   
    # datetimes no empty
    if not valid_ids:
        valid_ids = np.arange(max(len(datetimes), len(files_gt)))

    if not datetimes: 
        assert len(files_gt) == len(files_lq)

        for id, raw_id in enumerate(valid_ids):
            fits_gt = fitsio.FITS(os.path.join(root_path, files_gt[raw_id]))

            head = fits_gt[0].read_header()
            time_str = str(head['DATE-OBS']) + ' ' + str(head['TIME-OBS'])
            
            datetimes.append(pd.to_datetime(time_str))
        

    sorted_indices = sorted(valid_ids, key=lambda i: datetimes[i])
    
    return sorted_indices, datetimes
     

def split_CTI(gt_paths, lq_paths, test_ratio, valid_ratio, shuffle=True):
    
    numall = len(gt_paths)
    index = list(range(numall))
    
    if shuffle:
        np.random.shuffle(index)
    
    nval = np.maximum(1, int(numall*valid_ratio + 0.5))
    ntest = np.maximum(1, int(numall*test_ratio + 0.5))
    ntrain = numall - nval - ntest

    return gt_paths[:ntrain], lq_paths[:ntrain], gt_paths[ntrain:ntrain+nval], lq_paths[ntrain:ntrain+nval], gt_paths[-ntest:], lq_paths[-ntest:]

# retrieve_split(args.root_path, args.test_data_ratio, args.val_data_ratio)
def retrieve_CTI(root_path, expected_shape):

    def check_unzip_file(path):
        dir_list = os.listdir(path)
        gt_filepath, lq_filepath = "",""

        dir_list = sorted(dir_list,key = lambda i:len(i),reverse=True) 
        for fname in dir_list:
            raw_filepath = os.path.join(path, fname)
            sys.stdout.flush()
            try:

                if "_flc.fits" in raw_filepath:

                    if raw_filepath.endswith('.gz'):
                        print('do unzip')
                        gt_filepath = un_gz(raw_filepath)
                        gc.collect()
                        os.system(f'rm {str(raw_filepath)}')
                    else:
                        gt_filepath = raw_filepath
                        data, hdul = load(gt_filepath)
                        if data.shape[0] != expected_shape[0] or \
                            data.shape[1] != expected_shape[1]:
                            gt_filepath = ""
                            continue
                        hdul.close()
                        print(gt_filepath)
                        
                        
                elif "_flt.fits" in raw_filepath:
                    
                    if raw_filepath.endswith('.gz'):
                        lq_filepath = un_gz(raw_filepath)
                        gc.collect()
                        print('do unzip')
                        os.system(f'rm {str(raw_filepath)}')
                    else:
                        lq_filepath = raw_filepath
                        data, hdul = load(lq_filepath)
                        hdul.close()
                        if data.shape[0] != expected_shape[0] or \
                            data.shape[1] != expected_shape[1]:
                            lq_filepath = ""
                            continue
                        print(lq_filepath)
                elif raw_filepath.endswith('.gz'):
                    os.system(f'rm {str(raw_filepath)}') 
                else: continue 

            except Exception as e:
                gt_filepath=""
                lq_filepath=""
                print(f'{fname} extraction failed')
                break
        return (str(gt_filepath) != "") and (str(lq_filepath) != ""), gt_filepath, lq_filepath

    gt_paths, lq_paths = [],[]
    folders = os.listdir(root_path)

    # for folder in folders:
    for folder in folders:
        folder_path = os.path.join(root_path, folder)        
        
        if not os.path.isdir(folder_path):
            continue
     
        good_data, gt_file, lq_file = check_unzip_file(folder_path)

        if not good_data:
            continue

        #todo:test
        gt_file = os.path.relpath(gt_file, root_path)
        lq_file = os.path.relpath(lq_file, root_path)
        
        gt_paths.append(gt_file)
        lq_paths.append(lq_file)

    return gt_paths, lq_paths
    
def data_provider_astro(args, files_gt, files_lq, flag, nworkers=0):
    files_gt = [os.path.join(args.data_path, gt) for gt in files_gt]
    files_lq = [os.path.join(args.data_path, lq) for lq in files_lq]
    
    data_set = Dataset_HST(
        files_gt=files_gt,
        files_lq=files_lq, 
        half_plane=args.half_plane,
        left_quarter=args.left_quarter)
        
    if flag == 'train':
        shuffle_flag = True
    else:
        shuffle_flag = False

    if args.distributed:
        print("distributed data_provider_astro")
        sampler = torch.utils.data.distributed.DistributedSampler(data_set, \
            shuffle=shuffle_flag, num_replicas=args.world_size, rank=args.rank)
   
        data_loader = DataLoader(
        data_set,
        batch_size=1, 
        sampler=sampler,
        num_workers=nworkers)
    else:
        data_loader = DataLoader(
        data_set,
        batch_size=1, 
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=nworkers)
            
        sampler = None
    print(f"len of files_gt: {str(len(files_gt))} rank: {str(args.rank)}")
    print(f"len of data_loader: {str(len(data_loader))} rank: {str(args.rank)}")
    print(f"world size: {str(args.world_size)}")
    return data_set, data_loader, sampler

def ClassifyByYear(root_path, files, year_list=np.arange(2004, 2025, 1), expo_range=[500, 1000], num_max=-1):

    files_classified = [[] for _ in range(len(year_list))]

    for i, file in enumerate(files):
        #filer out abnormal exposure
        if expo_range is not None:
            expo = load_expo(os.path.join(root_path, file))
            if expo < expo_range[0] or expo > expo_range[1]:
                continue
        
        year = load_date(os.path.join(root_path, file)).year
        try:
            if year not in year_list:
                continue

            index = year_list.index(year)
            if num_max > 0 and len(files_classified[index]) <= num_max:
                files_classified[index].append(file)
        except StopIteration:
            continue

    return files_classified
    
def ClassifyByExpo(root_path, files, level=[0, 500, 1000, 5000]): #train_class [nlabel, nimgs]

    files_classified = [[] for _ in range(len(level)-1)]
    exist_flag = [0 for _ in range(len(level))]

    for i, file in enumerate(files):
        
        evalue = load_expo(os.path.join(root_path, file))
        
        if evalue == 0 or evalue >= level[-1]:
            print(f"expo: {str(evalue)} dropped in ClassifyByExpo")
            continue

        index = bisect.bisect_left(level, evalue)
        assert(index >= 0)
        
        index = max(0, index-1)
        
        files_classified[index].append(file)

        if not exist_flag[index]:
            exist_flag[index] = 1

    return files_classified, exist_flag