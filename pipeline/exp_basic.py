import os
import torch
import numpy as np
import torch.distributed as dist
from utils.tools import init_seeds

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._init_distributed_mode()
        init_seeds(self.args.random_seed, self.args.rank)

        self.model = self._build_model()
        print('Use GPU: cuda:{} rank:{}'.format(self.args.gpu, self.args.rank))

    def _build_model(self):
        raise NotImplementedError
        return None

    def _init_distributed_mode(self):
        device = torch.device('cuda:{}'.format(self.args.gpu))
        if self.args.use_gpu:
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                self.args.rank = int(os.environ["RANK"])
                self.args.world_size = int(os.environ["WORLD_SIZE"])
                self.args.gpu = int(os.environ["LOCAL_RANK"])
                self.args.distributed = True
                device = torch.device('cuda:{}'.format(self.args.gpu))               
            else:   
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                self.args.rank = 0
                self.args.world_size = 1
                self.args.distributed = True

                return device

            dist.init_process_group(backend='nccl', init_method='env://', world_size=self.args.world_size, rank=self.args.rank)
            print('device:  {}'.format(device))
            dist.barrier()
            return device
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
