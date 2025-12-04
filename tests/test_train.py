import glob
from decti.dataset import load_manager
from decti.trainer import Trainer


def main():

    data = load_manager('csst_msc_sim')

    flist1 = glob.glob('test_training/cti_v3.3.0/cti/10100059069/CSST_*_WIDE_*_L0_V01.fits')
    flist2 = glob.glob('test_training/cti_v3.3.0/nocti/10100059069/CSST_*_WIDE_*_L0_V01.fits')

    a = Trainer(flist1, flist2,
                data_manager=data, model_param={'data_length': data.ny},
                batch_size=16, loader_workers=2, backend='nccl')
    a.train(patience=5, log_path='./testdir/log')
    a.save('./testdir/trained_model.pth')


if __name__ == '__main__':
    main()
