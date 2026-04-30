import glob
import os
from decti import Trainer, init_manager, init_model


def main():

    manager = init_manager('csst_msc_sim')
    model, pars = init_model(model_name='DeCTIAbla', seq_len=9232)
    trainer = Trainer(model, manager, num_workers=2, batch_size=16)

    root = os.path.dirname(os.path.abspath(__file__))
    flist1 = glob.glob(os.path.join(root, 'data', 'csst_*_cti.fits.gz'))
    flist2 = [f.replace('_cti.fits.gz', '_nocti.fits.gz') for f in flist1]
    logpath = os.path.join(root, 'log')
    outpath = os.path.join(root, 'output')

    trainer.train(flist1, flist2, logpath, n_epochs=5, verbose=True)
    trainer.save(os.path.join(outpath, 'test_trained_model.pth'))


if __name__ == '__main__':
    main()
