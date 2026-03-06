import os
import shutil

from decti import init_model, init_manager, Trainer


def main():

    root = os.path.dirname(os.path.abspath(__file__))
    imgpath = os.path.join(root, 'data')
    logpath = os.path.join(root, 'log')
    if 'RANK' not in os.environ or os.environ['RANK'] == 0:
        if os.path.exists(logpath):
            shutil.rmtree(logpath)

    manager = init_manager('csst_msc_sim')
    model, pars = init_model(model_name='DeCTIAbla', seq_len=9232)
    trainer = Trainer(model, manager, num_workers=4, batch_size=32)

    flist1 = [os.path.join(imgpath, 'csst_1_cti.fits'), os.path.join(imgpath, 'csst_2_cti.fits')]
    flist2 = [os.path.join(imgpath, 'csst_1_nocti.fits'), os.path.join(imgpath, 'csst_2_nocti.fits')]
    trainer.train(flist1, flist2, logpath, n_epochs=15, verbose=True)
    trainer.save(os.path.join(logpath, 'output.pth'))

if __name__ == '__main__':
    main()
