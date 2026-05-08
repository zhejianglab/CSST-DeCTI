import os
from decti import DeCTI, init_model, init_manager


def main():

    root = os.path.dirname(os.path.abspath(__file__))

    model, params = init_model(seq_len=9232)
    data_manager = init_manager("csst_msc_sim")
    a = DeCTI(model, data_manager, state_path=os.path.join(root, 'data', 'trained_model.pth'))

    flist1 = [os.path.join(root, 'data', 'csst_1_cti.fits.gz')]
    flist2 = [os.path.join('testdir/output', os.path.basename(f).replace('.gz', '')) for f in flist1]
    a.batch_predict(flist1, flist2)


if __name__ == '__main__':
    main()
