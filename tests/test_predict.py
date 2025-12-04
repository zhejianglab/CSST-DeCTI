import glob
import os
from decti import DeCTI


def main():

    a = DeCTI(data_manager='csst_msc_sim', model_param='testdir/trained_model.param.pkl', model_state='testdir/trained_model.state.pth')

    flist1 = glob.glob('cti_v3.3.0/cti/10100059069/CSST_*_WIDE_*.fits')
    flist2 = [os.path.join('testdir/output', os.path.basename(f)) for f in flist1]

    a.batch_predict(flist1, flist2)


if __name__ == '__main__':
    main()
