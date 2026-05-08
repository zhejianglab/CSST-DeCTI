import os
from astropy.io import fits

from decti import init_manager, init_model, DeCTI


def main():

    root = os.path.dirname(os.path.abspath(__file__))
    imgpath = os.path.join(root, 'data')
    outpath = os.path.join(root, 'output')

    manager = init_manager('csst_msc_sim')
    model, pars = init_model(model_name='DeCTIAbla', seq_len=9232)
    fstate = os.path.join(imgpath, 'trained_model.pth')
    predictor = DeCTI(model, manager, state_path=fstate)

    finput = os.path.join(imgpath, 'csst_1_cti.fits.gz')
    foutput = os.path.join(outpath, 'test_predict_output.fits.gz')
    with fits.open(finput) as hdulist:
        output = predictor.predict(hdulist[1].data, batch_size=32, device='cuda', verbose=True)
        hdulist[1].data = output
        hdulist.writeto(foutput, overwrite=True)


if __name__ == '__main__':
    main()
