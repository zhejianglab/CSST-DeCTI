# DeCTI: Transformer-based Charge Transfer Inefficiency correction for CSST

## 1. Introduction
Charge Transfer Inefficiency (CTI) is a common defect in Charge-Coupled Device (CCD) imaging sensors, leading to charge trailing and signal distortion in astronomical images. The Chinese Space Station Telescope (CSST) acquires roughly two million images annually, all of which require CTI correction——posing an urgent demand for a solution that is both accurate and computationally efficient.

To address this challenge, we introduce **DeCTI**, a novel supervised deep learning pipeline designed to effectively mitigate CTI artifacts in astronomical images. As illustrated in the figures below, DeCTI restores the degraded raw image (left) to a high-fidelity reconstruction (middle) that closely matches the “ground truth” image (right). Compared with traditional state-of-the-art approaches, DeCTI achieves approximately **2× higher correction accuracy** and is over **100× faster**, enabling large-scale and high-fidelity image restoration.

<div align="center">
<img src="figs/vis_lq.png" width="23%" title="RAW"> <img src="figs/vis_pr.png" width="23%" title="prediction"> <img src="figs/vis_gt.png" width="23%" title="ground truth"> 
</div>

## 2. Architecture
The DeCTI architecture integrates convolutional layers for local feature extraction with Transformer encoders to model long-range charge trailing patterns. Key design highlights:

- Reformulating CTI correction as a 1-D sequence-to-sequence task by treating each column vector as an independent sample.

- Introducing a custom normalization method tailored to astronomical image distributions to stabilize and accelerate training.

- Employing a hybrid architecture that combines CNN layers and 1-D Transformer encoders within fixed processing windows.

<div align="center">
<img src="figs/DeCTI.png" width="80%" title="Architecture">
</div>

## 3. Evaluation Metrics
### Removal Ratio
We define the **removal ratio**, a custom metric designed to quantify residual CTI artifacts on a column-wise basis. It measures the fraction of remaining error after correction——**lower values indicate better performance**. From the distribution of removal ratio across multiple samples, we derive **bias** and **dispersion** metrics, which respectively characterize the central tendency and spread of the removal ratio distribution, reflecting both the accuracy and stability of the correction.

<div align="center">
<img src="figs/bias_rratio.png" width="40%" title="bias metrics"> <img src="figs/var_rratio.png" width="40%" title="dispersion metrics">  
</div>

### Relative Photometry Error
**Relative photometry error** is a standard astronomical metric that quantifies the flux deviation relative to the ground-truth flux on cropped 2-D image stamps. Models are trained and inferred separately on Hubble Space Telescope (HST) images observed in 2005 and 2012. Two flux-measurement methods—**Aperture** and **Kron**—are adopted for comparison. 

The error distributions for multiple objects are shown below. The horizontal axis denotes the ground-truth flux, while the dots and lines on the vertical axis represent the bias and standard deviation of the relative photometry error, respectively. From left to right, the panels correspond to: (a) Aperture flux (2005 data), (b) Aperture flux (2012 data), (c) Kron flux (2005 data), and (d) Kron flux (2012 data).

<div align="center">
<img src="figs/flux_aperture.png" width="40%" title="aperture flux"> <img src="figs/flux_kron.png" width="40%" title="kron flux">
</div>

### Speed
The table below compares the runtime performance of two existing state-of-the-art methods, SimpleCTI and arCTIc (both running on a single-core CPU), with our DeCTI model tested on 4 GPUs and 16 GPUs, respectively.
The rightmost column shows the average time per image, including both computation and I/O operations.

<div align="center">
<img src="figs/time_consuming.jpeg" width="80%" title="aperture flux">
</div>

## 4. Dataset
Each model is trained on **public-domain** Hubble Space Telescope (HST) observations from a single year, obtained with the ACS camera using the F814W optical filter. Filenames used for training, validation, and testing are listed in the corresponding files: [train](config/remove_j92t/train.csv)↗ [validation](config/remove_j92t/val.csv)↗ [test](config/remove_j92t/test.csv)↗. The images can be downloaded using their ```observation_id``` via  [astroquery](https://astroquery.readthedocs.io/en/latest/esa/hubble/hubble.html)↗.

## 5. Dependency
All software dependencies required to run the project are listed in [environment.yaml](environment.yaml)↗. To create or update the Conda environment, run the following command:

```bash
conda env update -f environment.yaml
```  

Please note that the environment includes all third-party libraries used in this work, including ```tensorboard```, ```pytorch```, ```numpy```, ```matplotlib```, ```pandas```, ```fitsio```, ```scikit-learn```, ```seaborn```, ```astroquery```, etc, all of which are essential for model development and evaluation. Users are encouraged to respect the respective licenses when using these tools.

## 6. File Structure
Below is the directory structure of this repository. It provides an overview of the main scripts and configuration files used in the project.

```latex
csst-DeCTI/
|
├── baseline.sh            # Script for training or inference
├── config                 # Configuration files listing dataset filenames
│   └── remove_j92t
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
├── data_provider          # I/O modules for data loading and organization
│   ├── data_factory.py
│   ├── data_loader.py
├── environment.yaml       # Conda environment configuration
├── LICENSE
├── main.py                # Main entry point of the project
├── models                 # Model architectures
│   ├── DeCTIAbla.py       # Core DeCTI model
│   ├── DnCNN.py           # Other SOTA methods
├── pipeline               # Training and evaluation pipelines
│   ├── exp_basic.py       # Project initialization
│   ├── exp_main.py        # Main experiment workflow
├── utils/                 # Utility functions
│   └── tools.py
└── README.md
```

## 7. License
This code repository are licensed under the [Apache License 2.0](https://github.com/zhejianglab/CSST-DeCTI/blob/main/LICENSE). The HST data hosted at MAST used in this work are in the public domain and are free to use. Users should note, however, that other data hosted at MAST may be subject to specific use restrictions, and compliance with any applicable license terms is required.

## 8. Citation and Acknowledgements
This work is based on observations made with the **NASA/ESA Hubble Space Telescope**, obtained from the **Mikulski Archive for Space Telescopes (MAST)**. The Space Telescope Science Institute (STScI), which operates the HST and manages the MAST archive, is operated by the Association of Universities for Research in Astronomy, Inc. (AURA) under NASA contract NAS5-26555. STScI provides the scientific support, data processing, and archival services that make HST data publicly accessible.

This work is also supported by the **China Manned Space Program** through its Space Application System.

If you use this work in your research, please cite the following paper:
```latex
@article{Men2025ChargeTransfer,
  author  = {Z. Men, L. Shao, P. Smirnov, M. Duan},
  title   = {DeCTI: Transformer-based Charge Transfer Inefficiency correction},
  journal = {IEEE Transactions on Image Processing},
  note    = {under review},
  year    = {2025}
}
```
