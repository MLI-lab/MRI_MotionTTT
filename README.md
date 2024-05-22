# MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI

This repository contains an implementation of the MotionTTT method as described in our manuscript MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI currently under review at NeurIPS2024.

## Requirements
CUDA-enabled GPU is necessary to run the code. We tested this code using:
- Ubuntu 20.04
- CUDA 11.3
- Python 3.10.14
- PyTorch 1.12.1
- torchkbnufft 1.4.0 [1]
- ptwt 0.1.8

## Usage

We start with the preparation of the training and validation/test data.
1. Start by downloading the [Calgary Campinas Brain MRI Dataset](https://portal.conp.ca/dataset?id=projects/calgary-campinas#) [2]. As the data is stored as in a hybrid k-space format run `CC359_convert_hybrid_kspace_to_kspace.py` create a converted version of the train and validation set consisting of 3D k-space data.
2. Next, install the BART toolbox  by following the instructions on their [home page](https://mrirecon.github.io/bart/) and run  `CC359_compute_sensitivity_maps.py` to pre-compute 3D sensitivity maps for training and validation set.

Now we can perfom the three steps of MotionTTT, 1) pre-training, 2) motion parameter estimation and 3) reconstruction.
1. Run `main_train.py` to pre-train the U-net [3] from the [fastMRI repository](https://github.com/facebookresearch/fastMRI) [4] on the train split specified in `volume_dataset_freqEnc170_train_len40.pickle` and undersampling mask in `mask_3D_size_218_170_256_R_4_poisson_disc.pickle`.
2. Run `main_motionTTT.py` to estimate motion parameters of one of the validation or test examples specifed in `volume_dataset_freqEnc170_val_len4.pickle` and `volume_dataset_freqEnc170_test_len5_1.pickle`, respectively. The number of motion events and maximal rotations/translations can be set in the script. Subsequently, L1-minimization is automatically performed based on the estimated motion parameters with DC loss thresholding.

Optionally, run `main_AltOpt.py` to compare to the results obtained with the alternating optimization baseline. 

## Acknowledgments and reference
- [1] Muckley et al. "TorchKbNufft: A High-Level, Hardware-Agnostic Non-Uniform Fast Fourier Transform2. In: *ISMRM Workshop on Data Sampling and Image Reconstruction* (2020).
- [2] Souza et al.  "An Open, Multi-Vendor, Multi-Field-Strength Brain MR Dataset and Analysis of Publicly Available Skull Stripping Methods Agreement". In: *NeuroImage* (2018).
- [3] Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation". In: *Medical Image Computing and Computer-Assisted Intervention* (2015).
- [4] Zbontar et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI". In: https://arxiv.org/abs/1811.08839 (2018).
