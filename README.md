# MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI

This repository contains the implementation of the MotionTTT method as described in our paper [MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI](https://arxiv.org/abs/2409.09370) published at NeurIPS 2024.

## Requirements

You can install the requirements for example using an [Anaconda](https://www.anaconda.com/download) environment:
```
conda create -n MotionTTT_env python=3.10.14 pip=24.2
conda activate MotionTTT_env
```
CUDA-enabled GPU is necessary to run the code. You should install a version of `PyTorch` that is compatible with your `CUDA` version. You can find a list of all `PyTorch` versions and the compatible `CUDA` versions [here](https://pytorch.org/get-started/previous-versions/). We tested MotionTTT with
```
conda install pytorch==2.3.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
The remaining requirements can be istalled via
```
pip install -r requirements.txt
```

## Usage

### Data preparation
For pre-training on motion-free 2D slices and testing our method on 3D volumes with simulated motion parameters we use the data provided in the training and validation set of the [Calgary Campinas Brain MRI Dataset (CC59)](https://portal.conp.ca/dataset?id=projects/calgary-campinas#) [1].

Start by downloading the CC359 datset. As the data comes in a hybrid k-space format run `prepare_CC359_dataset.py` to create a converted version of the train and validation set consisting of 3D k-space data and to compute and store the 3D sensitivity maps using the [bart toolbox](https://mrirecon.github.io/bart/). 

### MotionTTT tutorial
The notebook `run_motionTTT_on_CC359_example.ipynb` demonstrates how to use our codebase to apply MotionTTT to an example from the CC359 validation set with a simulated motion and sampling trajectory using our pre-trained U-net from https://huggingface.co/mli-lab/Unet48-2D-CC359.

### Replicate the results from the paper
In `codebase/` you can find scripts to replicate the results from the paper including
- `main_train_unet_motionFree.py` to train a U-net for motion-free undersampled 2D MRI reconstruction on our training set `codebase/data_files/volume_dataset_freqEnc170_train_len40.pickle` from CC359 resulting into the model provided at https://huggingface.co/mli-lab/Unet48-2D-CC359,
- `main_motionTTT_simData.py` to run MotionTTT for inter-/intra-shot motion estimation on our test set `codebase/data_files/volume_dataset_freqEnc170_test_len10.pickle` from CC359,
- `main_L1min_simData.py` to obtain lower and upper bound reconstruction performances with no motion correction and oracle known motion parameters,
- `main_altopt_simData.py` our implementation of the idea in [2] to obtain the baseline results for alternating optimization motion estimation and reconstruction,
- `main_train_stackedUnet_motionCorrection.py` our implementation of stacked U-nets with self-assisted priors [3] to obtain the baseline results for end-to-end 3D motion correction.

Run any of those scripts as, e.g.,
```
cd codespace
python main_motionTTT_simData.py
```

## vivo Data Reconstruction
We also support inference on the [PMoC3D](https://huggingface.co/datasets/mli-lab/PMoC3D) dataset, a real-world dataset consisting of paired motion-corrupted 3D brain MRI data.

**Step 1: Crop PMoC3D Data**

To reduce the computational cost, we also cropped the data along the fully-sampled read-out dimension. You can use the following command to crop the data:
```bash
python crop_PMoC3D_dataset.py \
  --pmoC3d_root /path/to/PMoC3D \
  --gpu 3 \
  [--save_path /path/to/output]
```

- `--pmoC3d_root` (required): Root directory of the downloaded PMoC3D dataset.
- `--gpu` (required): CUDA device ID to use.
- `--save_path` (optional): Output directory for the cropped data.  
  Defaults to `PMoC3D_root/derivatives/cropped_data`.


**Step 2: Running baselines on PMoC3D**

We support the MotionTTT, AltOpt, and stacked U-net running on the vivo data, an example using MotionTTT is:
```bash
cd codespace
python main_motionTTT_vivoData.py
```

## Contact

If you have any questions or problems, or if you found a bug in the code, please open an issue on GitHub or contact us.


## How to cite this work

 @inproceedings{  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Klug_Wang_Ruschke_Heckel_2024,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; title={MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; booktitle={Conference on Neural Information Processing Systems},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; author={Klug, Tobit and Wang, Kun and Ruschke, Stefan and Heckel, Reinhard},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; year={2024}  
}


## References
- [1] Souza et al.  "An Open, Multi-Vendor, Multi-Field-Strength Brain MR Dataset and Analysis of Publicly Available Skull Stripping Methods Agreement". In: *NeuroImage* (2018).
- [2] Cordero-Grande et al. "Sensitivity Encoding for Aligned Multishot Magnetic Resonance Reconstruction". In: *IEEE Transactions on Computational Imaging* (2016).
- [3] Al-Masni et al. "Stacked U-Nets with Self-Assisted Priors towards Robust Correction of Rigid Motion Artifact in Brain MRI". In: *NeuroImage* (2022).

## License

All files are provided under the terms of the BSD 2-Clause license.
