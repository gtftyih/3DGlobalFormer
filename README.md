# 3DGlobalFormer
3DGlobalFormer: Three Domain Global Feature Fusion in 3D Human Estimation

Thank you for your interest, the code and checkpoints are being updated.

## The released codes include
    checkpoint/:                        the folder for model weights of 3DGlobalFormer.
    dataset/:                           the folder for data loader.
    common/:                            the folder for basic functions.
    model/:                             the folder for 3DGlobalFormer network.
    run_3dgf.py:                        the python code for 3DGlobalFormer networks training.

## Environment
Make sure you have the following dependencies installed:
* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0

## Datasets
Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets.
### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). 
### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [P-STMO](https://github.com/paTRICK-swk/P-STMO). 

