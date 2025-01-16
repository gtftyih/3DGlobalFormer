# 3DGlobalFormer
3DGlobalFormer: Three Domain Global Feature Fusion in 3D Human Estimation

Thank you for your interest, the code and checkpoints are being updated.

## The released codes include
    checkpoint/:                        the folder for model weights of 3DGlobalFormer.
    dataset/:                           the folder for data loader.
    common/:                            the folder for basic functions.
    model/:                             the folder for 3DGlobalFormer network.
    demo/:                              the folder for visulization.
    run_global.py:                      the python code for 3DGlobalFormer-S and 3DGlobalFormer-B networks training and testing.
    run_global_L.py:                    the python code for 3DGlobalFormer-L networks training and testing.
    run_3dhp_global.py:                 the python code for tra1ining and testing on the MPI-INF-3DHP dataset.

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

## Evaluation
You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1MHIbJ82_IllUKwPuFA2zLmxcwV9p957a?usp=drive_link). Put them in the ./checkpoint directory.
### Human3.6M
To evaluate our 3DGlobalFormer model on the 2D keypoints obtained by CPN, please run:

3DGlobalFormer-S:
```bash
 python run_global.py -f 27 -b 128 --d_hid 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/your_best_epoch.pth
```
3DGlobalFormer-B:
```bash
 python run_global.py -f 243 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/your_best_epoch.pth
```
3DGlobalFormer-L:
```bash
 python run_global_L.py -f 243 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' --reload 1 --previous_dir ./checkpoint/your_best_epoch.pth
```
### MPI-INF-3DHP
The pre-trained models and codes for 3DGlobalFormer are currently undergoing updates. In the meantime, you can run this code to observe the results for 81 frames:
```bash
 python run_3dhp_global.py -f 81 -b 128 --train 0 --layers 6 -s 1 --reload 1 --previous_dir ./checkpoint/your_best_epoch.pth
```
## Training from scratch
### Human3.6M
To train our 3DGlobalFormer model on the 2D keypoints obtained by CPN, please run:

3DGlobalFormer-S:
```bash
 python run_global.py -f 27 -b 128 --d_hid 128 --train 1 --layers 6 -s 3
```
3DGlobalFormer-B:
```bash
 python run_global.py -f 243 -b 128 --train 1 --layers 6 -s 3
```
3DGlobalFormer-L:
```bash
 python run_global_L.py -f 243 -b 128 --train 1 --layers 6 -s 3
```
### MPI-INF-3DHP
To train our 3DGlobalFormer model, please run:
```bash
 python run_3dhp_global.py -f 81 -b 128 --train 1 --layers 6 -s 1
```

## Visulization
Accroding MHFormer, make sure to download the YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and put it in the './demo/lib/checkpoint' directory firstly. Then, you need to put your in-the-wild videos in the './demo/video' directory.

You can modify the 'get_pose3D' function in the 'vis.py' script according to your needs, including the checkpoint and model parameters, and then execute the following command:
```bash
 python demo/vis.py --video sample_video.mp4
```
