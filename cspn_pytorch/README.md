# Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network

By Xinjing Cheng*, Peng Wang* and Ruigang Yang (*Equal contribution)

## Contents
0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Models](#models)
0. [Testing](#testing)
0. [Training](#training)
0. [Citation](#citation)

## Introduction

This repo contains the CNN models trained for depth completion from a RGB and sparse depth points, as described in the paper "[Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.pdf)". 
The provided models are those that were used to obtain the results reported in the paper on the benchmark datasets NYU Depth v2 and KITTI for indoor and outdoor scenes respectively. Moreover, the provided code can be used for inference on arbitrary images. 

Notice: there is a minor formulation error in the original paper, i.e. way to apply affinity kernel at center pixel (please check code for details, we will rectify it in our journal submission.)

## Requirements

This code was tested with Python 3 and PyTorch 0.4.0.
- Install [PyTorch](http://pytorch.org/) on a machine with CUDA GPU.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and other dependencies (files in our pre-processed datasets are in HDF5 formats).
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py pandas matplotlib imageio scikit-image opencv-python
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset in HDF5 formats, and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space, and KITTI requires 81G.
- for NYU dataset
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
    mv nyudepthv2 nyudepth_hdf5
    ```
- for KITTI dataset
    ```bash
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
 	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
    mv kitti kitti_hdf5
	cd ..
	```

## Models

The pretrained models, resnet18 and resnet50 can be downloaded here, and please put it into **pretrained** folder, if this folder does not exit, please run **mkdir pretrained** in root folder:
- Resnet 18: [Pytorch model](https://drive.google.com/file/d/17adZHo5dkcU8_M_6OvYzGUTDguF6k-Qu/view?usp=sharing)
- Resnet 50: [Pytorch model](https://drive.google.com/file/d/1-jSYATFPmyXoV0Qte6kLK-CD2nTtjNlD/view?usp=sharing)

The trained models, namely **+UNet+CSPN** in the paper can be downloaded here:

- NYU Depth V2: [Pytorch model]() (Deprecated, only results images are available)
- NYU Depth V2 (Fast Unpool, pos): [Pytorch model](https://drive.google.com/file/d/1MM_ZPsB2Bb3c_D3cD-rLJta3Qo7A7i50/view?usp=sharing)
- NYU Depth V2 (Fast Unpool, non-pos): [Pytorch model](https://drive.google.com/open?id=1iJ-GzS9xm6IA07T0izjCvCMbP422ORks)
- KITTI: Pytorch model(coming soon)

Download it under  `output/${dataset}_pretrain_cspn_${model_config}/`, where `dataset` could be `nyu` or `kitti`, 
where `model_config` can be checked from `eval_nyudepth_cspn.sh`


## Testing
- For NYU Depth v2

Here we provide example for the model of `NYU(Fast Unpool, non-pos affinity)`. 
Download the model from above link and put it under `output/nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40_8norm`, then run, 

```bash
    bash eval_nyudepth_cspn.sh
```

Run it multiple times and take mean, you should able obtain our depth results close here (5 time average due to randomness of sampled sparse points): 

| Data | RMSE | REL | DELTA1.02 | DELTA1.05 | DELTA1.10 | Results |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
|`NYU(Slow Unpool)`| 0.1165| 0.0159 | 0.8331 | 0.9366 | 0.9716| [Download](https://drive.google.com/open?id=1mPGil99_46eXK7w4hb-XHDUL-hTrKhXf) |
|`NYU(Fast Unpool, pos affinity)`| 0.1169 | 0.0161 | 0.8300 | 0.9347 | 0.9708| [Download](https://drive.google.com/file/d/1WzL1jd5KVYfwY9Rds9WxjvOL1bhk-k4J/view?usp=sharing) |
|`NYU(Fast Unpool, non-pos affinity)`| 0.1172 | 0.0160 | 0.8344 | 0.9351 | 0.9707| [Download](https://drive.google.com/open?id=1nJkxw_FopEtUt1XY0aGPZ-WlzF2o_KjA) |

Here, the `Slow Unpool` means we originally loop over the image for unpooling. the `Fast Unpool` means we use adopt transpose conv to implement the unpooling. `pos affinity` means we enforce the affinity to be positive, i.e. affinities are normalized in [0, 1). `non-pos affinity` means we allow negative affinity, i.e. affinity are normalized in (-1, 1). 


## Training
- For NYU Depth

You may set the configuration in `train_cspn_nyu.sh` and run, 

```bash
    bash train_cspn_nyu.sh
```

We train with a Nvidia 1080Ti GPU,  and within 40 epochs you should be able to get results close to what we reported above.


## Citation

If you use this method in your research, please cite:
```
@inproceedings{cheng2018depth,
  title={Depth estimation via affinity learned with convolutional spatial propagation network},
  author={Cheng, Xinjing and Wang, Peng and Yang, Ruigang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={103--119},
  year={2018}
}
```
