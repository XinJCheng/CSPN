# Learning depth with convolutional spatial propagation network

By Xinjing Cheng, Peng Wang and Ruigang Yang

## Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Citation](#citation)

<!---3. [Usage](#usage)
4. [Citation](#citation)--->

## Introduction

PaddlePaddle implementation for CSPN Stereo Matching.

## Requirements

All the codes are tested in the following environment:

- Linux (tested on CentOS 7.5)
- Python 3.6+
- PaddlePaddle 1.5.2 with resize_trilinear and affinity_propogate

### Install the gpu version of [PaddlePaddle](http://paddlepaddle.org/)

- Installation

  - Download whl file. [Click here](https://drive.google.com/file/d/1YqkbflmadV_dLuKcpMlqoy-5mCrKmWDy/view?usp=sharing).
  - Install the package into python.

  ```bash
  python -m pip install paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_x86_64.whl
  ```

- Verification

  - Test your installation.

  ```python
  import paddle.fluid
  paddle.fluid.install_check.run_check()
  ```

  - Check resize_trilinear usage. [Click here](https://www.paddlepaddle.org.cn/documentation/docs/en/api/layers/resize_trilinear.html).
  - Display the documentation of affinity_propogate function.

  ```python
  import paddle.fluid
  help(paddle.fluid.layers.affinity_propagate)
  ```

  **Note**: A gate_weight tensor with shape [N, `kernel_size**2-1`, ...] is required when the shape of input tensor is [N, C, ...]. And it should be normalized in the channel dimension.
  
  **Note**: gate_weight would be shared in the channel dimension for input when C>1. And it is feasible to assign mutually independent gate_weight to each channel of input tensor by slicing input tensor along the channel dim and then employ the affinity_propagate function on the slices.

- Demo

  A simple demo of CSPN Module is available in this repo.

  - Clone this repository.

  ```bash
  git clone https://github.com/XinJCheng/CSPN.git
  ```

  - Run the demo for 2d cspn module.

  ```python
  cd ./CSPN/cspn_paddle
  python demo.py --dimNum=2
  ```

  - Run the demo for 3d cspn module.

  ```python
  cd ./CSPN/cspn_paddle
  python demo.py --dimNum=3
  ```

<!---

## Data Preparation

- Download Scene Flow Datasets. [Click here](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#downloads).
- Generate data list file.

--->

<!---## Usage

Coming soon.--->

<!---

### Train

It is recommended to use run.sh to train and validate the cspn model. Details can be seen in run.sh.

#### Train with cpu

```bash
python main.py --phase=train --model=stereo --stereoType=cspn --dataLists /path/to/list_file --dataRoots /path/to/dataset --batchSizes 4 --gpuIds -1
```

**Note**: batchSizes means batch size per device.

#### Train with multi gpus

```bash
python main.py ... --gpuIds 1,3
```

#### Train with validation

```bash
python main.py --phase=train_val --model=stereo --stereoType=cspn --dataLists /path/to/train/list_lile /path/to/val/listFile --dataRoots /path/to/dataset --batchSizes 6 4 --gpuIds 1 2 3
```

#### Train with pretrained model

```bash
python main.py ... --preModels /path/to/premodel1 /path/to/premodel2 --premodelFWs pytorch paddle
```

#### Resume training

```bash
python main.py ... --snapshots /path/to/snapshot
```

### Validate or Test

```bash
# validation
# ${PHASE}=val
# testing
# ${PHASE}=test
python main.py --phase=${PHASE} ... --snapshots /path/to/snapshot1 /path/to/snapshot2
```

The snapshot is available here. [Click here]().

--->

## Citation

If you use this method in your research, please cite:

```
@article{cheng2018learning,
  title={Learning depth with convolutional spatial propagation network},
  author={Cheng, Xinjing and Wang, Peng and Yang, Ruigang},
  journal={arXiv preprint arXiv:1810.02695},
  year={2018}
}
```
