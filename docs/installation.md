Modified from the official mmdet3d [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

# Prerequisites
D3PD is developed with the following version of modules.
- Linux or macOS (Windows is not currently officially supported)
- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3.1
- GCC 7.3.0
- MMCV==1.3.14
- MMDetection==2.14.0
- MMSegmentation==0.14.1


# Installation

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n D3PD python=3.8 -y
conda activate D3PD
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
or
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and other requirements.**

```shell
pip install -r requirements.txt
```

**f. Clone the D3PD repository.**

```shell
git clone https://github.com/no-Name128/D3PD.git
cd D3PD
```

**g.Install build requirements and then install D3PD.**

```shell
pip install -v -e .
```