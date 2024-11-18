# DD-DETR
Open source code and model for DD-DETR

# Installation
**Environment:** Anaconda, Python=3.9, PyTorch=2.0.0, torchvision=0.15.1(CUDA11.8), wandb

```bash
$ git clone https://github.com/hnuzhy/SSDA-YOLO.git
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Codes are only evaluated on GTX3090+CUDA11.2+PyTorch1.10.0. You can follow the same config if needed
# [method 1][directly install from the official website][may slow]
$ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
  
# [method 2]download from the official website and install offline][faster]
$ wget https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ pip3 install torch*.whl
```
# Dataset Preparation

# Model zoo
