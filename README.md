# DD-DETR
Open source code and model for DD-DETR

# Installation
**Environment:** Anaconda, Python=3.9, PyTorch=2.0.0, torchvision=0.15.1(CUDA11.8), wandb
1. Clone this repo
```sh
git clone 
cd 
```
2. Create a virtual environment and install Pytorch
```sh
conda create -n label python=3.9 
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 
```
3. Install other needed packages
```sh
pip install -r requirements.txt
```
4. Compiling CUDA operators
```sh
cd models/dab_deformable_detr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```
   
# Dataset Preparation

# Model zoo
