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
5. Change the data load code
```sh
# Open a new command line window
Ctrl+Alt+t
# Open the package for data loading
cd anaconda3/envs/XXX/lib/pythonx.x/site-packages/torchvision/datasets
# Delete coco.py
rm coco.py
# Delete coco.py
```
6. Monitor real-time GPU performance
```sh
pip install nvitop
nvitop
```
   
# Dataset Preparation

# Model zoo
| Task                                     | mAP50  | Config | Model | Where in Our Paper |
|:----------------------------------------:|:------:|:------:|:-----:|:------------------:|
| Cityscapes to Foggy Cityscapes(pretrain) | 62.05% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |
| Cityscapes to Foggy Cityscapes           | 53.52% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |
| Cityscapes to BDD100K-daytime(pretrain)  | 53.0%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |
| Cityscapes to BDD100K-daytime            | 36.57% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |
| Sim10k to Cityscapes(pretrain)           | 76.86% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |
| Sim10k to Cityscapes                     | 55.8%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table 1                   |


