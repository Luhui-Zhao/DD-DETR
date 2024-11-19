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
# Copy my py file past
```
6. Monitor real-time GPU performance
```sh
pip install nvitop
nvitop
```
   
# Dataset Preparation
## Project Structure

### CityScapes → CityScapes Foggy
* **CityScapes**: Please download it from the official [website](https://www.cityscapes-dataset.com/downloads/).Images ***leftImg8bit_trainvaltest.zip (11GB) [md5]***; Annotations ***gtFine_trainvaltest.zip (241MB) [md5]***.
* **Foggy CityScapes**: Download from the official [website](https://www.cityscapes-dataset.com/downloads/). Images ***leftImg8bit_trainval_foggyDBF.zip (20GB) [md5]***; Annotations are the same with `CityScapes`. Note, we chose foggy images with `beta=0.02` out of three kind of choices `(0.005,0.01, 0.02)`.
* **Normal-style → Foggy-style and Foggy-style → Normal-style**:We use datasets that have been transformed using [CUT(ECCV2020)](https://github.com/taesungp/contrastive-unpaired-translation) in [SSDA-YOLO](https://github.com/hnuzhy/SSDA-YOLO)
* **VOC foramt → coco format**:You can use the open source conversion code [cityscapes-to-coco-conversion](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion) or the converted coco format Annotations file I uploaded to [Google Drive]()
# Model zoo
| Task                                     | mAP50  | Config | Model | Where in Our Paper |
|:----------------------------------------:|:------:|:------:|:-----:|:------------------:|
| Cityscapes to Foggy Cityscapes(pretrain) | 62.05% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to Foggy Cityscapes           | 53.52% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to BDD100K-daytime(pretrain)  | 53.0%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to BDD100K-daytime            | 36.57% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Sim10k to Cityscapes(pretrain)           | 76.86% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Sim10k to Cityscapes                     | 55.8%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |


