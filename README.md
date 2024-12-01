# DPGT
Open source code and model for DPGT
## Outline

1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Model Zoo](#Model-zoo)
4. [Training and Evaluation](#Training-and-Evaluation)
5. [Citation and Acknowledgement](#Citation-and-Acknowledgement)
# Installation
**Environment:** Two NVIDIA A6000 GPUs, Anaconda, Python=3.9, PyTorch=2.0.0, torchvision=0.15.1(CUDA11.8), wandb
1. Clone this repo
```sh
git clone 
cd 
```
2. Create a virtual environment and install [Pytorch](https://pytorch.org/get-started/previous-versions/)
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
## Dataset Structure
```
[DATASET_PATH]
└─ city2foggy
   └─ annotations
      └─ sr
         └─ instances_train.json
         └─ instances_val.json
      └─ sf
         └─ instances_train.json
         └─ instances_val.json
      └─ tr
         └─ instances_train.json
         └─ instances_val.json
      └─ tf
         └─ instances_train.json
         └─ instances_val.json
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_sf
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
   └─ leftImg8bit_foggy_tf
      └─ train
      └─ val
└─ city2bdd100k
   └─ annotations(Same as city2foggy)
   └─ cityscapes
      └─ train
   └─ city2bdd
      └─ train
   └─ bdd100k
      └─ train
      └─ val
   └─ bdd2city
      └─ train
      └─ val
└─ Sim10K2city
   └─ annotations(Same as city2foggy)
   └─ sim10k
   └─ sim10k2city
   └─ cityscapes
      └─ train
      └─ val
   └─ city2sim10k
      └─ train
      └─ val
```


### CityScapes → CityScapes Foggy
* **CityScapes**: Please download it from the official [website](https://www.cityscapes-dataset.com/downloads/).Images ***leftImg8bit_trainvaltest.zip (11GB) [md5]***; Annotations ***gtFine_trainvaltest.zip (241MB) [md5]***.
* **Foggy CityScapes**: Download from the official [website](https://www.cityscapes-dataset.com/downloads/). Images ***leftImg8bit_trainval_foggyDBF.zip (20GB) [md5]***; Annotations are the same with `CityScapes`. Note, we chose foggy images with `beta=0.02` out of three kind of choices `(0.005,0.01, 0.02)`.
* **Normal-style → Foggy-style and Foggy-style → Normal-style**:We use datasets that have been transformed using [CUT(ECCV2020)](https://github.com/taesungp/contrastive-unpaired-translation) in [SSDA-YOLO](https://github.com/hnuzhy/SSDA-YOLO).
* **VOC foramt → coco format**:You can use the open source conversion code [cityscapes-to-coco-conversion](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion) or the converted coco format Annotations file I uploaded to [Google Drive]().
### CityScapes → BDD100K-daytime
* **CityScapes**: Please download it from the official [website](https://www.cityscapes-dataset.com/downloads/).Images ***leftImg8bit_trainvaltest.zip (11GB) [md5]***.
* **BDD100K-daytime**: Download from the official [website](https://dl.cv.ethz.ch/bdd100k/data/). Images ***100k_images_train.zip***.
* **Cityscapes-style → daytime-style and daytime-style → Cityscapes-style**:We use [CUT(ECCV2020)](https://github.com/taesungp/contrastive-unpaired-translation) for image style conversion.
* **Annotations：coco format**:Converted coco format annotation file I uploaded to [Google Drive]().
### Sim10k → CityScapes
* **Sim10k**: Download from the official [website](https://fcav.engin.umich.edu/projects/driving-in-the-matrix).
* **CityScapes**: Please download it from the official [website](https://www.cityscapes-dataset.com/downloads/).Images ***leftImg8bit_trainvaltest.zip (11GB) [md5]***.
* **Cityscapes-style → GTA5-style and GTA5-style → Cityscapes-style**:We use [CUT(ECCV2020)](https://github.com/taesungp/contrastive-unpaired-translation) for image style conversion.
* **Annotations：coco format**:Converted coco format annotation file I uploaded to [Google Drive]().
# Model zoo

| Task                                     | mAP50  | Config | Model | Where in Our Paper |
|:----------------------------------------:|:------:|:------:|:-----:|:------------------:|
| coco pre-training model                  | —     | —                                                | [DAB-Deformable-DETR-R50-v2](https://github.com/IDEA-Research/DAB-DETR?tab=readme-ov-file#model-zoo)      | —                   |
| Cityscapes to Foggy Cityscapes(pretrain) | 62.05% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to Foggy Cityscapes           | xx.xx% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to BDD100K-daytime(pretrain)  | 53.0%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Cityscapes to BDD100K-daytime            | xx.xx% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Sim10k to Cityscapes(pretrain)           | 76.86% | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
| Sim10k to Cityscapes                     | xx.xx%  | [cfg](config/DA/Cityscapes2FoggyCityscapes)       | [Google Drive]()      | Table x                   |
# Training and Evaluation
| Train or Test | Task                                      | command |
|---------------|-------------------------------------------|---------|
| Train         |  Cityscapes to Foggy Cityscapes(pretrain) | [cmd_train_c2f]()       |
| Train         | Cityscapes to Foggy Cityscapes            | [cmd_train_c2f]()       |
| Train         | Cityscapes to BDD100K-daytime(pretrain)   | [cmd_train_c2b]()       |
| Train         | Cityscapes to BDD100K-daytime             | [cmd_train_c2b]()       |
| Train         | Sim10k to Cityscapes(pretrain)            | [cmd_train_s2c]()       |
| Train         | Sim10k to Cityscapes                      | [cmd_train_s2c]()       |
| Test          | Cityscapes to Foggy Cityscapes            | [cmd_test_c2f]()       |
| Test          | Cityscapes to BDD100K-daytime             | [cmd_test_c2b]()       |
| Test          | Sim10k to Cityscapes                      | [cmd_test_s2c]()       |

# Citation and Acknowledgement
