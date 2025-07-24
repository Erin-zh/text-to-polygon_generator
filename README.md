<h1 align="Center"> A Text-only Weakly Supervised Learning Framework for Text Spotting via Text-to-Polygon Generator </h1> 
This is the official repo for the paper "A Text-only Weakly Supervised Learning Framework for Text Spotting via Text-to-Polygon Generator".

## 💡 Introduction

<img src="./text-to-polygon_generator.jpg" alt="image" style="zoom:50%;" />

## Usage
The environment setup follows the configuration used in the original [DPText-DETR](https://github.com/ymy-k/DPText-DETR?tab=readme-ov-file#installation) repository for compatibility. It is recommended to use Anaconda. Python 3.8 + PyTorch 1.9.1 (or 1.9.0) + CUDA 11.1 + Detectron2 (v0.6) are suggested.

- ### Installation
```
conda create -n project_name python=3.8 -y
conda activate project_name
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python scipy timm shapely albumentations Polygon3
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install setuptools==59.5.0
git clone https://github.com/Erin-zh/text-to-polygon_generator.git
cd text-to-polygon_generator
python setup.py build develop
```
- ### Preparing Datasets
We follow the dataset preparation procedure from [TESTR](https://github.com/mlpc-ucsd/TESTR?tab=readme-ov-file#preparing-datasets) to ensure compatibility with the original data format.
Please download TotalText, CTW1500, and MLT-2017, according to the guide provided by AdelaiDet: [README.md](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md).
And ICDAR2015 dataset can be download via [link](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/EWgEM5BSRjBEua4B_qLrGR0BaombUL8K3d23ldXOb7wUNA?e=7VzH34).
- ### Models
You can train from scratch or finetune the model by putting pretrained weights in `weights` folder.

Text-to-Polygon Generator: [syntext_only](https://1drv.ms/u/c/7ba880629a78428d/EfJkETCFo7xPtJRLw_HkdlgB6BLVVZWjVnBNudD2e29PbA?e=xuIDMe),  [syntext+mlt](https://1drv.ms/u/c/7ba880629a78428d/EXOvKWxfIphGo_uFqiWpElIBVySvq7PGuPQtJgb_2x41Tg?e=yLbaSe)

Chacter-level Text Encoder: [text_encoder](https://1drv.ms/u/s!AlK-tsynJvWIgV2W_gd7f2L7HXfR?e=4zeZvZ)

## Training Details
- ### Training
To train the text-to-polygon generator, please use the configuration file: `configs/Pretrain/R_50_poly.yaml`.  Make sure to adjust the number of GPUs and batch size according to your situation.
For example, a single NVIDIA RTX 4090 (24GB) can typically support a batch_size of 3.
To start training, run:
```
python tools/train_net.py --config-file configs/Pretrain/R_50_poly.yaml
```
- ### Evaluation
```
python tools/train_net.py --config-file ${CONFIG_FILE} --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```

## Acknowledgement
Our work is inspired a lot by [DPText-DETR](https://github.com/ymy-k/DPText-DETR?tab=readme-ov-file#installation) and [TESTR](https://github.com/mlpc-ucsd/TESTR). Thanks for their great works!

