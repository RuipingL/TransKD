# Transformer-based Knowledge Distillation for Efficient Semantic Segmentation of Road-driving Scenes [[arxiv](https://arxiv.org/abs/2202.13393)]
## Structure: TransKD
![](https://github.com/RuipingL/SKR_PEA/blob/main/structure.PNG)
## Introduction
We propose the structural framework, TransKD, to distill the knowledge from feature maps and patch embeddings of vision transformers.
## Usage
download [teacher checkpoints](https://1drv.ms/u/s!AlFXMOI-DJJhn3qvs5TOQlaWbbVr?e=ohlhOU) in the folder `\checkpoints`.
Example:
```
python train_TransKD/train_transkd.py --datadir /path/to/data --kdtype TransKD-Base
```
