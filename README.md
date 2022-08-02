# Transformer Knowledge Distillation for Efficient Semantic Segmentation [[arxiv](https://arxiv.org/abs/2202.13393)]
## Structure: TransKD
![TransKD](TransKDs.png)
## Introduction
We propose the structural framework, TransKD, to distill the knowledge from feature maps and patch embeddings of vision transformers.
## Usage
download [teacher checkpoints](https://1drv.ms/u/s!AlFXMOI-DJJhn3qvs5TOQlaWbbVr?e=ohlhOU) in the folder `checkpoints/`.

Example:
```
python train_TransKD/train_transkd.py --datadir /path/to/data --kdtype TransKD-Base
```

## Publication
If you find this repo useful, please consider referencing the following paper [[PDF](https://arxiv.org/pdf/2202.13393)]:
```
@article{liu2022transkd,
  title={TransKD: Transformer Knowledge Distillation for Efficient Semantic Segmentation},
  author={Liu, Ruiping and Yang, Kailun and Roitberg, Alina and Zhang, Jiaming and Peng, Kunyu and Liu, Huayao and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2202.13393},
  year={2022}
}
```
