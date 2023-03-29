
This repository contains associated code for the ICLR 2023 paper  *Differentiable Mathematical Programming for Object-Centric Representation Learning*.

## Requirements

- PyTorch
- Pytorch Lightning
- CuPy
- Scikit-sparse
- Wandb
- MediaPy
- ClevrTex dataset: https://www.robots.ox.ac.uk/~vgg/data/clevrtex/ 

## Usage

Pretrained model checkpoint can be loaded using the Jupyter notebook for evaluation.

```bash
PYTHONPATH=. python slot_cut/train.py
```

Set hyperparameters in `slot_cut/params.py`


## Acknowledgements

The repository uses code from https://github.com/untitled-ai/slot_attention and resnet code from https://github.com/kuangliu/pytorch-cifar
