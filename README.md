# Sparse R-CNN
End-to-End Object Detection with Learnable Proposal



## Installation
1. Install and build Detectron2 libs
```
git clone https://github.com/PeizeSun/SparseR-CNN.git
cd SparseR-CNN
python setup.py build develop
```
2. Link coco dataset path to SparseR-CNN/datasets/coco

2. Train SparseR-CNN
  * ```python projects/SparseR-CNN/train_net.py --num-gpus 8 --config-file projects/SparseR-CNN/configs/sparsercnn.res50.1x.yaml```
3. Evaluate SparseR-CNN using provided weights [here](https://drive.google.com/drive/folders/)
  * ```python projects/SparseR-CNN/train_net.py --num-gpus 8 --config-file projects/SparseR-CNN/configs/sparsercnn.res50.1x.yaml --eval-only MODEL.WEIGHTS path/to/provided/ckpt.pth```

## Citing

If you use SparseR-CNN in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposal},
  author  =  {},
  journal =  {arXiv preprint arXiv:},
  year    =  {2020}
}

```
