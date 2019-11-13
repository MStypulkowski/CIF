
![mainimg](https://github.com/MStypulkowski/CIF/blob/master/CIF_results.png)

# Conditional Invertible Flow for Point Cloud Generation

This is a PyTorch implementation of the paper:

[Conditional Invertible Flow for Point Cloud Generation](https://arxiv.org/abs/1910.07344) <br>
Michał Stypułkowski, Maciej Zamorski, Maciej Zięba, Jan Chorowski <br>
NeurIPS 2019 workshop on Sets & Partitions

## Introduction
This paper focuses on a novel generative approach for 3D point clouds that makes use of invertible flow-based models. The main idea of the method is to treat a point cloud as a probability density in 3D space that is modeled using a cloud-specific neural network. To capture the similarity between point clouds we rely on parameter sharing among networks, with each cloud having only a small embedding vector that defines it. We use invertible flows networks to generate the individual point clouds, and to regularize the embedding vectors. We evaluate the generative capabilities of the model both in qualitative and quantitative manner.

## Requirements
Stored in `requirements.txt`, Python dependencies are:
```
torch
torchvision
scikit-learn
numpy
matplotlib
scipy
Cython
pyyaml
```

## Training
Run the training process with:

`python experiments/train/train_model.py --config configs/cif_train.yaml`

## Evaluation
### Reconstruction of the training set:

`python experiments/test/train_reconstruction.py --config configs/cif_eval.yaml`

### Reconstruction of the test set:
First run the training process for new embeddings:

`python experiments/train/train_embeddings.py --config configs/cif_train.yaml`

Next run:

`python experiments/test/test_reconstruction.py --config configs/cif_eval.yaml`

### Sampling:

`python experiments/test/sampling.py --config configs/cif_eval.yaml`

### Interpolation:

`python experiments/test/interpolation.py --config configs/cif_eval.yaml`

### Coverage and MMD calculation:

`python experiments/test/metrics_eval.py --config configs/cif_eval.yaml`

## Citation
```
@article{stypulkowski2019cif,
  title={Conditional Invertible Flow for Point Cloud Generation},
  author={Stypu{\l}kowski, Micha{\l} and Zamorski, Maciej and Zi{\k{e}}ba, Maciej and Chorowski, Jan},
  journal={arXiv},
  year={2019}
}
```
