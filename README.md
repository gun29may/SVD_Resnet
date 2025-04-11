# Singular Value Decomposition (SVD) Compressed ResNet Architectures

This project explores the use of Singular Value Decomposition (SVD) to compress Convolutional Neural Networks (CNNs), specifically ResNet architectures, to reduce their computational cost and memory footprint while aiming to maintain accuracy.

## Project Overview

The core idea is to apply SVD to the weight matrices of convolutional layers within a ResNet model.  By decomposing the weight matrix and retaining only the most significant singular values and vectors, we can approximate the original convolution operation with a reduced number of parameters. This effectively replaces a single convolutional layer with two smaller ones, potentially accelerating inference and reducing model size.

The project provides implementations for:

*   **Baseline ResNet models:** Standard ResNet architectures (e.g., ResNet-18, ResNet-34) for comparison.
*   **SVD-compressed ResNet variants:**  Modified ResNet models where convolutional layers are replaced with their SVD-decomposed equivalents.  The compression factor (rank) is a tunable parameter.
*   **Training and evaluation scripts:** Tools to train the models on datasets like CIFAR-10 and ImageNet, and to evaluate their performance (accuracy, inference speed, model size).
*   **Comparison scripts:**  Scripts to compare the performance of baseline and compressed models.

## Files

*   `cifar10.py`:  Code related to CIFAR-10 dataset loading and preprocessing.
*   `compare.py`: Script to compare baseline and compressed models (likely deprecated, see `compare_new.py`).
*   `compare_new.py`:  Improved script for comparing model performance.
*   `conv.py`: Contains the implementation of the SVD-compressed convolutional layer (`LowRankConv2d`).
*   `new_cifar.py`:  Training and evaluation script for models on CIFAR-10.
*   `new_imagenet.py`: Training and evaluation script for models on ImageNet (currently unfinished).
*   `new_res.py`:  Definitions for the modified ResNet architectures incorporating SVD compression.
*   `norm_conv.py`: An alternative (likely unused) implementation of a compressed convolutional layer.
*   `res_normal.py`:  Definitions for the baseline (uncompressed) ResNet architectures.
*   `test_compare.py`: Unit tests for the comparison scripts.
*   `test_res.py`: Unit tests for the ResNet model definitions.

## Usage

1.  **Install dependencies:**  `pip install -r requirements.txt` (Note: You may need to create this file listing required packages like PyTorch, torchvision, etc.)
2.  **Train models:** Use `new_cifar.py` (or `new_imagenet.py` when completed) to train baseline and compressed models with different rank parameters. For example: `python new_cifar.py --model resnet18 --rank 4`
3.  **Evaluate and compare:**  Use `compare_new.py` to evaluate the trained models and compare their accuracy, inference time, and model size.  For example:  `python compare_new.py --models resnet18_baseline resnet18_rank4 --dataset cifar10`  (Assuming you have saved the trained models as `resnet18_baseline.pth` and `resnet18_rank4.pth`).

**Note:** This project is under development.  Some scripts may be incomplete or require further refinement.  Contributions and feedback are welcome.
