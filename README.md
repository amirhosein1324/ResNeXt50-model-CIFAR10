# ResNeXt-50 Implementation for CIFAR-10

This repository provides a complete, from-scratch PyTorch implementation of the **ResNeXt-50** architecture. The model is configured specifically for the **CIFAR-10** dataset, demonstrating the power of **aggregated residual transformations**.

## 🚀 Project Overview

ResNeXt is a simple, highly modularized network architecture for image classification. Its design results in a homogeneous, multi-branch architecture that has only a few hyperparameters to set.

This project implements the **$32 \times 4d$** template (32 paths of width 4), which has been shown to outperform standard ResNet architectures of similar capacity.

## 🏗️ Architecture Details

## The Aggregated Transformation

Unlike standard ResNets, which use a single transformation path, ResNeXt utilizes a **split–transform–merge** strategy within its bottleneck blocks:

- **Split**: The input is split into low-dimensional embeddings.  
- **Transform**: Each branch performs a convolution (implemented using grouped convolutions).  
- **Merge**: The outputs are concatenated and added back to the identity shortcut.

## Model Parameters

- **Cardinality ($C$)**: 32 — size of the set of transformations  
- **Base Width**: $4d$ — dimensionality of the bottleneck  
- **Layers**: 50 weighted layers (3–4–6–3 block distribution)

## Repository Structure

- **ResNeXt_50.ipynb**: Main Jupyter Notebook containing model definition, data loading, and training loops  
- **data/**: Directory where the CIFAR-10 dataset is automatically downloaded  
- **README.md**: Project documentation  

## Requirements & Setup

For optimal training speed, use a GPU environment like Google Colab with T4 GPU.

```bash
pip install torch  , torchvision , tqdm , numpy 
```

### Training Configuration:

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| Epochs | 3 | Short run for demonstration |
| Batch Size | 32 | Optimized for standard GPU memory |
| Learning Rate | 0.01 | Initial learning rate with SGD |
| Optimizer | SGD | Stochastic Gradient Descent with 0.9 momentum |
| Weight Decay | $5 \times 10^{-4}$ | L2 regularization to prevent overfitting |
| Loss | CrossEntropy | Standard loss for multi-class classification |


## Performance Benchmarks

In a quick 3-epoch run, the model achieves:

Final Training Loss: ~0.7931

Test Accuracy: 76.32%

Note: With 100+ epochs and a learning rate scheduler, this architecture typically reaches >94% accuracy on CIFAR-10.

---

## Training Details

- Device Selection: Automatically switches between CPU and CUDA if available

- Loss Function: CrossEntropyLoss

- Optimizer: SGD with momentum and weight decay

- Progress Tracking: tqdm progress bar for batch-level feedback

Training and evaluation are performed at the end of each epoch to monitor generalization.


## Implementation Notes

This implementation closely follows the original ResNeXt design principles while adapting the architecture for CIFAR-10:

- Uses grouped convolutions to implement aggregated transformations

- Bottleneck design: 1×1 → 3×3 (groups) → 1×1

- Identity shortcuts are adjusted using 1×1 convolutions when dimensions change

- Adaptive average pooling ensures input-size flexibility

The model is implemented from scratch in PyTorch, without relying on torchvision’s pretrained ResNeXt.


## Key Takeaways

- Increasing cardinality can improve representational power without significantly increasing depth
- Grouped convolutions offer a strong accuracy–efficiency tradeoff
-Even short training runs demonstrate ResNeXt’s superiority over standard ResNet variants

## Future Improvements

- To further boost performance beyond the current 76%:

- Data Augmentation: Add RandomCrop and RandomHorizontalFlip to training transforms

- Longer Training: Increase epochs to 100–200 for better convergence

 Label Smoothing: Apply label smoothing to the CrossEntropy loss for improved generalization

---

THE FULL PAPER : 
