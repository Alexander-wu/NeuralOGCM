# NeuralOGCM: Differentiable Ocean Modeling with Learnable Physics

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2502.00338)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of the paper **"NeuralOGCM: Differentiable Ocean Modeling with Learnable Physics"**.

**NeuralOGCM** is a novel hybrid ocean general circulation model (OGCM) that bridges the gap between traditional numerical models and pure AI approaches. By integrating a **Differentiable Physics Core** with a **Deep Learning Corrector**, it achieves high-fidelity simulation with the speed of AI models while maintaining long-term physical consistency.

## 🌟 Key Features

* **Hybrid Architecture:** Combines a differentiable dynamical solver (based on primitive equations) with a Spatio-temporal Evolution (STE) neural network.
* **Learnable Physics:** Key physical parameters, such as diffusion coefficients ($\nu$), are optimized end-to-end from data, allowing the model to autonomously tune its physics core.
* **High Performance:** Significantly outperforms pure AI baselines (e.g., FourCastNet, SimVP) in accuracy and stability while being orders of magnitude faster than traditional GCMs.
* **Distributed Training:** Built on PyTorch DDP (Distributed Data Parallel) for efficient training on multi-GPU clusters.

## 📂 Repository Structure

```text
/NeuralOGCM_ocean/clean_code/
├── checkpoints/       # Saved model weights
├── logs/              # Training logs
├── model/             # Model definitions
│   ├── ocean_clean.py # NeuralOGCM model
│   ├── UNet.py        # Baseline: U-Net
│   ├── ConvLSTM.py    # Baseline: ConvLSTM
│   ├── SimVP.py       # Baseline: SimVP
│   └── Fourcastnet.py # Baseline: FourCastNet
├── Dataloader.py      # Data loading and preprocessing logic
├── train.py           # Unified distributed training script
└── untils/            # Utility functions
```

## 🛠️ Installation

1. **Clone the repository:**

   ```
   git clone [https://github.com/your_username/NeuralOGCM.git](https://github.com/your_username/NeuralOGCM.git)
   cd NeuralOGCM/clean_code
   ```

2. Environment Setup:
   The code requires Python 3.8+ and PyTorch. We recommend PyTorch 1.10+ to support bfloat16 mixed precision and dist.ReduceOp.AVG.
   ```
   pip install torch numpy h5py tqdm
   ```

