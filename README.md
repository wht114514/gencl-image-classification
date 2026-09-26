# image-classification on CIFAR-10

> **Generative Contrastive Learning for Illumination-Robust Image Classification** · 基于生成式对比学习的光照鲁棒性图像分类算法研究

[English](#english) · [中文](#中文)

---
# co-author
Wang Haotian **Northwest A&F University**
Liu Xiang  **Qingdao University of Technology**
<a name="english"></a>

## English

### Overview

This project studies a practical and fundamental problem in image classification: **how does illumination variation (over-bright / under-dark) degrade recognition accuracy, and how can we recover it?**

We investigate three families of methods, reproduced with PyTorch on CIFAR-10:

1. **Linear network + histogram equalization** — a classical image-preprocessing baseline.
2. **ResNet-18 + HSV / brightness / histogram pre-processing** — deep CNN under illumination shifts.
3. **DCGAN-based data augmentation** — the core contribution: *generate brightness-augmented images to expand the training set* ("generative contrastive learning").

> This repository is both a reproducible experiment collection and a learning-oriented project. Each script corresponds to one verifiable scientific conclusion.

### Key Experiments & Results

#### 1. Linear Network — Illumination Sensitivity

| Condition | Accuracy |
|-----------|----------|
| Normal brightness | 37% |
| Brightness ×60% | 38% |
| Brightness ×140% | 36% |
| + Histogram equalization | improved |

A linear network stays near its ceiling (~35–38%) regardless of illumination, and histogram equalization yields a modest improvement.

#### 2. ResNet-18 — Illumination Robustness

| Condition | Accuracy |
|-----------|----------|
| Normal | 88.92% |
| Brightness ×40% | 80.97% |
| Brightness ×160% | 82.46% |
| HSV color space | 82.46% |
| HSV + brightness shift (×75% / ×125%) | significant drop |
| + Histogram equalization | improved |

#### 3. DCGAN Data Augmentation — the Core Contribution

| Experiment | Accuracy |
|------------|----------|
| "car" class, linear net (baseline) | 55.5% |
| "car" class, linear net + GAN-generated cars | **75.7%** |
| ResNet-18 (baseline) | 84% |
| ResNet-18 + brightness-augmented data | 85% |
| ResNet-18 + GAN-generated (bird/car/cat etc.), brightness ±20% | **86%** |

**Conclusion:** DCGAN-generated brightness-augmented images effectively improve illumination robustness when the original data is scarce or lacks brightness diversity — the core idea behind "generative contrastive learning."

### Repository Structure

```
.
├── README.md                     # This document
├── linear_cifar.py               # Linear network baseline (~35% acc)
├── cnn_cifar.py                  # Simple CNN classification (~70%+ acc)
├── show_cifar.py                 # Dataset visualization
├── brightness_experiment.py      # Brightness perturbation (linear)
├── brightness_experiment_cnn.py  # Brightness perturbation (CNN)
├── hsv_experiment.py             # RGB vs HSV color space
├── dcgan_cifar.py                # DCGAN training
├── dcgan_generate.py             # DCGAN image sampling
├── cdgan_cifar.py                # Conditional DCGAN training
├── generate_and_compare.py       # GAN augmentation vs pure-real comparison
└── data/                         # CIFAR-10 (auto-downloaded)
```

### Environment

- Python 3.10+ (recommend 3.11 / 3.12)
- PyTorch 2.x (GPU), torchvision, matplotlib, numpy
- NVIDIA GPU (8GB VRAM is enough) or CPU

```bash
conda create -n gencl python=3.12
conda activate gencl

# GPU version (China mirror accelerated)
pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu130/
pip install matplotlib numpy
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())  # should be True
```

### Quick Start

```bash
python show_cifar.py          # 1. visualize the dataset
python linear_cifar.py        # 2. linear baseline (~35%)
python cnn_cifar.py           # 3. CNN (~70%+)
python brightness_experiment.py   # 4. brightness perturbation
python hsv_experiment.py      # 5. RGB vs HSV
python dcgan_cifar.py         # 6. train DCGAN
python generate_and_compare.py    # 7. GAN augmentation comparison
```

### Key Concepts Covered

- Linear model ceiling vs. CNN spatial features
- Distribution shift (illumination) and its effect on accuracy
- RGB vs. HSV representation
- GAN / DCGAN / conditional DCGAN (cDCGAN)
- Generative data augmentation ("generative contrastive learning")

---

<a name="中文"></a>

## 中文

### 项目简介

本项目研究图像分类中一个实用而基础的问题：**光照变化（过亮/过暗）如何降低识别准确率，以及如何恢复它。**

我们用 PyTorch 在 CIFAR-10 上复现了三类方法：

1. **线性网络 + 直方图均衡化** —— 经典的图像预处理基线。
2. **ResNet-18 + HSV / 亮度 / 直方图预处理** —— 光照偏移下的深度卷积网络。
3. **基于 DCGAN 的数据增广** —— 核心贡献：*生成亮度增广图片来扩充训练集*（"生成式对比学习"）。

> 本仓库既是一个可复现的实验合集，也是一个面向学习的项目。每个脚本都对应一个可验证的科学结论。

### 核心实验与结果

#### 1. 线性网络 —— 光照敏感性

| 条件 | 准确率 |
|------|--------|
| 正常亮度 | 37% |
| 亮度 ×60% | 38% |
| 亮度 ×140% | 36% |
| + 直方图均衡化 | 有提升 |

线性网络受限于自身天花板（约 35~38%），光照偏移对其影响不大，直方图均衡化可带来小幅提升。

#### 2. ResNet-18 —— 光照鲁棒性

| 条件 | 准确率 |
|------|--------|
| 正常 | 88.92% |
| 亮度 ×40% | 80.97% |
| 亮度 ×160% | 82.46% |
| HSV 颜色空间 | 82.46% |
| HSV + 亮度偏移（×75% / ×125%） | 明显下降 |
| + 直方图均衡化 | 有提升 |

#### 3. DCGAN 数据增广 —— 核心贡献

| 实验 | 准确率 |
|------|--------|
| "汽车"类，线性网络（基线） | 55.5% |
| "汽车"类，线性网络 + GAN 生成汽车 | **75.7%** |
| ResNet-18（基线） | 84% |
| ResNet-18 + 亮度增广数据 | 85% |
| ResNet-18 + GAN 生成（鸟/车/猫等），亮度 ±20% | **86%** |

**结论：** 当原始数据稀缺或缺乏亮度多样性时，DCGAN 生成的亮度增广图片能有效提升光照鲁棒性——这正是"生成式对比学习"的核心思想。

### 目录结构

```
.
├── README.md                     # 本文档
├── linear_cifar.py               # 线性网络基线（约 35% 准确率）
├── cnn_cifar.py                  # 简单 CNN 分类（约 70%+ 准确率）
├── show_cifar.py                 # 数据集可视化
├── brightness_experiment.py      # 亮度扰动实验（线性）
├── brightness_experiment_cnn.py  # 亮度扰动实验（CNN）
├── hsv_experiment.py             # RGB vs HSV 颜色空间
├── dcgan_cifar.py                # DCGAN 训练
├── dcgan_generate.py             # DCGAN 图片采样
├── cdgan_cifar.py                # 条件 DCGAN 训练
├── generate_and_compare.py       # GAN 增广 vs 纯真实数据对比
└── data/                         # CIFAR-10（自动下载）
```

### 环境依赖

- Python 3.10+（推荐 3.11 / 3.12）
- PyTorch 2.x（GPU）、torchvision、matplotlib、numpy
- NVIDIA GPU（8GB 显存即可）或 CPU

```bash
conda create -n gencl python=3.12
conda activate gencl

# GPU 版（国内可用镜像加速）
pip install torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu130/
pip install matplotlib numpy
```

验证 GPU：

```python
import torch
print(torch.cuda.is_available())  # 应为 True
```

### 快速开始

```bash
python show_cifar.py          # 1. 可视化数据集
python linear_cifar.py        # 2. 线性基线（约 35%）
python cnn_cifar.py           # 3. CNN（约 70%+）
python brightness_experiment.py   # 4. 亮度扰动实验
python hsv_experiment.py      # 5. RGB vs HSV
python dcgan_cifar.py         # 6. 训练 DCGAN
python generate_and_compare.py    # 7. GAN 增广对比
```

### 涵盖的核心概念

- 线性模型天花板 vs. CNN 空间特征
- 分布偏移（光照）及其对准确率的影响
- RGB vs. HSV 表示
- GAN / DCGAN / 条件 DCGAN（cDCGAN）
- 生成式数据增广（"生成式对比学习"）

---

## Citation / 参考文献

本项目基于以下论文的实验思路（犀牛鸟中学科学人才培养计划科研实践结题报告）：

> 刘翔, 王浩田. 基于生成式对比学习的光照鲁棒性图像分类算法研究. 北京师范大学庆阳附属学校, 2022.

## License

MIT License
