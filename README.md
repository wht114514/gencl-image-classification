# gencl-image-classification
Research on Image Classification Algorithms with Lighting Robustness Based on Generative Contrastive Learning
# 基于生成式对比学习的光照鲁棒性图像分类算法研究

> **Research on Image Recognition Method under Complex Illumination Conditions Based on Deep Learning**
>
> 在 CIFAR-10 上研究复杂光照条件下的图像分类鲁棒性：以图像预处理 + 生成式数据增广的思路，缓解亮度扰动导致的性能下降。

<!-- 徽章区：上传后按需替换为你的用户名/仓库名 -->
<!--
![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
-->

---

## 目录

- [项目简介](#项目简介)
- [方法概览](#方法概览)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [实验细节](#实验细节)
- [局限与后续工作](#局限与后续工作)
- [参考文献](#参考文献)
- [许可证](#许可证)

---

## 项目简介

图像分类模型在理想光照条件下表现良好，但在真实场景中，光照是不可避免的干扰因素：过曝、欠曝、色温偏移都会显著降低识别准确率。本项目以 **CIFAR-10** 为基准数据集，围绕"**如何提升模型在复杂光照条件下的鲁棒性**"这一问题，比较了三条技术路线：

1. **图像预处理路线** —— 直方图均衡化、RGB → HSV 色彩空间变换
2. **模型结构路线** —— 从线性神经网络升级到残差网络 ResNet18
3. **生成式数据增广路线** —— 用 DCGAN 生成不同亮度条件下的样本，扩充训练集

三条路线逐层递进：预处理解决"输入质量"，模型结构解决"特征表达能力"，生成式增广解决"训练分布覆盖不足"。项目最终给出了一组在亮度扰动下仍能保持稳定的分类方案。

> **一句话结论：** 在 CIFAR-10 上，仅靠预处理与换模型只能带来小幅提升；**用生成式方法主动扩充不同光照条件下的样本，是提升光照鲁棒性更有效的手段。**

---

## 方法概览

### 1. 数据与光照扰动建模

- 数据集：**CIFAR-10**（10 类，60,000 张 32×32 彩色图，50,000 训练 / 10,000 测试）
- 光照扰动方式：
  - **直方图均衡化** —— 重新分配灰度分布，增强对比度
  - **HSV 色彩空间变换** —— 在亮度（V）通道上解耦光照与颜色信息
  - **ColorJitter 亮度参数扰动** —— 在训练时对亮度做随机增减，模拟真实光照波动

### 2. 分类模型

| 模型 | 作用 |
| --- | --- |
| 线性神经网络 | 作为基线（baseline），验证预处理本身的效果 |
| **ResNet18** | 主力模型，残差连接缓解深层网络的退化问题 |
| DCGAN（深度全卷积生成对抗网络） | 生成器 + 判别器对抗训练，产出指定亮度条件下的新样本 |

### 3. 生成式增广流程

```text
原始 CIFAR-10
      │
      ├──► 按类别拆分（鸟 / 车 / 猫 / …）
      │
      ├──► DCGAN 训练：学习该类别在不同亮度下的分布
      │
      ├──► 生成指定亮度（如 ±20%）的合成样本
      │
      └──► 合成样本 + 原始样本 ──► 混合数据集 ──► 训练分类器 ──► 评估
```

---

## 实验结果

> 以下数字来自课题结题报告中的实验记录，用于说明各方法的相对效果。**请以本仓库代码实际复现的结果为准**，如与下表不一致，请更新此表。

### 线性神经网络：GAN 数据增广的效果

| 设置 | 汽车类别准确率 |
| --- | --- |
| 原始 CIFAR-10 | 55.5% |
| 原始 + GAN 生成样本 | **75.7%** |

### ResNet18：逐步引入光照扰动与生成样本

| 设置 | 准确率 |
| --- | --- |
| ResNet18 · 原始数据集 | 84% |
| ResNet18 · 训练/测试亮度增强 | 85% |
| ResNet18 · GAN 生成样本（鸟/车/猫等）+ 原数据 + 亮度 ±20% | **86%** |

**观察：**

- 从线性网络到 ResNet18，准确率有**较大幅度**提升，说明特征提取能力是基础；
- 仅靠亮度增强（预处理/增广变换）带来的提升**有限**（84% → 85%）；
- 引入 GAN 生成的亮度样本后进一步提升到 **86%**，且**在训练集亮度被降低 20% 时，模型仍能保持稳定**——这正是鲁棒性的直接证据。

---

## 项目结构

<!-- ===== 代码放进来之后，把下面这棵树替换成真实结构 ===== -->

```text
.
├── README.md
├── requirements.txt
├── （待补充：主训练脚本）
├── （待补充：模型定义）
├── （待补充：数据加载与光照扰动）
├── （待补充：DCGAN 生成模块）
├── （待补充：评估脚本）
└── （待补充：结果与图表输出目录）
```

<!-- 建议按功能拆分为以下模块，命名仅供参照：
     data/        数据集加载、预处理、光照扰动
     models/      线性网络、ResNet18、DCGAN
     train.py     分类器训练入口
     generate.py  DCGAN 生成样本
     evaluate.py  准确率评估与对比
     configs/     超参数配置
     results/     日志、曲线、混淆矩阵
-->

---

## 环境依赖

<!-- ===== 请根据本机实际环境补全版本号 ===== -->

- Python **×.×**
- PyTorch **×.×** ／ torchvision
- NumPy、Pillow、OpenCV（图像预处理）
- Matplotlib、tqdm（训练可视化与进度）
- 运行环境：<!-- CPU / GPU 型号、显存 -->

```bash
# 待补充：虚拟环境创建与依赖安装
pip install -r requirements.txt
```

---

## 快速开始

<!-- ===== 以下命令待代码上传后补全 ===== -->

### 1. 准备数据

```bash
# 待补充：CIFAR-10 下载 / 目录约定
# 默认应自动下载到 ./data ，请说明是否需要手动放置
```

### 2. 训练分类器

```bash
# 待补充：训练入口
# 预期参数：--model {linear,resnet18}  --epochs  --batch-size  --lr  --brightness
```

### 3. 训练 DCGAN 并生成增广样本

```bash
# 待补充：GAN 训练与样本生成命令
```

### 4. 评估与对比

```bash
# 待补充：评估命令
# 输出应包含：各设置下的测试准确率、亮度扰动下的准确率曲线
```

---

## 实验细节

<!-- ===== 以下为待填写项，请按实际代码填写，勿凭印象 ===== -->

| 项目 | 设置 |
| --- | --- |
| 数据集 | CIFAR-10（50k 训练 / 10k 测试） |
| 输入尺寸 | 32 × 32 |
| 分类模型 | 线性神经网络 / ResNet18 |
| 生成模型 | DCGAN（深度全卷积 GAN） |
| 光照扰动 | 直方图均衡化 / HSV 变换 / ColorJitter 亮度参数 |
| 亮度扰动幅度 | ±20%（待确认其他档位） |
| 优化器 | <!-- 待补充：SGD / Adam，学习率，weight decay --> |
| 批大小 | <!-- 待补充 --> |
| 训练轮数 | <!-- 待补充 --> |
| 随机种子 | <!-- 待补充：固定种子以保证可复现 --> |
| 评价指标 | 分类准确率（Accuracy） |
| 硬件 | <!-- 待补充 --> |

> **可复现性提示：** 若已固定随机种子，请在此处写明所用种子，并在 README 中说明"重复运行可得到一致结果"。这会让仓库的专业度显著提升。

---

## 局限与后续工作

本项目在结题时已识别出以下局限：

- **GAN 生成样本的质量与多样性有限**，对识别准确率的贡献没有达到预期；
- 生成样本与真实样本之间存在**分布差异**，直接混合训练可能引入噪声；
- CycleGAN 等更先进的生成模型在本课题开展时**可用资料较少**，未能充分验证；
- 实验主要在 CIFAR-10 这一小尺寸数据集上完成，**向真实场景图像迁移的效果未知**。

**后续方向：**

- [ ] 引入 CycleGAN / 更高质量的生成模型，做跨光照域的风格迁移
- [ ] 从"数据增广"转向"**表征学习**"——用对比学习目标让模型学到光照无关的特征表示
- [ ] 引入域适应（Domain Adaptation）方法，缩小不同光照域之间的特征分布差异
- [ ] 在 CIFAR-10-C（含各类腐蚀/扰动版本）等更具挑战性的基准上评测
- [ ] 与 RandAugment、CutMix 等主流增广策略做横向对比

---

## 参考文献

1. He K., Zhang X., Ren S., Sun J. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
2. Radford A., Metz L., Chintala S. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR, 2016.
3. Gonzalez R. C., Woods R. E. *Digital Image Processing*. 3rd Edition, Prentice Hall, 2009.
4. Kim Y. T. *Contrast Enhancement Using Brightness Preserving Bi-Histogram Equalization*. IEEE Trans. Consumer Electronics, 43(1), 1997.
5. Goodfellow I., et al. *Generative Adversarial Networks*. NeurIPS, 2014.
6. Hendrycks D., Dietterich T. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR, 2019.

---

## 许可证

<!-- 待确认：若用于公开项目，建议选用 MIT / Apache-2.0 -->
本项目采用 **MIT License**，详见 `LICENSE` 文件。

---


